"""
live_gesture_recognition.py
===========================
Real-time gesture recognition using the trained model.

Captures from a webcam, runs MediaPipe pose detection, feeds a rolling
frame buffer into the trained neural network, and overlays the detected
gesture on the live video feed.

Optional: send detected gesture events to the slideshow server.

Usage
-----
    # Basic webcam test (camera 0)
    python live_gesture_recognition.py

    # Choose camera index
    python live_gesture_recognition.py --camera 1

    # Mirror / flip the image (useful for laptops)
    python live_gesture_recognition.py --flip

    # Also control the slideshow (make sure slideshow_server.py is running first)
    python live_gesture_recognition.py --slideshow

    # All options combined
    python live_gesture_recognition.py --camera 0 --flip --slideshow

Press  Q  or  ESC  to quit.
"""

import argparse
import collections
import pathlib
import time
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import requests
import yaml

# ─────────────────────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
MODEL_DIR    = SCRIPT_DIR / "data" / "processed"
KP_YAML      = SCRIPT_DIR / "process_videos" / "keypoint_mapping.yml"

SLIDESHOW_URL = "http://127.0.0.1:8800/event"

# ─────────────────────────────────────────────────────────────────────────────
#  Model / pipeline constants  (must match training / log_emitted_events_to_csv)
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE  = 18       # frames per inference window  (0.6 s @ 30 FPS)
HISTORY_LEN  = 5        # majority-vote history length
MIN_CONF     = 0.6      # minimum softmax confidence to accept non-idle
MIN_CONSEC   = 5        # debounce: require N consecutive non-idle windows
BUFFER_SIZE  = WINDOW_SIZE * 3   # rolling frame buffer (ensures correct velocity)

UPPER_BODY_PARTS = [
    "left_shoulder",  "right_shoulder",
    "left_elbow",     "right_elbow",
    "left_wrist",     "right_wrist",
    "left_pinky",     "right_pinky",
    "left_index",     "right_index",
    "left_thumb",     "right_thumb",
    "left_hip",       "right_hip",
    "nose",
]
SUFFIXES = ["_x", "_y", "_z"]

# Map short model labels → ground-truth full names (used for on-screen display)
LABEL_TO_FULL = {
    "idle":  "idle",
    "sl":    "swipe_left",
    "sr":    "swipe_right",
    "r_cw":  "rotate_clockwise",
    "r_ccw": "rotate_anticlockwise",
    "sd":    "swipe_down",
    "su":    "swipe_up",
}

# Map short model labels → slideshow command names (what client.js understands)
# Navigation: swipe_left = move LEFT through slides (Reveal.left)
#             swipe_right = move RIGHT through slides (Reveal.right)
# Image ops:  rotate (CW) / rotate_counter_clock (CCW)
LABEL_TO_SLIDESHOW = {
    "idle":  "idle",
    "sl":    "swipe_left",
    "sr":    "swipe_right",
    "r_cw":  "rotate",
    "r_ccw": "rotate_counter_clock",
    "sd":    "swipe_down",
    "su":    "swipe_up",
}

# Colour palette for overlay  (BGR)
COLOURS = {
    "idle":               (180, 180, 180),
    "swipe_left":         (0,   200, 255),
    "swipe_right":        (0,   255, 100),
    "rotate_clockwise":   (255, 180,   0),
    "rotate_anticlockwise":(255,  80, 200),
    "swipe_up":           (255, 255,   0),
    "swipe_down":         (100, 100, 255),
}

# Human-readable display names
DISPLAY_NAMES = {
    "idle":                "  idle",
    "swipe_left":          "← Swipe Left",
    "swipe_right":         "→ Swipe Right",
    "rotate_clockwise":    "↻ Rotate CW",
    "rotate_anticlockwise":"↺ Rotate CCW",
    "swipe_up":            "↑ Swipe Up",
    "swipe_down":          "↓ Swipe Down",
}

# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing (mirrors log_emitted_events_to_csv exactly)
# ─────────────────────────────────────────────────────────────────────────────
def normalize_chest_centered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cx = (df["left_shoulder_x"] + df["right_shoulder_x"]) / 2
    cy = (df["left_shoulder_y"] + df["right_shoulder_y"]) / 2
    cz = (df["left_shoulder_z"] + df["right_shoulder_z"]) / 2
    sw = np.sqrt(
        (df["left_shoulder_x"] - df["right_shoulder_x"])**2 +
        (df["left_shoulder_y"] - df["right_shoulder_y"])**2 +
        (df["left_shoulder_z"] - df["right_shoulder_z"])**2
    )
    sw = sw.replace(0, np.nan)
    coord_cols = [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]
    for col in coord_cols:
        if   col.endswith("_x"): df[col] = (df[col] - cx) / sw
        elif col.endswith("_y"): df[col] = (df[col] - cy) / sw
        elif col.endswith("_z"): df[col] = (df[col] - cz) / sw
    return df


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Returns shape (N, 90): chest-centred position + velocity."""
    df = normalize_chest_centered(df).fillna(0.0)
    feat_cols = [p + s for p in UPPER_BODY_PARTS for s in SUFFIXES if (p + s) in df.columns]
    pos = df[feat_cols].values.astype(np.float64)
    vel = np.vstack([np.zeros((1, pos.shape[1])), np.diff(pos, axis=0)])
    return np.hstack([pos, vel])   # (N, 90)


# ─────────────────────────────────────────────────────────────────────────────
#  Neural network forward pass (NumPy only)
# ─────────────────────────────────────────────────────────────────────────────
def _relu(x):    return np.maximum(0.0, x)
def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def forward_pass(X, weights, biases):
    a = np.atleast_2d(X)
    for W, b in zip(weights[:-1], biases[:-1]):
        a = _relu(a @ W + b)
    return _softmax(a @ weights[-1] + biases[-1])


# ─────────────────────────────────────────────────────────────────────────────
#  Build the column name list from keypoint_mapping.yml
# ─────────────────────────────────────────────────────────────────────────────
def load_column_names(yaml_path: pathlib.Path) -> list:
    with open(yaml_path, "r") as f:
        m = yaml.safe_load(f)
    all_kp = m["face"] + m["body"]
    return [f"{kp}_{d}" for kp in all_kp for d in ("x", "y", "z", "confidence")]


# ─────────────────────────────────────────────────────────────────────────────
#  Live Gesture Recogniser
# ─────────────────────────────────────────────────────────────────────────────
class LiveGestureRecogniser:
    def __init__(self, camera_index: int = 0, flip: bool = False, slideshow: bool = False):
        self.camera_index = camera_index
        self.flip         = flip
        self.slideshow    = slideshow

        self._load_model()
        self._col_names = load_column_names(KP_YAML)

        # Rolling buffer: stores raw DataFrame rows (dicts)
        self._buffer: collections.deque = collections.deque(maxlen=BUFFER_SIZE)

        # Prediction smoothing
        self._history: list = ["idle"] * HISTORY_LEN
        self._consec:  int  = 0
        self._last_gesture  = "idle"

        # Display state
        self._current_label:  str   = "idle"
        self._display_label:  str   = "idle"
        self._display_until:  float = 0.0    # timestamp to keep label on screen

        print("[LiveGestureRecogniser] Ready.")
        print(f"  Camera : {camera_index}  |  Flip: {flip}  |  Slideshow: {slideshow}")
        print(f"  Classes: {list(self.idx_to_label.values())}")
        if slideshow:
            print(f"  Slideshow URL: {SLIDESHOW_URL}")

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self):
        for p, name in [(MODEL_DIR / "model_weights.npz", "model_weights"),
                        (MODEL_DIR / "scaler_params.npz",  "scaler_params"),
                        (MODEL_DIR / "label_mapping.npz",  "label_mapping")]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing: {p}\n"
                    "Run the training notebook first to generate model artefacts."
                )

        data = np.load(MODEL_DIR / "model_weights.npz")
        n = len([k for k in data.files if k.startswith("W")])
        self.weights = [data[f"W{i}"] for i in range(n)]
        self.biases  = [data[f"b{i}"] for i in range(n)]

        scaler = np.load(MODEL_DIR / "scaler_params.npz")
        self.mean_ = scaler["mean"]
        self.std_  = scaler["std"].copy()
        self.std_[self.std_ == 0] = 1.0

        lm = np.load(MODEL_DIR / "label_mapping.npz", allow_pickle=True)
        self.idx_to_label = {
            int(i): str(l)
            for l, i in zip(lm["labels"].tolist(), lm["indices"].tolist())
        }

    # ── Add one frame to buffer ───────────────────────────────────────────────
    def _add_frame(self, landmarks) -> None:
        """Build one row dict from MediaPipe landmarks and push it to the buffer."""
        row = {}
        # self._col_names pattern: [kp_x, kp_y, kp_z, kp_confidence, ...] for each keypoint
        # [::4] gives only the _x columns; strip "_x" to get bare joint names in order.
        joint_names_flat = [c[:-2] for c in self._col_names[::4]]   # strip last 2 chars "_x"
        for ji, kp_name in enumerate(joint_names_flat):
            lmp = landmarks.landmark[ji]
            row[f"{kp_name}_x"]          = lmp.x
            row[f"{kp_name}_y"]          = lmp.y
            row[f"{kp_name}_z"]          = lmp.z
            row[f"{kp_name}_confidence"] = lmp.visibility
        self._buffer.append(row)

    # ── Run one inference step ─────────────────────────────────────────────────
    def _infer(self) -> str:
        if len(self._buffer) < WINDOW_SIZE:
            return "idle"

        # Build DataFrame from buffer, extract features
        df = pd.DataFrame(list(self._buffer))

        features = extract_features(df)        # (BUFFER_SIZE, 90)
        # Take the LAST WINDOW_SIZE feature rows → correct velocities everywhere
        window_feat = features[-WINDOW_SIZE:]  # (WINDOW_SIZE, 90)
        window_vec  = window_feat.flatten().reshape(1, -1)

        # Standardise
        window_std  = (window_vec - self.mean_) / self.std_

        # Forward pass
        probs    = forward_pass(window_std, self.weights, self.biases)
        pred_idx = int(np.argmax(probs))
        conf     = float(probs[0, pred_idx])

        raw_label = self.idx_to_label.get(pred_idx, "idle")
        if conf < MIN_CONF:
            raw_label = "idle"

        # Majority-vote smoothing
        self._history.pop(0)
        self._history.append(raw_label)
        voted = Counter(self._history).most_common(1)[0][0]

        # Debounce
        if voted != "idle":
            if voted == self._last_gesture or self._last_gesture == "idle":
                self._consec += 1
            else:
                self._consec = 1
            self._last_gesture = voted
            if self._consec < MIN_CONSEC:
                voted = "idle"
        else:
            self._consec       = 0
            self._last_gesture = "idle"

        return voted

    # ── Send event to slideshow ────────────────────────────────────────────────
    def _send_event(self, gesture: str) -> None:
        try:
            requests.post(SLIDESHOW_URL, json={"command": gesture}, timeout=1.0)
            print(f"[slideshow] → {gesture}")
        except Exception as e:
            print(f"[slideshow] Failed to send '{gesture}': {e}")

    # ── Draw HUD overlay on frame ──────────────────────────────────────────────
    def _draw_hud(self, image: np.ndarray, label: str, conf_bar: float) -> np.ndarray:
        h, w = image.shape[:2]
        colour = COLOURS.get(label, (200, 200, 200))
        text   = DISPLAY_NAMES.get(label, label)

        # Semi-transparent background bar
        overlay = image.copy()
        cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

        # Gesture label (big)
        cv2.putText(image, text, (20, h - 25),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, colour, 2, cv2.LINE_AA)

        # Buffer fill indicator (thin bar at top)
        filled = min(len(self._buffer) / WINDOW_SIZE, 1.0)
        cv2.rectangle(image, (0, 0), (int(w * filled), 6),
                      (0, 255, 180) if filled >= 1.0 else (120, 120, 120), -1)

        # Controls hint (small)
        cv2.putText(image, "Q/ESC to quit", (w - 160, h - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (150, 150, 150), 1, cv2.LINE_AA)
        return image

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self) -> None:
        mp_pose    = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mp_styles  = mp.solutions.drawing_styles

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self.camera_index}. "
                "Try --camera 1 (or another index) if you have multiple cameras."
            )

        print(f"\nCamera opened. Press Q or ESC to stop.\n")
        last_event_label = "idle"

        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    print("[Warning] Failed to read frame — retrying…")
                    continue

                if self.flip:
                    frame = cv2.flip(frame, 1)

                # MediaPipe expects RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = pose.process(rgb)
                rgb.flags.writeable = True
                display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # Draw skeleton
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        display,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                    )
                    # Add frame to buffer and infer
                    self._add_frame(results.pose_landmarks)
                    predicted = self._infer()

                    # Map short label to both display name and slideshow command
                    full_label     = LABEL_TO_FULL.get(predicted, predicted)
                    slideshow_cmd  = LABEL_TO_SLIDESHOW.get(predicted, predicted)

                    # Fire event once per gesture (rising edge)
                    if full_label != "idle" and last_event_label == "idle":
                        print(f"[GESTURE DETECTED] {full_label}  →  slideshow: {slideshow_cmd}")
                        if self.slideshow:
                            self._send_event(slideshow_cmd)
                        self._display_label = full_label
                        self._display_until = time.time() + 2.0   # show for 2 s
                    last_event_label = full_label if full_label != "idle" else "idle"

                else:
                    # No pose detected — show warning
                    cv2.putText(display, "No person detected", (20, 50),
                                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

                # Keep display label on screen for 2 seconds after gesture fires
                show_label = (self._display_label
                              if time.time() < self._display_until
                              else "idle")

                display = self._draw_hud(display, show_label, 0)
                cv2.imshow("Live Gesture Recognition — Press Q to quit", display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):   # Q or ESC
                    break

        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time gesture recognition via webcam"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Webcam index to use (default: 0)"
    )
    parser.add_argument(
        "--flip", action="store_true",
        help="Horizontally flip the webcam image (mirror mode)"
    )
    parser.add_argument(
        "--slideshow", action="store_true",
        help="Send detected gesture events to the slideshow server at "
             f"{SLIDESHOW_URL}"
    )
    args = parser.parse_args()

    recogniser = LiveGestureRecogniser(
        camera_index=args.camera,
        flip=args.flip,
        slideshow=args.slideshow,
    )
    recogniser.run()
