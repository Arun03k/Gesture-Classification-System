"""Real-time gesture recognition using MediaPipe pose estimation and a trained NumPy neural network.
Optionally sends gesture events to a slideshow server via HTTP.
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

from pipeline.gesture_pipeline import (
    UPPER_BODY_PARTS, SUFFIXES, LABEL_TO_FULL,
    MODEL_MANDATORY, MODEL_OPTIONALS, MODEL_GAME, AVAILABLE_MODELS,
    extract_features, forward_pass,
    load_model_artifacts, get_model_dir,
)

# Paths
SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
BASE_MODEL_DIR = SCRIPT_DIR / "data" / "processed"
KP_YAML      = SCRIPT_DIR / "notebooks" / "process_videos" / "keypoint_mapping.yml"

SLIDESHOW_URL = "http://127.0.0.1:8800/event"

# Pipeline constants (window/smoothing/debounce — tuned for live use)
WINDOW_SIZE  = 18       # frames per inference window
HISTORY_LEN  = 11       # majority-vote history length (wider = more stable)
MIN_CONF     = 0.80     # minimum softmax confidence to accept non-idle
MIN_CONSEC   = 8        # debounce: require N consecutive non-idle windows
BUFFER_SIZE  = WINDOW_SIZE * 3   # rolling frame buffer (ensures correct velocity)
COOLDOWN_SEC = 2.0      # seconds to wait before next gesture can fire
MIN_MARGIN   = 0.15     # top prediction must beat runner-up by this margin
IDLE_PRIME_FRAMES = 10  # require stable idle frames before any new gesture can trigger
OPPOSITE_LOCKOUT_SEC = 2.0  # block su right after sd (and vice versa)
MAX_IDLE_PROB_FOR_GESTURE = 0.30  # suppress non-idle if idle probability is still high

# Per-class confidence overrides
# su/sd: low threshold because direction verification is the real guard.
# sr: high threshold because small body sway mimics a rightward move.
PER_CLASS_MIN_CONF = {
    "su": 0.60,
    "sd": 0.62,
    "sr": 0.85,
}

# ── Wrist keypoint indices in the 45-column position feature vector ──
# UPPER_BODY_PARTS order: l_shoulder(0), r_shoulder(1), l_elbow(2), r_elbow(3),
#   l_wrist(4), r_wrist(5), ...   each × 3 axes (x=0, y=1, z=2)
_LW_X_IDX = 12   # left_wrist_x   = 4 × 3 + 0
_LW_Y_IDX = 13   # left_wrist_y   = 4 × 3 + 1
_RW_X_IDX = 15   # right_wrist_x  = 5 × 3 + 0
_RW_Y_IDX = 16   # right_wrist_y  = 5 × 3 + 1
_MIN_VERT_DISP  = 0.10  # minimum |wrist-Y net displacement| for su/sd
_MIN_HORIZ_DISP = 0.12  # minimum |wrist-X net displacement| for sl/sr
_VERT_DOM_RATIO = 1.35  # require |Y| to dominate |X| for vertical swipes
_MAX_SWIPE_CIRCULARITY = 2.2  # reject su/sd if wrist trajectory looks circular
# Absolute wrist-height gates (chest-centered coords) to ensure full strokes.
# These gates block "prepare up before down" from firing su.
_SU_START_MIN_Y = 0.10   # su should start below shoulder line
_SU_END_MAX_Y = -0.02    # su should end above shoulder line
_SD_START_MAX_Y = -0.06   # sd should start above shoulder line
_SD_END_MIN_Y = 0.06      # sd should end below shoulder line

# Short model label → slideshow command name used by client.js
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

# Column name loader
def load_column_names(yaml_path: pathlib.Path) -> list:
    """Read keypoint_mapping.yml and return ordered column name list."""
    with open(yaml_path, "r") as f:
        m = yaml.safe_load(f)
    all_kp = m["face"] + m["body"]
    return [f"{kp}_{d}" for kp in all_kp for d in ("x", "y", "z", "confidence")]


# Live gesture recogniser class
class LiveGestureRecogniser:
    def __init__(self, camera_index: int = 0, flip: bool = False,
                 slideshow: bool = False, model_name: str = MODEL_MANDATORY):
        self.camera_index = camera_index
        self.flip         = flip
        self.slideshow    = slideshow
        self.model_name   = model_name

        self._load_model()
        self._col_names = load_column_names(KP_YAML)

        # Rolling buffer: stores raw DataFrame rows (dicts)
        self._buffer: collections.deque = collections.deque(maxlen=BUFFER_SIZE)

        # Prediction smoothing
        self._history: list = ["idle"] * HISTORY_LEN
        self._consec:  int  = 0
        self._last_gesture  = "idle"
        self._idle_frames: int = IDLE_PRIME_FRAMES
        self._last_su_fired: float = 0.0
        self._last_sd_fired: float = 0.0
        # Display state
        self._current_label:  str   = "idle"
        self._display_label:  str   = "idle"
        self._display_until:  float = 0.0    # timestamp to keep label on screen

        print("[LiveGestureRecogniser] Ready.")
        print(f"  Model  : {model_name}")
        print(f"  Camera : {camera_index}  |  Flip: {flip}  |  Slideshow: {slideshow}")
        print(f"  Classes: {list(self.idx_to_label.values())}")
        if slideshow:
            print(f"  Slideshow URL: {SLIDESHOW_URL}")

    # Model loading
    def _load_model(self):
        model_dir = get_model_dir(BASE_MODEL_DIR, self.model_name)
        self.weights, self.biases, self.mean_, self.std_, self.idx_to_label = \
            load_model_artifacts(model_dir)
        self.label_to_idx = {v: k for k, v in self.idx_to_label.items()}

    # Append one MediaPipe frame to the rolling buffer
    def _add_frame(self, landmarks) -> None:
        """Build a keypoint row from MediaPipe landmarks and append to the buffer."""
        row = {}
        joint_names_flat = [c[:-2] for c in self._col_names[::4]]  # _x columns → strip suffix
        for ji, kp_name in enumerate(joint_names_flat):
            lmp = landmarks.landmark[ji]
            row[f"{kp_name}_x"]          = lmp.x
            row[f"{kp_name}_y"]          = lmp.y
            row[f"{kp_name}_z"]          = lmp.z
            row[f"{kp_name}_confidence"] = lmp.visibility
        self._buffer.append(row)

    # Run inference on the current buffer window
    def _infer(self) -> str:
        if len(self._buffer) < WINDOW_SIZE:
            return "idle"

        # Build DataFrame from buffer, extract features
        df = pd.DataFrame(list(self._buffer))

        features = extract_features(df)
        window_feat = features[-WINDOW_SIZE:]  # last WINDOW_SIZE rows  shape (18, 90)

        # Net displacement: final position minus initial position over the window.
        # This is the primary discriminator for su/sd/r_cw:
        #   su  → wrist y net negative (wrist rose)
        #   sd  → wrist y net positive (wrist fell)
        #   r_cw → wrist net ≈ 0 (circular, returns to start)
        n_pos = window_feat.shape[1] // 2   # 45 position features
        net_disp = window_feat[-1, :n_pos] - window_feat[0, :n_pos]  # shape (45,)
        window_vec = np.concatenate([window_feat.flatten(), net_disp]).reshape(1, -1)

        # Standardise
        window_std  = (window_vec - self.mean_) / self.std_

        # Forward pass
        probs    = forward_pass(window_std, self.weights, self.biases)
        pred_idx = int(np.argmax(probs))
        conf     = float(probs[0, pred_idx])

        raw_label = self.idx_to_label.get(pred_idx, "idle")
        # Apply per-class confidence threshold
        threshold = PER_CLASS_MIN_CONF.get(raw_label, MIN_CONF)
        if conf < threshold:
            raw_label = "idle"

        # If idle probability is still high, suppress non-idle prediction.
        if raw_label != "idle":
            idle_idx = self.label_to_idx.get("idle")
            if idle_idx is not None and float(probs[0, idle_idx]) > MAX_IDLE_PROB_FOR_GESTURE:
                raw_label = "idle"

        # Confidence margin: top prediction must clearly beat runner-up.
        # SKIPPED for su/sd — direction verification is the real safety gate
        # and the margin check was blocking valid weak-confidence su predictions.
        if raw_label != "idle" and raw_label not in ("su", "sd"):
            sorted_probs = np.sort(probs[0])[::-1]
            if sorted_probs[0] - sorted_probs[1] < MIN_MARGIN:
                raw_label = "idle"

        # ── Direction verification for su / sd ────────────────────────
        # Also runs when model predicts anything with conf >= 0.40 and
        # the physical check identifies it as a vertical gesture.
        # MediaPipe Y increases downward: su → negative Y disp, sd → positive.
        # Require Y displacement to DOMINATE X (no diagonal gestures).
        if raw_label in ("su", "sd"):
            lw_y = net_disp[_LW_Y_IDX]; lw_x = net_disp[_LW_X_IDX]
            rw_y = net_disp[_RW_Y_IDX]; rw_x = net_disp[_RW_X_IDX]
            # active hand = larger absolute Y movement
            if abs(lw_y) >= abs(rw_y):
                wrist_y_disp, wrist_x_disp = lw_y, lw_x
                wrist_xy = window_feat[:, [_LW_X_IDX, _LW_Y_IDX]]
            else:
                wrist_y_disp, wrist_x_disp = rw_y, rw_x
                wrist_xy = window_feat[:, [_RW_X_IDX, _RW_Y_IDX]]

            start_y = float(wrist_xy[0, 1])
            end_y = float(wrist_xy[-1, 1])

            # Circularity check: circular wrist tracks (high path/net ratio)
            # are usually rotations, not vertical swipes.
            path_len = np.linalg.norm(np.diff(wrist_xy, axis=0), axis=1).sum()
            net_len = np.linalg.norm(wrist_xy[-1] - wrist_xy[0])
            circularity = path_len / (net_len + 1e-6)

            if (
                abs(wrist_y_disp) < _MIN_VERT_DISP
                or abs(wrist_y_disp) <= _VERT_DOM_RATIO * abs(wrist_x_disp)
                or circularity > _MAX_SWIPE_CIRCULARITY
            ):
                raw_label = "idle"        # insufficient or diagonal motion
            elif wrist_y_disp < 0:
                raw_label = "su"          # upward (negative Y)
            else:
                raw_label = "sd"          # downward (positive Y)

            # Full-stroke gate: require meaningful start/end heights so prep
            # motion (lifting hand before a down swipe) does not fire su.
            if raw_label == "su" and not (start_y >= _SU_START_MIN_Y and end_y <= _SU_END_MAX_Y):
                raw_label = "idle"
            if raw_label == "sd" and not (start_y <= _SD_START_MAX_Y and end_y >= _SD_END_MIN_Y):
                raw_label = "idle"

            # Opposite lockout: block false opposite trigger from arm return.
            now = time.time()
            if raw_label == "sd" and (now - self._last_su_fired) < OPPOSITE_LOCKOUT_SEC:
                raw_label = "idle"
            if raw_label == "su" and (now - self._last_sd_fired) < OPPOSITE_LOCKOUT_SEC:
                raw_label = "idle"

        # ── Horizontal displacement check for sl / sr ─────────────────
        # Body sway and pose drift can look like a horizontal swipe.
        # Require: (a) minimum X displacement AND (b) X > Y displacement
        # so only deliberate horizontal arm movement passes.
        if raw_label in ("sl", "sr"):
            lw_x = net_disp[_LW_X_IDX]
            rw_x = net_disp[_RW_X_IDX]
            lw_y = net_disp[_LW_Y_IDX]
            rw_y = net_disp[_RW_Y_IDX]
            # active hand = larger horizontal movement
            if abs(lw_x) >= abs(rw_x):
                wrist_x_disp, wrist_y_disp = lw_x, lw_y
            else:
                wrist_x_disp, wrist_y_disp = rw_x, rw_y
            if abs(wrist_x_disp) < _MIN_HORIZ_DISP or abs(wrist_x_disp) <= abs(wrist_y_disp):
                raw_label = "idle"        # drift/diagonal — not a real swipe

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

        # Idle-prime gate: require a short stable idle period before allowing
        # any new gesture to fire, which blocks self-triggering from pose drift.
        if voted == "idle":
            self._idle_frames = min(IDLE_PRIME_FRAMES, self._idle_frames + 1)
        else:
            if self._idle_frames < IDLE_PRIME_FRAMES:
                voted = "idle"
                self._consec = 0
                self._last_gesture = "idle"
            else:
                self._idle_frames = 0

        return voted

    # Send HTTP event to the slideshow server
    def _send_event(self, gesture: str) -> None:
        try:
            requests.post(SLIDESHOW_URL, json={"command": gesture}, timeout=1.0)
            print(f"[slideshow] → {gesture}")
        except Exception as e:
            print(f"[slideshow] Failed to send '{gesture}': {e}")

    # Draw gesture label and buffer status on the video frame
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

    # Main capture-and-infer loop
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
        last_event_time  = 0.0

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

                    # Fire event once per gesture (rising edge + cooldown guard)
                    now = time.time()
                    if full_label != "idle" and last_event_label == "idle" and (now - last_event_time) >= COOLDOWN_SEC:
                        print(f"[GESTURE DETECTED] {full_label}  →  slideshow: {slideshow_cmd}")
                        if self.slideshow:
                            self._send_event(slideshow_cmd)
                        self._display_label = full_label
                        self._display_until = now + 2.0
                        last_event_time     = now
                        if predicted == "su":
                            self._last_su_fired = now
                        elif predicted == "sd":
                            self._last_sd_fired = now
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
    parser.add_argument(
        "--model", type=str, default=MODEL_MANDATORY,
        choices=AVAILABLE_MODELS,
        help=f"Which model to use for inference (default: {MODEL_MANDATORY}). "
             f"Options: {', '.join(AVAILABLE_MODELS)}"
    )
    args = parser.parse_args()

    recogniser = LiveGestureRecogniser(
        camera_index=args.camera,
        flip=args.flip,
        slideshow=args.slideshow,
        model_name=args.model,
    )
    recogniser.run()


