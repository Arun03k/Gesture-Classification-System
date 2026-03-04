import os
import sys
import argparse
import pathlib
import numpy as np
import pandas as pd
from collections import Counter
# Example parameters:
#   --input_frames_csv=demo_data/demo_video_rotate_frames.csv
#   --output_csv_name=demo_video_rotate_predicted_events.csv


# ─────────────────────────────────────────────────────────────────────────────
#  Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR    = PROJECT_ROOT / "data" / "processed"

WINDOW_SIZE  = 18          # frames per window (0.6 s @ 30 FPS)
HISTORY_LEN  = 5           # majority-vote over last N predictions
MIN_CONF     = 0.6         # minimum softmax confidence to accept non-idle
MIN_CONSEC   = 5           # debounce: require N consecutive non-idle windows

UPPER_BODY_PARTS = [
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
    "left_pinky",    "right_pinky",
    "left_index",    "right_index",
    "left_thumb",    "right_thumb",
    "left_hip",      "right_hip",
    "nose",
]
SUFFIXES = ["_x", "_y", "_z"]

# Map short model labels → full ground-truth label names used by calculator.py
LABEL_TO_FULL = {
    "idle":  "idle",
    "sl":    "swipe_left",
    "sr":    "swipe_right",
    "r_cw":  "rotate_clockwise",
    "r_ccw": "rotate_anticlockwise",
    "sd":    "swipe_down",
    "su":    "swipe_up",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Minimal neural network (NumPy only — no frameworks)
# ─────────────────────────────────────────────────────────────────────────────
def _relu(x):
    return np.maximum(0.0, x)


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def forward_pass(X, weights, biases):
    """Forward pass: ReLU hidden layers + softmax output."""
    a = np.atleast_2d(X)
    for W, b in zip(weights[:-1], biases[:-1]):
        a = _relu(a @ W + b)
    return _softmax(a @ weights[-1] + biases[-1])


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing (mirrors the notebook pipeline exactly)
# ─────────────────────────────────────────────────────────────────────────────
def normalize_chest_centered(df):
    """Translate all keypoints so shoulder midpoint = origin; scale by shoulder width."""
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
        if col.endswith("_x"):   df[col] = (df[col] - cx) / sw
        elif col.endswith("_y"): df[col] = (df[col] - cy) / sw
        elif col.endswith("_z"): df[col] = (df[col] - cz) / sw

    return df


def extract_features(df):
    """
    Chest-centred normalisation → feature selection → velocity.
    Returns feature array shape (N, 90).
    """
    df = normalize_chest_centered(df)
    df = df.fillna(0.0)

    feat_cols = [p + s for p in UPPER_BODY_PARTS for s in SUFFIXES
                 if (p + s) in df.columns]
    pos = df[feat_cols].values.astype(np.float64)
    vel = np.vstack([np.zeros((1, pos.shape[1])), np.diff(pos, axis=0)])
    return np.hstack([pos, vel])  # (N, 90)


def make_windows(features, window_size=WINDOW_SIZE):
    """
    Slide a window over the feature sequence with stride=1.
    Returns (windows, center_indices).
    """
    n_frames, n_feats = features.shape
    mid = window_size // 2
    windows, centers = [], []

    for start in range(0, n_frames - window_size + 1):
        windows.append(features[start:start + window_size].flatten())
        centers.append(start + mid)

    if not windows:
        return np.empty((0, window_size * n_feats)), np.array([], dtype=int)
    return np.stack(windows), np.array(centers, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
#  Prediction with smoothing
# ─────────────────────────────────────────────────────────────────────────────
def predict_smoothed(windows, weights, biases, idx_to_label,
                     history_size=HISTORY_LEN, min_conf=MIN_CONF):
    """
    Run inference with majority-vote smoothing over a sliding history.
    Returns list of predicted short labels, one per window.
    """
    history = ["idle"] * history_size
    preds = []

    for i in range(len(windows)):
        probs = forward_pass(windows[i].reshape(1, -1), weights, biases)
        pred_idx = int(np.argmax(probs))
        conf = float(probs[0, pred_idx])

        label = idx_to_label.get(pred_idx, "idle")
        if conf < min_conf:
            label = "idle"

        history.pop(0)
        history.append(label)
        voted = Counter(history).most_common(1)[0][0]
        preds.append(voted)

    return preds


# ─────────────────────────────────────────────────────────────────────────────
#  Application class (replaces DemoApplication)
# ─────────────────────────────────────────────────────────────────────────────
class GestureApplication:

    def __init__(self):
        weights_path = MODEL_DIR / "model_weights.npz"
        scaler_path  = MODEL_DIR / "scaler_params.npz"
        labels_path  = MODEL_DIR / "label_mapping.npz"

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}\n"
                "Run the training notebook to save model artefacts first."
            )

        # Load model weights
        data = np.load(weights_path)
        n_layers = len([k for k in data.files if k.startswith("W")])
        self.weights = [data[f"W{i}"] for i in range(n_layers)]
        self.biases  = [data[f"b{i}"] for i in range(n_layers)]

        # Load scaler
        scaler = np.load(scaler_path)
        self.mean_ = scaler["mean"]
        self.std_  = scaler["std"].copy()
        self.std_[self.std_ == 0] = 1.0

        # Load label mapping
        lm = np.load(labels_path, allow_pickle=True)
        labels_arr  = lm["labels"].tolist()
        indices_arr = lm["indices"].tolist()
        self.idx_to_label = {int(i): str(l) for l, i in zip(labels_arr, indices_arr)}

        print(f"[GestureApplication] Loaded — {len(self.idx_to_label)} classes: "
              f"{list(self.idx_to_label.values())}")

    # make sure you simulate live prediction; this means that for each frame you must only
    # regard the data of the current frame or past frames, never future frames!
    def compute_events(self, frames):
        """
        Main entry point.
        `frames` is the full pose DataFrame (index = timestamp).
        Returns a list of event labels, one per frame.
        Each gesture fires EXACTLY once per gesture segment; all other frames are 'idle'.
        """
        n_total = len(frames)
        df = frames.reset_index()

        # Extract features and create windows
        features = extract_features(df)

        if len(features) < WINDOW_SIZE:
            print(f"[Warning] Only {len(features)} frames — returning all idle.")
            return ["idle"] * n_total

        windows, center_indices = make_windows(features, WINDOW_SIZE)

        # Standardise
        windows_std = (windows - self.mean_) / self.std_

        # Predict with majority-vote smoothing
        smoothed = predict_smoothed(
            windows_std, self.weights, self.biases,
            self.idx_to_label,
            history_size=HISTORY_LEN,
            min_conf=MIN_CONF,
        )

        # Debounce at the window level: require MIN_CONSEC consecutive
        # non-idle window predictions before confirming a gesture.
        # This prevents early firing from partial window overlaps.
        consecutive = 0
        last_gesture = "idle"
        for i in range(len(smoothed)):
            if smoothed[i] != "idle":
                if smoothed[i] == last_gesture or last_gesture == "idle":
                    consecutive += 1
                else:
                    consecutive = 1
                last_gesture = smoothed[i]
                if consecutive < MIN_CONSEC:
                    smoothed[i] = "idle"
            else:
                consecutive = 0
                last_gesture = "idle"

        # Map window predictions to frame indices
        frame_labels = ["idle"] * n_total
        for ci, label in zip(center_indices, smoothed):
            if ci < n_total:
                frame_labels[ci] = label

        # Convert short labels → full names, fire exactly ONE event per gesture
        events = []
        last_event = "idle"
        for label in frame_labels:
            full = LABEL_TO_FULL.get(label, label)
            if full != "idle" and last_event == "idle":
                events.append(full)
                last_event = full
            else:
                events.append("idle")
                if full == "idle":
                    last_event = "idle"

        return events


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point  (matches professor's expected interface exactly)
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input_frames_csv",
                    help="CSV file containing the video transcription from MediaPipe",
                    required=True)
parser.add_argument("--output_csv_name",
                    help="output CSV file containing the events",
                    default="predicted_events.csv")

args = parser.parse_known_args()[0]

input_path = args.input_frames_csv

output_directory, input_csv_filename = os.path.split(args.input_frames_csv)
output_path = "%s/%s" % (output_directory, args.output_csv_name)

# Auto-detect timestamp column name (professor uses "timestamp", our data uses "time")
_peek = pd.read_csv(input_path, nrows=0)
ts_col = "timestamp" if "timestamp" in _peek.columns else "time"

frames = pd.read_csv(input_path, index_col=ts_col)
frames.index = frames.index.astype(int)
print(f"Loaded {len(frames)} frames from: {input_path}")

# ================================= your application =============================
my_model = GestureApplication()
# ================================================================================

# determine events
frames["events"] = my_model.compute_events(frames)

# the CSV has to have the columns "timestamp" and "events"
# but may also contain additional columns, which will be ignored during the score evaluation
frames["events"].to_csv(output_path, index=True) # since "timestamp" is the index, it will be saved also
print("events exported to %s" % output_path)

print("\nEvent distribution:")
print(frames["events"].value_counts().to_string())
