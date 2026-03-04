import os
import sys
import argparse
import pathlib
import numpy as np
import pandas as pd
from collections import Counter

# Add project root to path for pipeline imports
SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.gesture_pipeline import (  # noqa: E402
    LABEL_TO_FULL, TARGET_FPS,
    extract_features, forward_pass,
    subsample_to_fps,
    load_model_artifacts,
)

# Paths and tuning constants
MODEL_DIR   = PROJECT_ROOT / "data" / "processed"

WINDOW_SIZE = 18    # frames per window (0.6 s @ 30 FPS)
HISTORY_LEN = 5     # majority-vote history length
MIN_CONF    = 0.6   # minimum softmax confidence to accept non-idle
MIN_CONSEC  = 5     # debounce: require N consecutive non-idle windows


def make_windows(features, window_size=WINDOW_SIZE):
    """Slide a fixed-length window over the feature sequence (stride=1). Returns (windows, center_indices)."""
    n_frames, n_feats = features.shape
    mid = window_size // 2
    windows, centers = [], []

    for start in range(0, n_frames - window_size + 1):
        windows.append(features[start:start + window_size].flatten())
        centers.append(start + mid)

    if not windows:
        return np.empty((0, window_size * n_feats)), np.array([], dtype=int)
    return np.stack(windows), np.array(centers, dtype=int)


# Prediction with majority-vote smoothing
def predict_smoothed(windows, weights, biases, idx_to_label,
                     history_size=HISTORY_LEN, min_conf=MIN_CONF):
    """Run forward pass on each window and apply majority-vote smoothing. Returns one label per window."""
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


# Application class
class GestureApplication:

    def __init__(self):
        self.weights, self.biases, self.mean_, self.std_, self.idx_to_label = \
            load_model_artifacts(MODEL_DIR)
        print(f"[GestureApplication] Loaded — {len(self.idx_to_label)} classes: "
              f"{list(self.idx_to_label.values())}")

    def compute_events(self, frames):
        """Run inference on the full pose DataFrame and return one event label per frame.
        Each gesture fires exactly once per contiguous segment; all other frames are 'idle'.
        Only past and current frames are used at each step (causal, no lookahead).
        """
        n_total = len(frames)
        df = frames.reset_index().copy()  # .copy() defragments the DataFrame

        # ── FPS normalisation ──────────────────────────────────────────────
        # Model was trained on 30fps data; subsample_to_fps handles high-fps input.
        df_sub, sub_idx = subsample_to_fps(df, TARGET_FPS)

        # Extract features and create windows
        features = extract_features(df_sub)

        if len(features) < WINDOW_SIZE:
            print(f"[Warning] Only {len(features)} frames — returning all idle.")
            return ["idle"] * n_total

        windows, center_indices = make_windows(features, WINDOW_SIZE)
        center_indices_orig = sub_idx[center_indices]

        # Standardise
        windows_std = (windows - self.mean_) / self.std_

        # Predict with majority-vote smoothing
        smoothed = predict_smoothed(
            windows_std, self.weights, self.biases,
            self.idx_to_label,
            history_size=HISTORY_LEN,
            min_conf=MIN_CONF,
        )

        # Debounce: require MIN_CONSEC consecutive non-idle windows before confirming.
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

        # Assign labels to subsampled frame positions
        n_sub = len(sub_idx)
        sub_frame_labels = ["idle"] * n_sub
        for ci, label in zip(center_indices, smoothed):
            if ci < n_sub:
                sub_frame_labels[ci] = label

        # Fire one event per gesture segment at subsampled resolution
        sub_events = []
        last_event = "idle"
        for label in sub_frame_labels:
            full = LABEL_TO_FULL.get(label, label)
            if full != "idle" and last_event == "idle":
                sub_events.append(full)
                last_event = full
            else:
                sub_events.append("idle")
                if full == "idle":
                    last_event = "idle"

        # Map fired events back to original frame positions
        events = ["idle"] * n_total
        for sub_i, event in enumerate(sub_events):
            if event != "idle":
                orig_i = int(sub_idx[sub_i])
                if orig_i < n_total:
                    events[orig_i] = event

        return events


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

frames = frames.copy()
frames["events"] = my_model.compute_events(frames)

# Output must have 'timestamp' index and 'events' column
frames["events"].to_csv(output_path, index=True)
print("events exported to %s" % output_path)

print("\nEvent distribution:")
print(frames["events"].value_counts().to_string())
