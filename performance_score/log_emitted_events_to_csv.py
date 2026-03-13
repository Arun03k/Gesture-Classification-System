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
    MODEL_MANDATORY, MODEL_OPTIONALS, MODEL_GAME, AVAILABLE_MODELS,
    extract_features, forward_pass,
    subsample_to_fps,
    load_model_artifacts,
    get_model_dir,
)

# Paths and tuning constants
BASE_MODEL_DIR = PROJECT_ROOT / "data" / "processed"

WINDOW_SIZE    = 18    # frames per window (0.6 s @ 30 FPS)
HISTORY_LEN    = 5     # majority-vote history length
MIN_CONF       = 0.6   # minimum softmax confidence to accept non-idle
MIN_CONSEC     = 5     # debounce: require N consecutive non-idle windows
IDLE_RESET_MIN = 15    # consecutive idle windows before same gesture can re-fire


def make_windows(features, window_size=WINDOW_SIZE):
    """Slide a fixed-length window over the feature sequence (stride=1)."""
    n_frames, n_feats = features.shape
    mid = window_size // 2
    windows, centers = [], []
    for start in range(0, n_frames - window_size + 1):
        windows.append(features[start:start + window_size].flatten())
        centers.append(start + mid)
    if not windows:
        return np.empty((0, window_size * n_feats)), np.array([], dtype=int)
    return np.stack(windows), np.array(centers, dtype=int)


def predict_smoothed(windows, weights, biases, idx_to_label,
                     history_size=HISTORY_LEN, min_conf=MIN_CONF):
    """Run forward pass on each window and apply majority-vote smoothing."""
    history = ["idle"] * history_size
    preds = []
    for i in range(len(windows)):
        probs    = forward_pass(windows[i].reshape(1, -1), weights, biases)
        pred_idx = int(np.argmax(probs))
        conf     = float(probs[0, pred_idx])
        label = idx_to_label.get(pred_idx, "idle")
        if conf < min_conf:
            label = "idle"
        history.pop(0)
        history.append(label)
        voted = Counter(history).most_common(1)[0][0]
        preds.append(voted)
    return preds


class GestureApplication:

    def __init__(self, model_name=MODEL_MANDATORY):
        model_dir = get_model_dir(BASE_MODEL_DIR, model_name)
        self.weights, self.biases, self.mean_, self.std_, self.idx_to_label = \
            load_model_artifacts(model_dir)
        print(f"[GestureApplication] Model '{model_name}' loaded — "
              f"{len(self.idx_to_label)} classes: {list(self.idx_to_label.values())}")

    def compute_events(self, frames):
        """Run inference on the full pose DataFrame and return one event label per frame."""
        n_total = len(frames)
        df = frames.reset_index().copy()

        # FPS normalisation
        df_sub, sub_idx = subsample_to_fps(df, TARGET_FPS)
        features = extract_features(df_sub)

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

        # Debounce: require MIN_CONSEC consecutive non-idle windows before confirming
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

        # Fire one event per gesture segment; require IDLE_RESET_MIN consecutive
        # idle frames before the same gesture can re-fire (prevents double-fire)
        sub_events = []
        last_event = "idle"
        idle_consec = 0
        for label in sub_frame_labels:
            full = LABEL_TO_FULL.get(label, label)
            if full != "idle":
                idle_consec = 0
                if last_event == "idle":
                    sub_events.append(full)
                    last_event = full
                else:
                    sub_events.append("idle")
            else:
                idle_consec += 1
                sub_events.append("idle")
                if idle_consec >= IDLE_RESET_MIN:
                    last_event = "idle"

        # Map fired events back to original frame positions
        events = ["idle"] * n_total
        for sub_i, event in enumerate(sub_events):
            if event != "idle":
                orig_i = int(sub_idx[sub_i])
                if orig_i < n_total:
                    events[orig_i] = event

        return events


# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Run gesture inference on a poses CSV and emit predicted events."
)
parser.add_argument("--input_frames_csv", required=True,
                    help="CSV with MediaPipe keypoint columns and 'time' or 'timestamp' index")
parser.add_argument("--output_csv_name", default=None,
                    help="Output filename (default: <input_stem>_events.csv)")
parser.add_argument("--output_dir", default=None,
                    help="Directory for events CSV (default: data/predicted_events/)")
parser.add_argument("--model", default=MODEL_MANDATORY, choices=AVAILABLE_MODELS,
                    help=f"Model to use (default: {MODEL_MANDATORY})")

args = parser.parse_known_args()[0]

input_path = pathlib.Path(args.input_frames_csv)

# Resolve output path — default to data/predicted_events/
out_name = args.output_csv_name or (input_path.stem + "_events.csv")
if args.output_dir:
    out_dir = pathlib.Path(args.output_dir)
else:
    out_dir = PROJECT_ROOT / "data" / "predicted_events"
out_dir.mkdir(parents=True, exist_ok=True)
output_path = out_dir / out_name

# Auto-detect timestamp column
_peek = pd.read_csv(input_path, nrows=0)
ts_col = "timestamp" if "timestamp" in _peek.columns else "time"

frames = pd.read_csv(input_path, index_col=ts_col)
# Handle both integer ms timestamps and timedelta strings (e.g. '0 days 00:00:00.033300')
try:
    frames.index = frames.index.astype(int)
except (ValueError, TypeError):
    frames.index = (pd.to_timedelta(frames.index).total_seconds() * 1000).astype(int)

# Drop non-keypoint columns if present (ground_truth, gesture, participant, etc.)
extra_cols = [c for c in frames.columns if c in ("ground_truth", "gesture", "participant", "label", "source_file", "person")]
if extra_cols:
    print(f"[Info] Dropping non-keypoint columns: {extra_cols}")
    frames = frames.drop(columns=extra_cols)

print(f"Loaded {len(frames)} frames from: {input_path}")

# ── Run inference ────────────────────────────────────────────────────────────
my_model = GestureApplication(model_name=args.model)
frames = frames.copy()
frames["events"] = my_model.compute_events(frames)

frames["events"].to_csv(output_path, index=True)
print(f"Events exported to: {output_path}")

print("\nEvent distribution:")
print(frames["events"].value_counts().to_string())
