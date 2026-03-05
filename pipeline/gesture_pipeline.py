"""Shared inference pipeline. Imported by live_gesture_recognition.py and log_emitted_events_to_csv.py."""

import pathlib
import numpy as np
import pandas as pd

from ml_framework.layers.activations import relu as _relu, softmax as _softmax

# Constants
TARGET_FPS = 30        # model was trained at 30 FPS

# ── Model name constants ──────────────────────────────────────────
MODEL_MANDATORY  = "mandatory"     # idle, sl, sr, r_cw  (4 classes)
MODEL_OPTIONALS  = "optionals"     # idle, sl, sr, r_cw, r_ccw, su, sd (7 classes)
MODEL_GAME       = "game"          # idle, sl, sr, su, sd (5 classes)
AVAILABLE_MODELS = [MODEL_MANDATORY, MODEL_OPTIONALS, MODEL_GAME]

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

# Short model label → full ground-truth name (used by calculator.py & display)
LABEL_TO_FULL = {
    "idle":  "idle",
    "sl":    "swipe_left",
    "sr":    "swipe_right",
    "r_cw":  "rotate_clockwise",
    "r_ccw": "rotate_anticlockwise",
    "sd":    "swipe_down",
    "su":    "swipe_up",
}


def get_model_dir(base_dir: pathlib.Path, model_name: str = MODEL_MANDATORY) -> pathlib.Path:
    """Return the directory containing model artefacts for the given model name.

    Layout:
      base_dir/                       ← mandatory (backward compat)
      base_dir/optionals/             ← optionals (all 7 gestures)
      base_dir/game/                  ← game (directional gestures)
    """
    if model_name == MODEL_MANDATORY:
        return pathlib.Path(base_dir)          # files live at the base level
    return pathlib.Path(base_dir) / model_name  # sub-directory


# Preprocessing
def normalize_chest_centered(df: pd.DataFrame) -> pd.DataFrame:
    """Translate keypoints to shoulder-midpoint origin and scale by shoulder width."""
    df = df.copy()
    cx = (df["left_shoulder_x"] + df["right_shoulder_x"]) / 2
    cy = (df["left_shoulder_y"] + df["right_shoulder_y"]) / 2
    cz = (df["left_shoulder_z"] + df["right_shoulder_z"]) / 2
    sw = np.sqrt(
        (df["left_shoulder_x"] - df["right_shoulder_x"]) ** 2 +
        (df["left_shoulder_y"] - df["right_shoulder_y"]) ** 2 +
        (df["left_shoulder_z"] - df["right_shoulder_z"]) ** 2
    )
    sw = sw.replace(0, np.nan)
    coord_cols = [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]
    for col in coord_cols:
        if   col.endswith("_x"): df[col] = (df[col] - cx) / sw
        elif col.endswith("_y"): df[col] = (df[col] - cy) / sw
        elif col.endswith("_z"): df[col] = (df[col] - cz) / sw
    return df


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Return (N, 90) array of chest-normalised position + velocity for 15 upper-body keypoints."""
    df = normalize_chest_centered(df).fillna(0.0)
    feat_cols = [p + s for p in UPPER_BODY_PARTS for s in SUFFIXES if (p + s) in df.columns]
    pos = df[feat_cols].values.astype(np.float64)
    vel = np.vstack([np.zeros((1, pos.shape[1])), np.diff(pos, axis=0)])
    return np.hstack([pos, vel])   # (N, 90)


# FPS utilities
def detect_fps(df: pd.DataFrame) -> int:
    """Estimate recording FPS from the 'time' column (millisecond timestamps)."""
    time_col = "time" if "time" in df.columns else df.columns[0]
    vals = df[time_col].values.astype(float)
    diffs = np.diff(vals)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return TARGET_FPS
    return int(round(1000.0 / np.median(diffs)))


def subsample_to_fps(df: pd.DataFrame, target_fps: int = TARGET_FPS):
    """Subsample df to target_fps if recorded at a higher rate. Returns (subsampled_df, original_indices)."""
    src_fps = detect_fps(df)
    step = max(1, round(src_fps / target_fps))
    if step > 1:
        print(f"[FPS] Detected {src_fps} fps — subsampling by {step} to ~{round(src_fps/step)} fps")
    orig_idx = np.arange(0, len(df), step)
    return df.iloc[orig_idx].reset_index(drop=True), orig_idx


# Neural network (using ml_framework activations)
def forward_pass(X, weights, biases):
    """ReLU hidden layers + softmax output."""
    a = np.atleast_2d(X)
    for W, b in zip(weights[:-1], biases[:-1]):
        a = _relu(a @ W + b)
    return _softmax(a @ weights[-1] + biases[-1])


# Model loading
def load_model_artifacts(model_dir: pathlib.Path):
    """Load weights, scaler parameters, and label mapping from model_dir."""
    model_dir = pathlib.Path(model_dir)
    for fname in ("model_weights.npz", "scaler_params.npz", "label_mapping.npz"):
        p = model_dir / fname
        if not p.exists():
            raise FileNotFoundError(
                f"Missing model artifact: {p}\n"
                "Run the training notebook first to generate model artefacts."
            )

    data = np.load(model_dir / "model_weights.npz")
    n = len([k for k in data.files if k.startswith("W")])
    weights = [data[f"W{i}"] for i in range(n)]
    biases  = [data[f"b{i}"] for i in range(n)]

    scaler  = np.load(model_dir / "scaler_params.npz")
    mean_   = scaler["mean"]
    std_    = scaler["std"].copy()
    std_[std_ == 0] = 1.0

    lm = np.load(model_dir / "label_mapping.npz", allow_pickle=True)
    idx_to_label = {
        int(i): str(l)
        for l, i in zip(lm["labels"].tolist(), lm["indices"].tolist())
    }

    return weights, biases, mean_, std_, idx_to_label
