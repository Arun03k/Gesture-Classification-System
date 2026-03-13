# Final Submission – Machine Learning Project (WS25)

---

## Project Overview

This repository contains the final submission of our Machine Learning project for WS25.

The objective is to develop a **gesture classification system implemented in Python** that enables users to control a slideshow using body gestures detected from live camera input.

The system:

- Extracts body keypoints using **MediaPipe Pose**
- Stores motion data in CSV format
- Trains a neural network implemented from scratch using **NumPy only**
- Detects gestures in real time via webcam
- Integrates predictions into a **Reveal.js** slideshow controlled over WebSockets
- Provides a performance evaluation mode for scoring against ground-truth transcripts

---

## Team Members

| Name | Student ID |
| ---- | ---------- |
| Arun Kumar | 3030010 |
| Aswathy Baiju | 3168821 |
| Nayana S Pawar | 3304785 |

---

## Related Repositories (Workspace Paths)

- `../orga` — central organization and submission overview
- `../ml-package` — reusable custom ML framework package (O1, O1.1)

---

## Team Responsibilities

- All mandatory requirements and all selected optionals were implemented collaboratively by the full team.
- **Game implementation (O10)** in `snake_game/` was implemented by **Arun Kumar**.
- All other optional requirements were handled jointly by Arun Kumar, Aswathy Baiju, and Nayana S Pawar.

---

## Implemented Requirements

### Mandatory Requirements

| ID | Description | Status |
| -- | ----------- | ------ |
| M1 | Neural network implemented with Python and NumPy only | Done |
| M2 | Detection of mandatory gestures: Swipe Right, Swipe Left, Rotate Clockwise | Done |
| M3 | Reveal.js slideshow control via real-time gesture prediction | Done |
| M4 | Deployable using `pip` / `requirements.txt` | Done |
| M5 | Teaser video (H264, mp4 format, 1–2 min) | Done |
| M6 | Performance evaluation mode with `calculator.py` and scoring formula | Done |
| M7 | Notebooks covering data preparation, hyperparameter search, and evaluation | Done |

---

### Optional Requirements

| ID | Description | Status |
| -- | ----------- | ------ |
| O1 | ML Framework Package — reusable pipeline module (`pipeline/`, `model_creation/`) | Done |
| O1.1 | Visualization Module — training curves, confusion matrix, multi-model comparison plots | Done |
| O2 | Principal Component Analysis (PCA) from scratch using NumPy eigendecomposition | Done |
| O3 | Additional gesture: Swipe Up | Done |
| O5 | Additional gesture: Rotate Counter-Clockwise | Done |
| O9 | Attention Detection — confidence gating, debounce, cooldown, no-person guard | Done |
| O10 | Game / Slideshow Image Control — zoom, move, rotate, reset via gestures | Done |
| O12 | Gradient Descent Variations — SGD, Momentum, Adam (all implemented from scratch) | Done |

---

## Game Implementation (Snake)

As part of optional requirement **O10**, a gesture-controlled Snake game is included.

- **File:** `snake_game/snake_game.py`

### How to Play

1.  Run the script from the `final-submission` directory:
    ```bash
    python snake_game/snake_game.py
    ```
2.  A Pygame window will open, and gesture detection will start automatically.

### Controls

The snake is controlled using the following gestures:

- **Swipe Up**: Move the snake up.
- **Swipe Down**: Move the snake down.
- **Swipe Left**: Move the snake left.
- **Swipe Right**: Move the snake right.

The game uses the same real-time gesture recognition pipeline as the slideshow controller.

---

## Project Structure

```
final-submission/
├── .git/
├── .gitignore
├── .ipynb_checkpoints/
├── data/
│   ├── csv_results/
│   │   ├── ground_truth/
│   │   ├── poses_only/
│   │   └── poses_with_ground_truth/
│   ├── elan_data/
│   │   ├── elan_files/
│   │   └── text_files/
│   ├── predicted_events/
│   ├── processed/
│   │   ├── game/
│   │   └── optionals/
│   └── videos/
├── live_gesture_recognition.py
├── ml_project_team_31_teaser_video.mp4
├── model_creation/
│   ├── .ipynb_checkpoints/
│   ├── adam_neural_net.py
│   ├── base_neural_net.py
│   ├── base_neural_net_pca.py
│   ├── gesture_recognition_game.ipynb
│   ├── gesture_recognition_ml.ipynb
│   ├── gesture_recognition_optionals.ipynb
│   ├── gesture_recognition_preproc.ipynb
│   ├── gesture_recognition_preproc_optionals.ipynb
│   ├── helper_functions.py
│   ├── label_mapping_optionals.npz
│   ├── model_weights_optionals.npz
│   ├── pca_functions.py
│   ├── scaler_params_optionals.npz
│   └── __pycache__/
├── notebooks/
│   ├── create_csv.ipynb
│   └── process_videos/
│       ├── helpers/
│       ├── keypoint_mapping.yml
│       ├── live_video_feed.py
│       └── video_to_csv.py
├── performance_score/
│   ├── .ipynb_checkpoints/
│   ├── calculator.py
│   ├── events_visualization.py
│   └── log_emitted_events_to_csv.py
├── pipeline/
│   ├── gesture_pipeline.py
│   └── __init__.py
├── presentation/
│   └── Team 31.pptx
├── README.md
├── requirements.txt
├── RUN_COMMANDS.txt
├── slideshow/
│   ├── control_slideshow_example.py
│   ├── event_listeners.js
│   ├── helper_methods.js
│   ├── slideshow.html
│   ├── slideshow_server.py
│   └── static/
│       ├── images/
│       ├── index.html
│       ├── js/
│       └── slides.csv
├── snake_game/
│   ├── .ipynb_checkpoints/
│   └── snake_game.py
├── venv/
└── visualizations/
    ├── eda/
    │   ├── arun_optionals/
    │   ├── aswathy_optionals/
    │   └── nayana_optionals/
    └── training/
```
│   └── gesture_recognition_optionals.ipynb # Optionals experiments (O2, O12)
│
├── slideshow/
│   ├── slideshow_server.py         # FastAPI + WebSocket server (M3)
│   ├── slideshow.html              # Reveal.js slideshow page
│   ├── control_slideshow_example.py
│   ├── event_listeners.js
│   ├── helper_methods.js
│   └── static/
│       ├── index.html
│       ├── slides.csv              # Slide definitions (image, title, subtitle)
│       ├── js/client.js            # WebSocket listener + image manipulation (O10)
│       └── images/
│
├── performance_score/
│   ├── calculator.py               # Performance scoring (M6)
│   ├── events_visualization.py     # Ground-truth vs. predicted visualization (M6)
│   └── log_emitted_events_to_csv.py
│
├── process_videos/
│   ├── video_to_csv.py             # Extract keypoints from video files
│   ├── live_video_feed.py
│   ├── keypoint_mapping.yml
│   └── helpers/
│       ├── data_to_csv.py
│       └── video_to_dataframe.py
│
├── data/
│   ├── processed/                  # Saved model weights, scaler, label mapping
│   ├── csv_results/
│   │   ├── ground_truth/           # Ground-truth CSVs for all performers
│   │   └── poses_with_ground_truth/
│   ├── elan_data/                  # ELAN annotation files and text transcripts
│   └── videos/
│
├── notebooks/
│   └── create_csv.ipynb            # Keypoint extraction notebook
│
└── visualizations/
    └── eda/                        # Exploratory data analysis plots
```

---

## Setup Instructions (M4)

### 1. Clone the repository

```bash
git clone <repository-url>
cd final-submission
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the System

### Real-Time Gesture Prediction (M3)

Start the slideshow server first:

```bash
cd slideshow
python slideshow_server.py
```

Then open `http://127.0.0.1:8800` in a browser and start the gesture recogniser:

```bash
# Basic (no slideshow connection)
python live_gesture_recognition.py

# With slideshow integration
python live_gesture_recognition.py --slideshow

# Mirror the webcam image
python live_gesture_recognition.py --slideshow --flip

# Use a different camera index
python live_gesture_recognition.py --camera 1 --slideshow
```

Press **Q** or **ESC** to stop.

### Performance Evaluation Mode (M6)

```bash
python -m performance_score.calculator \
    --predicted_events_csv <path/to/predicted.csv> \
    --ground_truth_csv <path/to/ground_truth.csv>
```

---

## Machine Learning Approach

### Data Collection

- Three performers: Arun Kumar, Aswathy Baiju, Nayana S Pawar
- Gestures recorded using MediaPipe Pose (upper-body keypoints: 15 joints × 3 axes)
- Annotated with ELAN and exported to CSV

### Feature Engineering

- Chest-centred normalisation (shoulder-midpoint origin, scaled by shoulder width)
- Temporal velocity features (frame-to-frame differences)
- Feature vector per window: 15 keypoints × 3 axes × 2 (position + velocity) = **90 features**
- Sliding window of 18 frames at 30 FPS
- Optional PCA dimensionality reduction (O2)

### Neural Network (M1)

- Fully-connected feedforward network
- ReLU hidden layers, Softmax output
- Cross-entropy loss with optional class weighting
- Custom backpropagation — NumPy only, no external ML frameworks

### Gradient Descent Variations (O12)

Implemented from scratch in `adam_neural_net.py` (`NeuralNetwork` class):

| Optimizer | Details |
| --------- | ------- |
| Vanilla SGD | Standard gradient descent |
| Momentum SGD | Velocity-based gradient accumulation |
| Adam | Adaptive moment estimation (β₁=0.9, β₂=0.999) |

### Evaluation Metrics

- Accuracy
- Macro F1 Score
- Confusion Matrix (counts + normalised)
- Learning curves (loss, accuracy, F1)
- Multi-model comparison plots (O1.1)
- Performance Score (M6 formula: bonus per correct gesture, malus per false positive)

---

## Detected Gestures

| Gesture | Label | Requirement |
| ------- | ----- | ----------- |
| Swipe Right | `sr` | M2 (mandatory) |
| Swipe Left | `sl` | M2 (mandatory) |
| Rotate Clockwise | `r_cw` | M2 (mandatory) |
| Swipe Up | `su` | O3 (optional) |
| Rotate Counter-Clockwise | `r_ccw` | O5 (optional) |
| Swipe Down | `sd` | O3 (optional) |
| Idle | `idle` | — |

---

## Optional Requirements — Details

### O1 — ML Framework Package

The project exposes a reusable Python package:

- **`pipeline/gesture_pipeline.py`** — shared inference pipeline imported by both the live recogniser and the evaluation logger. Provides normalisation, feature extraction, FPS detection/subsampling, and a NumPy-only forward pass.
- **`model_creation/`** — modular classes (`BaseNeuralNetwork`, `BaseNeuralNetworkPCA`, `AdamNeuralNetwork`, `NeuralNetwork`) and utility functions that can be imported independently.

### O1.1 — Visualization Module

`model_creation/helper_functions.py` provides:

- `plot_metrics` — 3-panel learning curves (loss / accuracy / F1)
- `plot_confusion_matrix` — heatmap with optional normalisation
- `save_training_history` / `load_training_history` — persist training runs as `.npz`
- `plot_model_comparison` — single-metric comparison across models
- `plot_multi_model_summary` — 3-panel validation comparison for multiple models

### O2 — Principal Component Analysis

`model_creation/pca_functions.py` implements `ManualPCA`:

- Covariance matrix via `np.cov`
- Eigendecomposition with `np.linalg.eigh`
- Explained variance ratio computation
- `fit`, `transform`, `fit_transform` API
- `BaseNeuralNetworkPCA` accepts PCA-reduced inputs

### O3 — Additional Gesture: Swipe Up

Swipe Up (`su`) is detected and mapped to the `swipe_up` slideshow command (moves to the previous vertical sub-slide in Reveal.js).

### O5 — Additional Gesture: Rotate Counter-Clockwise

Rotate Counter-Clockwise (`r_ccw`) is detected and mapped to `rotate_counter_clock`, which rotates the current slide image by −90° in the slideshow.

### O9 — Attention Detection

The live recogniser filters unintended commands through a multi-layer guard:

1. **Presence detection** — if MediaPipe finds no person in frame, all inference is suppressed and a "No person detected" warning is shown.
2. **Confidence gate** (`MIN_CONF = 0.82`) — softmax probabilities below 82% are classified as `idle`.
3. **Majority-vote smoothing** — a 9-frame history window suppresses single-frame outliers.
4. **Consecutive-frame debounce** (`MIN_CONSEC = 10`) — requires 10 consecutive agreeing windows before a gesture fires.
5. **Cooldown guard** (`COOLDOWN_SEC = 2`) — minimum 2 seconds between successive events prevents repeated triggers.

Together these layers ensure only deliberate, sustained gestures emit commands.

### O10 — Snake-Game


| Command | Effect |
| ------- | ------ |
| `move_left / right / up / down` | Control game |


### O12 — Gradient Descent Variations

The `NeuralNetwork` class in `model_creation/adam_neural_net.py` implements:

- **Vanilla SGD** (`optimizer='sgd'`)
- **Momentum SGD** (`optimizer='momentum'`, configurable momentum coefficient)
- **Adam** (`optimizer='adam'`, β₁, β₂, ε all configurable)

Additional features: L2 regularisation, inverted dropout, gradient clipping, mini-batch training, early stopping, and learning-rate scheduling.

---

## Performance Evaluation (M6)

The scorer in `performance_score/calculator.py` implements the official formula:

- **+bonus** (default 10) for each correctly detected gesture (first event during a gesture interval)
- **−malus** (default 0.2) for each false positive (event fired outside a gesture)
- Final score normalised by the number of ground-truth gestures

Run the visualisation separately to compare predicted events against ground truth frame-by-frame:

```bash
python -m performance_score.events_visualization \
    --predicted_events_csv <predicted.csv> \
    --ground_truth_csv <ground_truth.csv>
```

---

## Teaser Video (M5)

- Format: mp4
- Codec: H264
- Duration: 1 minute
- Includes official HCI intro/outro template

---

## Compliance

- Python ≥ 3.6
- Only whitelisted libraries used (`numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv-python`, `mediapipe`, `pyyaml`, `tqdm`, `jupyter`, `fastapi`, `uvicorn`, `websockets`)
- No forbidden ML frameworks (no TensorFlow, PyTorch, scikit-learn, etc.)
- All ML logic (neural network, PCA, optimisers, metrics) implemented from scratch with NumPy

