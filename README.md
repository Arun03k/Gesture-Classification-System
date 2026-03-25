````markdown
# Gesture-Controlled Slideshow System

A real-time gesture recognition system built in Python that lets users control a slideshow using body movements detected from a live camera feed.

The project combines **MediaPipe Pose**, a **NumPy-only neural network**, real-time webcam inference, and **Reveal.js** slideshow control over WebSockets. It also includes a gesture-controlled Snake game and an evaluation pipeline for measuring prediction quality against ground-truth annotations.

---

## Overview

This system captures body keypoints from a webcam, transforms them into motion-aware feature vectors, and classifies gestures in real time.

Detected gestures can be used to:

- control a **Reveal.js** slideshow
- rotate slides through gesture commands
- control a **Snake game**
- evaluate prediction quality using recorded event logs and transcripts

---

## Demo

Add your teaser video or preview asset inside the `presentation/` folder.

**Teaser video:**
[▶ Watch the teaser video](presentation/teaser.mp4)

> If you have a GIF or screenshot preview, you can also place it here for better GitHub rendering.

---

## Key Features

- Real-time gesture recognition from live webcam input
- Pose-based motion capture using MediaPipe Pose
- NumPy-only neural network with custom forward and backpropagation
- Reveal.js slideshow control via WebSockets
- Performance evaluation mode with event scoring against ground truth
- PCA from scratch using eigendecomposition
- Multiple optimizers implemented from scratch: SGD, Momentum, Adam
- Visualization utilities for training curves, confusion matrices, and model comparison
- Gesture-controlled Snake game
- Attention and confidence filtering to suppress accidental commands

---

## Supported Gestures

| Gesture | Label | Action |
|--------|-------|--------|
| Swipe Right | `sr` | Next slide / move right |
| Swipe Left | `sl` | Previous slide / move left |
| Rotate Clockwise | `r_cw` | Rotate slide clockwise |
| Rotate Counter-Clockwise | `r_ccw` | Rotate slide counter-clockwise |
| Swipe Up | `su` | Previous vertical slide / move up |
| Swipe Down | `sd` | Next vertical slide / move down |
| Idle | `idle` | No action |

---

## How It Works

### 1. Pose Extraction
Body keypoints are extracted from webcam frames using **MediaPipe Pose**.

### 2. Feature Engineering
Each gesture window is transformed into a motion-aware feature vector using:

- chest-centered normalization
- shoulder-width scaling
- frame-to-frame velocity features
- sliding temporal windows

**Feature size per window:**

```text
15 keypoints × 3 axes × 2 (position + velocity) = 90 features
````

### 3. Classification

The feature vectors are passed into a fully connected neural network implemented entirely with **NumPy**.

### 4. Live Prediction

Predictions are smoothed and filtered using:

* no-person detection
* confidence thresholding
* majority voting
* consecutive-frame debounce
* cooldown timing

### 5. Action Layer

Recognized gestures are mapped to slideshow commands or game controls.

---

## Model Details

### Neural Network

* Fully connected feedforward architecture
* ReLU hidden activations
* Softmax output layer
* Cross-entropy loss
* Custom backpropagation using NumPy only

### Optimization

Implemented from scratch in `model_creation/adam_neural_net.py`:

* **SGD**
* **Momentum SGD**
* **Adam**

Additional training features include:

* mini-batch training
* dropout
* L2 regularization
* gradient clipping
* early stopping
* learning-rate scheduling

### PCA

`model_creation/pca_functions.py` includes a manual PCA implementation with:

* covariance matrix computation
* eigendecomposition via `np.linalg.eigh`
* explained variance ratio
* `fit`, `transform`, and `fit_transform`

---

## Project Structure

```text
final-submission/
├── data/
├── model_creation/
├── notebooks/
├── performance_score/
├── pipeline/
├── presentation/
├── slideshow/
├── snake_game/
├── visualizations/
├── live_gesture_recognition.py
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd final-submission
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the System

### Real-Time Gesture Recognition

Run the recognizer directly:

```bash
python live_gesture_recognition.py
```

Useful options:

```bash
python live_gesture_recognition.py --slideshow
python live_gesture_recognition.py --slideshow --flip
python live_gesture_recognition.py --camera 1 --slideshow
```

Press **Q** or **ESC** to stop.

---

## Slideshow Integration

Start the slideshow server first:

```bash
cd slideshow
python slideshow_server.py
```

Then open:

```text
http://127.0.0.1:8800
```

Now run the gesture recognizer with slideshow support:

```bash
python live_gesture_recognition.py --slideshow
```

---

## Performance Evaluation

The scoring tool compares predicted gesture events with ground-truth annotations.

```bash
python -m performance_score.calculator \
    --predicted_events_csv <path/to/predicted.csv> \
    --ground_truth_csv <path/to/ground_truth.csv>
```

### Scoring Logic

* **Bonus** for each correctly detected gesture
* **Malus** for each false positive
* Final score normalized by the number of ground-truth gestures

To visualize predicted events against ground truth frame by frame:

```bash
python -m performance_score.events_visualization \
    --predicted_events_csv <predicted.csv> \
    --ground_truth_csv <ground_truth.csv>
```

---

## Snake Game

A gesture-controlled Snake game is included in:

```text
snake_game/snake_game.py
```

Run it with:

```bash
python snake_game/snake_game.py
```

### Controls

* **Swipe Up** → move up
* **Swipe Down** → move down
* **Swipe Left** → move left
* **Swipe Right** → move right

The game uses the same real-time gesture recognition pipeline as the slideshow controller.

---

## Visualization Utilities

The visualization tools in `model_creation/helper_functions.py` support:

* training curves
* confusion matrices
* saved training histories
* single-metric model comparison
* multi-model validation summaries

These utilities help compare experiments and interpret model behavior more effectively.

---

## Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* OpenCV
* MediaPipe
* FastAPI
* Uvicorn
* WebSockets
* Jupyter

---

## Team

Built by:

* Arun Kumar
* Aswathy Baiju
* Nayana S Pawar

---

## Related Repositories

* `../orga` — organization and submission overview
* `../ml-package` — reusable custom ML framework package

---

## Notes

* Python 3.6+
* No external ML frameworks used
* All core ML logic, including neural network training, PCA, optimization, and evaluation, is implemented from scratch using **NumPy**

```

I can also turn it into a cleaner GitHub-style version with badges, a preview image section, and a more polished landing-page layout.
```
