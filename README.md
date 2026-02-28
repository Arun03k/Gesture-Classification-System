## Final Submission – Machine Learning Project (WS25)

---

## 📌 Project Overview

This repository contains the final submission of our Machine Learning project.

The objective of this project is to develop a **gesture classification system implemented in Python** that enables users to control a slideshow using body gestures detected from live camera input.

The system:

* Extracts body keypoints using **MediaPipe**
* Stores motion data in CSV format
* Trains a neural network implemented from scratch using **NumPy**
* Detects gestures in real time
* Integrates predictions into a **Reveal.js** slideshow
* Provides an evaluation mode for performance scoring

The implementation follows the official project requirements .

---

# 🧠 Implemented Requirements

## ✅ Mandatory Requirements

| ID | Description                                                             |
| -- | ----------------------------------------------------------------------- |
| M1 | Neural network implemented with Python and NumPy                        |
| M2 | Detection of mandatory gestures (Swipe Right, Swipe Left, Rotation)     |
| M3 | Reveal.js slideshow control via gesture prediction                      |
| M4 | Deployable using pip / pipenv                                           |
| M5 | Teaser video (H264, mp4 format)                                         |
| M6 | Performance evaluation mode                                             |
| M7 | Presentation of data preparation, hyperparameter search, and evaluation |

---

## ⭐ Selected Optional Requirements

The following optional requirements are implemented and integrated:

* O1 – Machine Learning Framework Package
* O1.1 – Visualization Module
* O2 – Principal Component Analysis (PCA)
* O3 – Additional Gesture
* O5 – Additional Gesture
* O9 – Attention Detection (Blocking unintended commands)
* O10 – Game Control via Gestures
* O12 – Gradient Descent Variations

The selected optionals are documented in the `orga` repository.

---

# 🏗️ System Architecture

```text
Camera / Video Input
        ↓
MediaPipe Pose Extraction
        ↓
CSV Motion Data
        ↓
Feature Engineering (+ PCA)
        ↓
Neural Network (Custom Framework)
        ↓
Gesture Prediction
        ↓
Reveal.js / Game Control
```

---

# 📂 Repository Structure

```bash
final/
│
├── src/
│   ├── data_processing/
│   ├── feature_engineering/
│   ├── model/
│   ├── prediction/
│   ├── evaluation/
│
├── reveal_slideshow/
├── demo_data/
│
├── run_prediction.py
├── run_performance_test.py
│
├── performance_results.csv
├── requirements.txt / Pipfile
└── README.md
```

---

# 🚀 Setup Instructions (M4)

## 1️⃣ Clone Repository

```bash
git clone <repository-url>
cd final
```

## 2️⃣ Install Dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Or using pipenv:

```bash
pipenv install
pipenv shell
```

---

# ▶️ Prediction Mode (M3)

To start real-time gesture detection and slideshow control:

```bash
python run_prediction.py
```

This will:

* Start camera input
* Extract pose keypoints
* Perform gesture classification
* Send events to the Reveal.js slideshow

---

# 🧪 Performance Evaluation Mode (M6)

To evaluate performance on a given transcript:

```bash
python run_performance_test.py --input <video_transcript.csv>
```

Output:

* `performance_results.csv`
* Console performance score

---

# 📊 Machine Learning Approach

## Feature Engineering

* Keypoint normalization
* Temporal aggregation (sliding windows)
* Velocity-based features
* Optional PCA dimensionality reduction

## Neural Network

* Fully connected feedforward network
* Custom backpropagation
* Cross-Entropy with Softmax
* Implemented without external ML frameworks

## Optimization

* Vanilla Gradient Descent
* Momentum-based Gradient Descent
* Nesterov Accelerated Gradient

## Evaluation Metrics

* Accuracy
* F1 Score
* Confusion Matrix
* Learning Curves
* Performance Score (M6)

---

# 🎮 Supported Gestures

## Mandatory

* Swipe Right
* Swipe Left
* Rotation

## Additional

* Rotate (Counter Clockwise)
* Swipe Up / Swipe Down
* Attention Detection
* Game Control Gestures

---

# 🎥 Teaser Video (M5)

The teaser video:

* Duration: 1–2 minutes
* Format: mp4
* Codec: H264
* Includes official HCI intro/outro template

📎 Location: *[Insert link or path]*

---

# 👥 Team Responsibilities

| Team Member | Responsibilities         | Optional Ownership |
| ----------- | ------------------------ | ------------------ |
| Name        | Core Model               | O?                 |
| Name        | Data & Features          | O?                 |
| Name        | Integration & Evaluation | O?                 |

Each team member can explain all mandatory components.
Optional requirements are distributed equally across the team.

---

# 🔖 Submission Information

* Git tag used for submission:

```bash
git tag final
```

* Submission via GitLab group repository
* All code required for grading is versioned

---

# 📎 Compliance

* Python ≥ 3.6
* Only whitelisted libraries used
* No forbidden ML frameworks (TensorFlow, PyTorch, scikit-learn, etc.)
* Fully executable on HCI lab workstation

---

# 🏁 Summary

This repository contains the complete, deployable, and evaluable final system fulfilling all mandatory requirements and the selected optional extensions.

The system demonstrates a full machine learning pipeline from motion capture to real-time gesture-based interaction.

---

