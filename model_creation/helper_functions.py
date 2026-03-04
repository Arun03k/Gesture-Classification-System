"""
HelperFunctions.py — Activation functions, loss, metrics, and plotting utilities.
Used by all neural network classes and training notebooks.

O1.1 Visualization: Includes functions to save/load training histories and
generate combined multi-model comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ════════════════════════════════════════════════════════════════════
#  ACTIVATION FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(A):
    return A * (1 - A)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(A):
    return (A > 0).astype(A.dtype)


def softmax(Z):
    """Numerically stable softmax."""
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)


# ════════════════════════════════════════════════════════════════════
#  LOSS FUNCTION
# ════════════════════════════════════════════════════════════════════

def cross_entropy_loss(y_true, y_pred, class_weights=None):
    """
    Cross-entropy loss.
    y_true: one-hot encoded labels (N, C)
    y_pred: softmax output (N, C)
    class_weights: optional array of shape (C,) for class weighting
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    losses = -np.sum(y_true * np.log(y_pred), axis=1)

    if class_weights is not None:
        weights = np.array(class_weights)
        sample_weights = (y_true * weights).sum(axis=1)
        losses = losses * sample_weights

    return np.mean(losses)


# ════════════════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════════════════

def accuracy(y_true, y_pred):
    """
    Compute accuracy from one-hot encoded labels and predictions.
    Supports both one-hot (N, C) and integer (N,) label formats.
    """
    if y_true.ndim == 2:
        true_labels = np.argmax(y_true, axis=1)
    else:
        true_labels = y_true
    if y_pred.ndim == 2:
        pred_labels = np.argmax(y_pred, axis=1)
    else:
        pred_labels = y_pred
    return np.mean(true_labels == pred_labels)


def f1_score(y_true, y_pred):
    """
    Macro-averaged F1 score (NumPy only, no sklearn).
    Supports both one-hot (N, C) and integer (N,) label formats.
    """
    if y_true.ndim == 2:
        true_labels = np.argmax(y_true, axis=1)
    else:
        true_labels = y_true
    if y_pred.ndim == 2:
        pred_labels = np.argmax(y_pred, axis=1)
    else:
        pred_labels = y_pred

    classes = np.unique(true_labels)
    f1_total = 0

    for cls in classes:
        tp = np.sum((pred_labels == cls) & (true_labels == cls))
        fp = np.sum((pred_labels == cls) & (true_labels != cls))
        fn = np.sum((pred_labels != cls) & (true_labels == cls))

        precision = tp / (tp + fp + 1e-15)
        recall = tp / (tp + fn + 1e-15)
        f1 = 2 * precision * recall / (precision + recall + 1e-15)
        f1_total += f1

    return f1_total / len(classes)


# ════════════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════════════

def plot_metrics(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s,
                 title=None, save_path=None):
    """
    Plot learning curves: loss, accuracy, and F1 score.

    Parameters
    ----------
    title : str, optional
        Super-title for the figure.
    save_path : str or Path, optional
        If provided, saves the figure to this path (PNG, dpi=150).
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, train_losses, label='Train Loss')
    if val_losses is not None:
        axes[0].plot(epochs, val_losses, label='Val Loss')
    axes[0].set_title('Cross-Entropy Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, train_accs, label='Train Acc')
    if val_accs is not None:
        axes[1].plot(epochs, val_accs, label='Val Acc')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1 Score
    axes[2].plot(epochs, train_f1s, label='Train F1')
    if val_f1s is not None:
        axes[2].plot(epochs, val_f1s, label='Val F1')
    axes[2].set_title('Macro F1 Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def confusion_matrix_np(y_true, y_pred, num_classes=None):
    """Compute confusion matrix (NumPy only)."""
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, class_names=None, normalize=False,
                          title="Confusion Matrix", cmap="Blues",
                          save_path=None):
    """
    Plot confusion matrix heatmap.

    Parameters
    ----------
    save_path : str or Path, optional
        If provided, saves the figure to this path (PNG, dpi=150).
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(cm.shape[0])]
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap=cmap)
    plt.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ════════════════════════════════════════════════════════════════════
#  O1.1: TRAINING HISTORY — SAVE / LOAD / COMPARE
# ════════════════════════════════════════════════════════════════════

def save_training_history(history, save_path, model_name="model", metadata=None):
    """
    Save training history (raw data) to an .npz file for later comparison.

    Parameters
    ----------
    history : dict
        Dictionary with keys like 'train_loss', 'val_loss', 'train_acc',
        'val_acc', 'train_f1', 'val_f1', and optionally 'lr'.
    save_path : str or Path
        File path for the .npz output (e.g., 'visualizations/training/history_adam.npz').
    model_name : str
        Human-readable model name stored inside the file (used in legends).
    metadata : dict, optional
        Extra key-value pairs to store (e.g., architecture, lr, epochs).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    data = {k: np.array(v, dtype=np.float64) for k, v in history.items()}
    data['model_name'] = np.array(model_name)

    if metadata:
        for k, v in metadata.items():
            data[f'meta_{k}'] = np.array(v)

    np.savez(str(save_path), **data)
    print(f"Training history saved -> {save_path}  ({model_name})")


def load_training_history(load_path):
    """
    Load a training history .npz file saved by save_training_history.

    Parameters
    ----------
    load_path : str or Path

    Returns
    -------
    history : dict   — metric arrays keyed by name
    model_name : str — the model name stored inside the file
    metadata : dict  — any extra metadata stored with 'meta_' prefix
    """
    data = np.load(str(load_path), allow_pickle=True)
    history = {}
    metadata = {}
    model_name = "unknown"

    for key in data.files:
        if key == 'model_name':
            model_name = str(data[key])
        elif key.startswith('meta_'):
            metadata[key[5:]] = data[key]
        else:
            history[key] = data[key]

    return history, model_name, metadata


def load_multiple_histories(*paths):
    """
    Load multiple training history files for comparison.

    Parameters
    ----------
    *paths : str or Path
        Variable number of paths to .npz history files.

    Returns
    -------
    list of (history_dict, model_name, metadata) tuples
    """
    results = []
    for p in paths:
        h, name, meta = load_training_history(p)
        results.append((h, name, meta))
        print(f"Loaded: {name} ({len(h.get('train_loss', []))} epochs) from {p}")
    return results


def plot_model_comparison(histories, metric='val_loss', title=None, save_path=None):
    """
    Plot a single metric across multiple models for side-by-side comparison.

    Parameters
    ----------
    histories : list of (history_dict, model_name, metadata)
        As returned by load_multiple_histories.
    metric : str
        Key to plot, e.g. 'val_loss', 'val_acc', 'val_f1', 'train_loss', etc.
    title : str, optional
    save_path : str or Path, optional
    """
    METRIC_LABELS = {
        'train_loss': 'Training Loss', 'val_loss': 'Validation Loss',
        'train_acc': 'Training Accuracy', 'val_acc': 'Validation Accuracy',
        'train_f1': 'Training Macro F1', 'val_f1': 'Validation Macro F1',
        'lr': 'Learning Rate',
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for hist, name, _ in histories:
        if metric not in hist:
            print(f"Warning: '{metric}' not found for model '{name}', skipping.")
            continue
        values = hist[metric]
        epochs = np.arange(1, len(values) + 1)
        ax.plot(epochs, values, label=name, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title or f'Model Comparison — {METRIC_LABELS.get(metric, metric)}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_multi_model_summary(histories, save_path=None):
    """
    Generate a 3-panel comparison (Loss, Accuracy, F1) across multiple models.
    Each model's validation curve is plotted with a distinct color and label.

    Parameters
    ----------
    histories : list of (history_dict, model_name, metadata)
        As returned by load_multiple_histories.
    save_path : str or Path, optional
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    metrics = [('val_loss', 'Validation Loss'),
               ('val_acc', 'Validation Accuracy'),
               ('val_f1', 'Validation Macro F1')]

    for ax, (key, label) in zip(axes, metrics):
        for hist, name, _ in histories:
            if key not in hist:
                continue
            values = hist[key]
            epochs = np.arange(1, len(values) + 1)
            ax.plot(epochs, values, label=name, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Multi-Model Comparison (Validation Metrics)', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.show()
    return fig