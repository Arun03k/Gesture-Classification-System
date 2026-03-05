"""
HelperFunctions.py — Backward-compatible re-exports from the ml_framework package.

All ML utility functions (activations, metrics, losses, plotting, history I/O)
are provided by the pip-installed ml_framework package.  This module re-exports
them so that existing project code can continue to ``from helper_functions import …``
without changes.
"""

# ── Activations ───────────────────────────────────────────────────
from ml_framework.layers.activations import (          # noqa: F401
    relu, relu_derivative,
    sigmoid, sigmoid_derivative,
    softmax,
)

# ── Metrics ───────────────────────────────────────────────────────
from ml_framework.metrics import (                     # noqa: F401
    accuracy,
    f1_score,
    confusion_matrix as confusion_matrix_np,
)

# ── Visualization ─────────────────────────────────────────────────
from ml_framework.visualization.training_plots import ( # noqa: F401
    plot_metrics,
    plot_confusion_matrix,
)
from ml_framework.visualization.comparison_plots import ( # noqa: F401
    plot_model_comparison,
    plot_multi_model_summary,
)

# ── History I/O ───────────────────────────────────────────────────
from ml_framework.utils import (                       # noqa: F401
    save_training_history,
    load_training_history,
    load_multiple_histories,
)

# ── Loss (backward-compatible functional wrapper) ─────────────────
from ml_framework.losses import CrossEntropy as _CE


def cross_entropy_loss(y_true, y_pred, class_weights=None):
    """Backward-compatible cross-entropy loss wrapper."""
    return _CE(class_weights=class_weights).forward(y_true, y_pred)