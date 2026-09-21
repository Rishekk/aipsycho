import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
    roc_auc_score,
)
from config.constants import PSYCHOLOGICAL_STATES
from src.utils.logger import get_logger

logger = get_logger("metrics")


def evaluate(y_true: List[int], y_pred: List[int],
             y_proba: np.ndarray = None) -> Dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true:  Ground-truth class indices.
        y_pred:  Predicted class indices.
        y_proba: Predicted probabilities (N, num_classes) — needed for AUC.

    Returns:
        Dict of metric name → value.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "f1_weighted":      float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro":         float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "precision_macro":  float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":     float(recall_score(y_true, y_pred, average="macro",    zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report":           classification_report(
                                y_true, y_pred,
                                target_names=PSYCHOLOGICAL_STATES,
                                zero_division=0,
                            ),
    }

    if y_proba is not None:
        try:
            metrics["auc_macro"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["auc_macro"] = None

    return metrics


def per_class_accuracy(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Return accuracy broken down per class."""
    cm = confusion_matrix(y_true, y_pred)
    per_class = {}
    for i, state in enumerate(PSYCHOLOGICAL_STATES):
        if cm[i].sum() > 0:
            per_class[state] = float(cm[i, i] / cm[i].sum())
        else:
            per_class[state] = 0.0
    return per_class


class MetricTracker:
    """
    Accumulates batch-level predictions and labels during an epoch,
    then computes epoch-level metrics in one call.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._y_true:  List[int]   = []
        self._y_pred:  List[int]   = []
        self._y_proba: List[np.ndarray] = []
        self._losses:  List[float] = []

    def update(self, y_true: np.ndarray, y_pred: np.ndarray,
               y_proba: np.ndarray = None, loss: float = None):
        self._y_true.extend(y_true.tolist())
        self._y_pred.extend(y_pred.tolist())
        if y_proba is not None:
            self._y_proba.extend(y_proba.tolist())
        if loss is not None:
            self._losses.append(loss)

    def compute(self) -> Dict:
        proba = np.array(self._y_proba) if self._y_proba else None
        metrics = evaluate(self._y_true, self._y_pred, proba)
        if self._losses:
            metrics["loss"] = float(np.mean(self._losses))
        return metrics
