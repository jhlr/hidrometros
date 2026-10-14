"""Classification metrics. `classification_metrics` is from the group scaffold (Mikael)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

IntArray = npt.NDArray[np.int64]


def classification_metrics(y_true: IntArray, y_pred: IntArray) -> dict[str, float | IntArray]:
    """Per-crop accuracy, macro F1 over the 10 classes and the 10x10 confusion matrix."""
    return {
        "acc_digit": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=list(range(10)), zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(10))),
    }


def reading_accuracy(image_ids: IntArray, y_true: IntArray, y_pred: IntArray) -> float:
    """Fraction of images whose crops are ALL predicted correctly (all 8 digits)."""
    ok: dict[int, bool] = {}
    for i, t, p in zip(image_ids.tolist(), y_true.tolist(), y_pred.tolist(), strict=True):
        ok[i] = ok.get(i, True) and t == p
    return float(np.mean(list(ok.values())))
