import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def classification_metrics(y_true, y_pred):
    return {
        "acc_digit": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(10))),
    }
