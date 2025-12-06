"""
evaluate.py
Computes accuracy, precision, recall, F1 (macro for imbalance), ROC-AUC.
Prints metrics and a note about probability calibration.
"""

from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate(clf, X_test, y_test) -> Dict[str, float]:
    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }
    try:
        y_prob = clf.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        metrics["calibration_note"] = 1
    except Exception:
        metrics["roc_auc"] = float("nan")
        metrics["calibration_note"] = 0
    return metrics

def print_metrics(metrics: Dict[str, float]) -> None:
    print("=== Test Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    if metrics.get("calibration_note") == 1:
        print("Note: Probabilities available; consider calibration (e.g., isotonic) if thresholding decisions.")
