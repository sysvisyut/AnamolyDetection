"""
Evaluation metrics logic for Model Evaluation (M15).
Computes §9.4 metrics, avoiding label leakage by taking array inputs.
"""

from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve
)
from collections import Counter
from anomaly_detection.common.models.enums import AnomalyCategory


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fused_scores: np.ndarray,
    class_probs: np.ndarray,
    risk_tiers: np.ndarray,
    classes: List[str]
) -> Dict[str, Any]:
    """
    Computes exactly the §9.4 metric set.
    
    Args:
        y_true: True class labels (string).
        y_pred: Predicted class labels (string).
        fused_scores: 1D array of fused scores (for budget metrics).
        class_probs: 2D array of class probabilities (N x num_classes).
        risk_tiers: 1D array of risk tiers assigned to each event.
        classes: List of class names, where indices match class_probs columns.
        
    Returns:
        Dictionary of metrics per §9.4.
    """
    # Create binary labels for anomaly detection (normal vs anomaly)
    y_true_binary = (y_true != AnomalyCategory.NORMAL.value).astype(int)
    
    metrics: Dict[str, Any] = {}
    
    # 1. Per-class Precision, Recall, F1
    precisions = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    
    metrics["per_class"] = {}
    for i, cls in enumerate(classes):
        metrics["per_class"][cls] = {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i])
        }
        
    # 2. Macro-F1
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
    
    # 3. AUROC (one-vs-rest per class)
    # class_probs must correspond exactly to the order of `classes`
    try:
        metrics["auroc_ovr"] = float(roc_auc_score(y_true, class_probs, labels=classes, multi_class="ovr", average="macro"))
    except ValueError:
        # Fallback if only one class exists in y_true
        metrics["auroc_ovr"] = 0.0
        
    # 4. AUPRC (binary anomaly vs. normal)
    # We use fused_scores as the confidence of it being anomalous
    try:
        metrics["auprc"] = float(average_precision_score(y_true_binary, fused_scores))
    except ValueError:
        metrics["auprc"] = 0.0
        
    # 5. Precision@1% (binary) against fused_score ranking
    # The analyst alert budget is top 1%.
    if len(fused_scores) > 0:
        top_1_percent_k = max(1, int(0.01 * len(fused_scores)))
        # Argsort ascending, so take the last k for descending
        top_k_indices = np.argsort(fused_scores)[-top_1_percent_k:]
        top_k_true = y_true_binary[top_k_indices]
        precision_at_1 = float(np.sum(top_k_true) / top_1_percent_k)
    else:
        precision_at_1 = 0.0
    metrics["precision_at_1_percent"] = precision_at_1
    
    # 6. FPR@TPR=0.9
    # Use ROC curve to find FPR when TPR reaches >= 0.9
    if np.sum(y_true_binary) > 0 and np.sum(1 - y_true_binary) > 0:
        fpr, tpr, thresholds = roc_curve(y_true_binary, fused_scores)
        # Find first index where TPR >= 0.9
        idx = np.searchsorted(tpr, 0.9, side="left")
        if idx < len(fpr):
            metrics["fpr_at_tpr_90"] = float(fpr[idx])
        else:
            metrics["fpr_at_tpr_90"] = float(fpr[-1])
    else:
        metrics["fpr_at_tpr_90"] = 0.0
        
    # 7. Risk tier distribution of true positives
    tp_indices = (y_true_binary == 1) & (y_true == y_pred)  # Correctly classified attacks
    tp_tiers = risk_tiers[tp_indices]
    counts = Counter(tp_tiers)
    metrics["tp_risk_tier_distribution"] = {
        "critical": counts.get("critical", 0),
        "high": counts.get("high", 0),
        "medium": counts.get("medium", 0),
        "low": counts.get("low", 0)
    }
    
    return metrics
