"""
Unit tests for the M15 Evaluation Module.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock

from anomaly_detection.evaluation.evaluator import Evaluator
from anomaly_detection.evaluation.metrics import compute_metrics
from anomaly_detection.common.models.enums import AnomalyCategory


def test_chronological_split():
    """
    Acceptance Criteria 1: Proves chronological split boundary is honored.
    Creates 100 events over 100 days. Evaluator should only take the last 15% (Days 85-100).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = os.path.join(tmpdir, "raw.parquet")
        
        # Create 100 events, spaced 1 day apart
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        
        df = pd.DataFrame({
            "event_id": [f"evt_{i}" for i in range(100)],
            "timestamp": dates.astype(str),
            "entity_id": ["user_1"] * 100
        })
        df.to_parquet(raw_path)
        
        evaluator = Evaluator("test_run", raw_path, "dummy.parquet")
        
        test_split = evaluator.get_test_split()
        
        # Should be last 15 elements
        assert len(test_split) == 15
        assert test_split.iloc[0]["event_id"] == "evt_85"
        assert test_split.iloc[-1]["event_id"] == "evt_99"


def test_label_leakage_rejection():
    """
    Proves that if raw data contains a label, it is immediately rejected.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = os.path.join(tmpdir, "raw.parquet")
        
        df = pd.DataFrame({
            "event_id": ["evt_1"],
            "timestamp": ["2023-01-01T00:00:00Z"],
            "label": ["normal"]
        })
        df.to_parquet(raw_path)
        
        evaluator = Evaluator("test_run", raw_path, "dummy.parquet")
        with pytest.raises(ValueError, match="Label leakage detected"):
            evaluator.get_test_split()


def test_metrics_types_and_ranges():
    """
    Acceptance Criteria 2: Proves metrics are computed with sane ranges.
    """
    classes = [e.value for e in AnomalyCategory]
    
    y_true = np.array(["normal", "brute_force", "impossible_travel", "normal", "normal"])
    y_pred = np.array(["normal", "normal", "impossible_travel", "lateral_movement", "normal"])
    fused_scores = np.array([0.1, 0.4, 0.9, 0.8, 0.2])
    
    # Random probs
    class_probs = np.random.rand(5, len(classes))
    # Normalize to sum to 1
    class_probs = class_probs / class_probs.sum(axis=1, keepdims=True)
    
    risk_tiers = np.array(["low", "medium", "critical", "high", "low"])
    
    metrics = compute_metrics(y_true, y_pred, fused_scores, class_probs, risk_tiers, classes)
    
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["auroc_ovr"] <= 1.0
    assert 0.0 <= metrics["auprc"] <= 1.0
    assert 0.0 <= metrics["precision_at_1_percent"] <= 1.0
    assert 0.0 <= metrics["fpr_at_tpr_90"] <= 1.0
    
    for cls in classes:
        assert 0.0 <= metrics["per_class"][cls]["f1"] <= 1.0


def test_precision_at_1_percent_uses_ranking():
    """
    Acceptance Criteria 3: Proves Precision@1% is computed against fused_score ranking.
    We'll construct a fixture where a naive threshold metric would disagree.
    Suppose we have 100 events, 99 normal, 1 anomaly.
    If naive threshold is 0.5, and all normals have 0.6, they are false positives.
    But our top 1% (which is 1 event) has score 0.9 and is an anomaly.
    Ranking-based precision@1% should be 1.0.
    """
    y_true = np.array(["normal"] * 99 + ["brute_force"]) # 1 anomaly
    y_pred = np.array(["brute_force"] * 100) # Naive threshold predicts everything as anomaly
    
    # Normal events got 0.6 (would be FP under 0.5 threshold)
    # The anomaly got 0.9 (ranked highest)
    fused_scores = np.array([0.6] * 99 + [0.9])
    
    classes = [e.value for e in AnomalyCategory]
    class_probs = np.zeros((100, len(classes)))
    risk_tiers = np.array(["high"] * 100)
    
    metrics = compute_metrics(y_true, y_pred, fused_scores, class_probs, risk_tiers, classes)
    
    # 1% of 100 is 1 event. The highest score is 0.9, which is the anomaly.
    # Therefore, out of the top 1%, 1/1 is an anomaly.
    assert metrics["precision_at_1_percent"] == 1.0
    
    # Note: If it were naive precision, it would be 1 / 100 = 0.01
    naive_precision = metrics["per_class"]["brute_force"]["precision"]
    assert naive_precision == 0.01


def test_no_label_leakage_ast_scan():
    """
    Acceptance Criteria 4: Static test to prove labels_<run_id>.parquet is ONLY referenced
    in label_store.py and evaluator.py.
    """
    import ast
    
    # Walk through the entire src directory
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
    
    violations = []
    
    for root, _, files in os.walk(src_dir):
        for file in files:
            if not file.endswith(".py"):
                continue
                
            file_path = os.path.join(root, file)
            
            # Exempt files
            if "label_store.py" in file_path or "evaluator.py" in file_path:
                continue
                
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Quick string check first for speed
            if "labels_" not in content and "data/labeled" not in content:
                continue
                
            # If string found, do a proper AST parse to ensure it's not a comment,
            # actually we can just flag it if the string appears at all in code.
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        if "labels_" in node.value or "data/labeled" in node.value:
                            violations.append(f"{file_path}: {node.value}")
            except SyntaxError:
                pass
                
    assert len(violations) == 0, f"Label leakage found in unauthorized files: {violations}"
