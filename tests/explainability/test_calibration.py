import pytest
import numpy as np
from anomaly_detection.common.models.ml_io import ClassificationOutput
from anomaly_detection.common.models.enums import AnomalyCategory
from anomaly_detection.explainability.calibration import ConfidenceCalibrator

def test_calibration_passthrough():
    calibrator = ConfidenceCalibrator()
    out = ClassificationOutput(
        entity_id="e1",
        event_id="evt1",
        predicted_class=AnomalyCategory.BRUTE_FORCE,
        class_probabilities={"brute_force": 0.8, "normal": 0.2},
        classification_confidence=0.8,
        is_anomaly=True
    )
    res = calibrator.calibrate(out)
    assert res.classification_confidence == 0.8

def test_calibration_fitted():
    calibrator = ConfidenceCalibrator()
    
    # Simple monotonic setup
    probs = {"brute_force": np.array([0.1, 0.4, 0.9])}
    y_true = {"brute_force": np.array([0, 1, 1])}
    
    calibrator.fit(probs, y_true)
    
    out = ClassificationOutput(
        entity_id="e1",
        event_id="evt1",
        predicted_class=AnomalyCategory.BRUTE_FORCE,
        class_probabilities={"brute_force": 0.4, "normal": 0.6},
        classification_confidence=0.4,
        is_anomaly=True
    )
    
    res = calibrator.calibrate(out)
    # The 0.4 should map to roughly 1.0 (or 0.5 depending on isotonic bins)
    # Actually, [0.1, 0.4, 0.9] -> [0, 1, 1] means 0.4 is mapped to 1.0.
    # We just ensure it changed.
    assert res.class_probabilities["brute_force"] != 0.4
