"""
Confidence Calibration (T3 Polish).
Applies post-processing (Platt scaling or Isotonic Regression) to raw classifier outputs
to produce well-calibrated confidence scores without affecting upstream pipeline logic.
"""

import numpy as np
from typing import Dict, List, Any
from sklearn.isotonic import IsotonicRegression

from anomaly_detection.common.models.ml_io import ClassificationOutput

class ConfidenceCalibrator:
    """
    Applies Isotonic Regression to calibrate classifier confidences.
    """
    
    def __init__(self):
        self.calibrators: Dict[str, IsotonicRegression] = {}
        self.is_fitted = False
        
    def fit(self, probs_dict: Dict[str, np.ndarray], y_true_binary_dict: Dict[str, np.ndarray]) -> None:
        """
        Fits a separate calibrator for each class (One-vs-Rest).
        
        Args:
            probs_dict: Map of class_name -> 1D array of raw predicted probabilities for that class.
            y_true_binary_dict: Map of class_name -> 1D array of binary ground truth (1 if that class, 0 otherwise).
        """
        for cls, probs in probs_dict.items():
            y_true = y_true_binary_dict[cls]
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs, y_true)
            self.calibrators[cls] = ir
            
        self.is_fitted = True
        
    def calibrate(self, classification: ClassificationOutput) -> ClassificationOutput:
        """
        Applies calibration to a single ClassificationOutput.
        Has zero blast radius to upstream components as it returns a modified copy.
        """
        if not self.is_fitted:
            # If not fitted, act as a transparent passthrough
            return classification
            
        calibrated_probs = {}
        for cls, prob in classification.class_probabilities.items():
            if cls in self.calibrators:
                # IsotonicRegression.predict expects 1D array
                calibrated = float(self.calibrators[cls].predict([prob])[0])
            else:
                calibrated = prob
            calibrated_probs[cls] = calibrated
            
        # Re-normalize just in case
        total = sum(calibrated_probs.values())
        if total > 0:
            for cls in calibrated_probs:
                calibrated_probs[cls] /= total
                
        # The predicted class is usually NOT changed by calibration, 
        # but the confidence IS updated.
        predicted_class = classification.predicted_class
        new_confidence = calibrated_probs.get(predicted_class.value, classification.classification_confidence)
        
        return ClassificationOutput(
            entity_id=classification.entity_id,
            event_id=classification.event_id,
            predicted_class=predicted_class,
            class_probabilities=calibrated_probs,
            classification_confidence=new_confidence,
            is_anomaly=classification.is_anomaly
        )
