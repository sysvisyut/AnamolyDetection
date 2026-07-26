"""
Configuration for the Anomaly Classifier (M08).
"""
from typing import Dict, Any
from pydantic import BaseModel, Field

class ClassifierConfig(BaseModel):
    """
    Configuration parameters for the Anomaly Classifier.
    """
    # Threshold below which predictions fall back to UNCLASSIFIED
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    
    # Discount applied to the max probability if the entity is in cold start
    cold_start_discount_factor: float = Field(default=0.8, ge=0.0, le=1.0)
    
    # Label smoothing applied to INSIDER_DRIFT during training to preserve ambiguity
    insider_drift_smoothing: float = Field(default=0.2, ge=0.0, le=1.0)
    
    # Max profile age for normalization (e.g. 100 events = fully mature for normalization purposes)
    max_profile_age_for_norm: int = Field(default=100, ge=1)
    
    # LightGBM hyperparameters
    lgbm_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "objective": "multiclass",
            "num_class": 8,  # 8 classes: 7 attacks + normal
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "is_unbalance": False,  # We use custom class weights instead
            "verbosity": -1,
            "random_state": 42
        }
    )
    
    model_path: str = Field(default="data/models/classifier_model.txt")
