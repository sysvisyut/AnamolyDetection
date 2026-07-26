"""
Consistency Validator (M09).
Ensures that the top cited features are plausibly associated with the predicted attack class.
"""

from typing import List, Set, Tuple
from pydantic import BaseModel

from anomaly_detection.common.models.ml_io import FeatureAttribution
from anomaly_detection.common.logging import setup_logger

logger = setup_logger("consistency_validator")

class ValidationResult(BaseModel):
    is_consistent: bool
    consistency_score: float

# Primary affected fields mapped directly from ATTACK_TAXONOMY.md
EXPECTED_PRIMARY_FEATURES = {
    "brute_force":          {"failure_count_norm", "auth_outcome_enc", "session_event_count_norm", "ip_entity_ratio"},
    "impossible_travel":    {"geo_velocity_kmph", "is_new_geo"},
    "credential_stuffing":  {"ip_entity_ratio", "failure_count_norm", "auth_outcome_enc", "entity_ip_ratio"},
    "lateral_movement":     {"resource_rarity_score", "resource_breadth_norm", "has_exfil_command", "command_seq_length_norm", "command_rarity_score"},
    "device_spoofing":      {"fingerprint_mac_match", "fingerprint_os_match", "fingerprint_protocol_match"},
    "low_and_slow":         {"has_exfil_command", "hour_of_day_sin", "hour_of_day_cos", "inter_event_gap_norm", "session_duration_norm"},
    "insider_drift":        {"resource_rarity_score", "resource_breadth_norm", "command_rarity_score"},
    "unclassified":         set(),
    "normal":               set()
}

class ConsistencyValidator:
    def __init__(self, threshold: float = 0.33):
        self.threshold = threshold

    def validate_explanation_consistency(self, feature_attributions: List[FeatureAttribution], predicted_class: str, top_n: int = 3) -> ValidationResult:
        """
        Validates if the top cited features overlap with the expected primary features.
        Never blocks execution. Emits a warning if consistency is low.
        """
        expected = EXPECTED_PRIMARY_FEATURES.get(predicted_class, set())
        
        # If no expected features defined (e.g. unclassified, normal), we assume consistent
        if not expected:
            return ValidationResult(is_consistent=True, consistency_score=1.0)
            
        # Get top-N features pushing toward anomaly
        cited_features = {fa.feature_name for fa in feature_attributions[:top_n] if fa.direction == "toward_anomaly"}
        
        if not cited_features:
            return ValidationResult(is_consistent=True, consistency_score=1.0)
            
        overlap = cited_features.intersection(expected)
        consistency_score = len(overlap) / max(len(expected), 1.0)
        
        if consistency_score >= self.threshold:
            return ValidationResult(is_consistent=True, consistency_score=consistency_score)
            
        # Fallback Mode: Check if top-1 feature is in expected
        top_1 = [fa for fa in feature_attributions if fa.direction == "toward_anomaly"]
        if top_1 and top_1[0].feature_name in expected:
            # Case A: Override
            return ValidationResult(is_consistent=True, consistency_score=consistency_score)
            
        # Case B: Warning, return False
        top_feature = top_1[0].feature_name if top_1 else "None"
        logger.warning(f"Consistency check failed: top feature {top_feature} not in expected set for {predicted_class}. Consistency score: {consistency_score:.2f}")
        
        return ValidationResult(is_consistent=False, consistency_score=consistency_score)
