import pytest
import numpy as np

from anomaly_detection.common.models.ml_io import DetectionOutput, ClassificationOutput, AnomalyCategory, FeatureAttribution
from anomaly_detection.common.models.features import EntityFeatureVector
from src.profiling.profile_model import ExtendedProfilingOutput
from src.explainability.config import ExplainabilityConfig
from src.explainability.engine import ExplainabilityEngine
from src.explainability.feature_phrase_map import HUMAN_LABEL_MAP
from src.explainability.attribution_engine import AttributionEngine

class DummyBPMExplainer:
    def shap_values(self, X):
        # Return negative values so when negated they become positive (pushing toward anomaly)
        # We will make feature 5 (failure_count_norm) the highest
        sv = np.zeros((1, 24))
        sv[0, 5] = -0.9 # failure_count_norm
        sv[0, 20] = -0.5 # session_event_count_norm
        sv[0, 0] = -0.2 # hour_of_day_sin
        return sv

class DummySDMAttribution:
    def get_feature_attributions(self, x, mask):
        # We'll map by feature name
        d = {k: 0.0 for k in HUMAN_LABEL_MAP.keys()}
        d["failure_count_norm"] = 0.8 # confirms BPM
        d["ip_entity_ratio"] = 0.6
        return d

@pytest.fixture
def config():
    return ExplainabilityConfig(top_n_features=3)

@pytest.fixture
def engine(config):
    return ExplainabilityEngine(config, bpm_explainer=DummyBPMExplainer())

@pytest.fixture
def inputs():
    det_out = DetectionOutput(
        entity_id="user_1",
        event_id="evt_1",
        model_id="sdm",
        anomaly_score=0.8,
        confidence=0.9,
        cold_start_flag=False,
        top_contributing_features=["f1"]
    )
    prof_out = ExtendedProfilingOutput(
        entity_id="user_1",
        event_id="evt_1",
        model_id="bpm",
        anomaly_score=0.7,
        confidence=0.8,
        cold_start_flag=False,
        top_contributing_features=["f2"],
        deviation_score=0.75,
        per_feature_deviations={"failure_count_norm": 0.5},
        entity_status="warm",
        is_cold_start=False,
        profile_age=150
    )
    class_out = ClassificationOutput(
        entity_id="user_1",
        event_id="evt_1",
        predicted_class=AnomalyCategory.BRUTE_FORCE,
        class_probabilities={"brute_force": 0.8},
        classification_confidence=0.8,
        is_anomaly=True
    )
    fvec = EntityFeatureVector(root=[0.1] * 24)
    # Set the value for feature 5 to 0.5 (10 failures)
    fvec.root[5] = 0.5
    fvec.root[20] = 0.1 # 20 events
    
    return det_out, class_out, prof_out, fvec

def test_explain_brute_force(engine, inputs):
    det, cls, prof, fvec = inputs
    
    explanation = engine.explain(det, cls, prof, fvec)
    
    assert explanation.predicted_category == AnomalyCategory.BRUTE_FORCE
    assert explanation.consistency_check_passed == True
    assert explanation.is_ambiguous == False
    
    # Narrative check
    assert "Brute force attack detected" in explanation.narrative
    assert "10 consecutive authentication failures" in explanation.narrative
    
    # Feature attributions (capped at 5 from BPM since no SDM)
    assert len(explanation.feature_attributions) == 5
    assert explanation.feature_attributions[0].feature_name == "failure_count_norm"

def test_insider_drift_ambiguity(engine, inputs):
    det, cls, prof, fvec = inputs
    cls.predicted_class = AnomalyCategory.INSIDER_DRIFT
    cls.classification_confidence = 0.5
    
    explanation = engine.explain(det, cls, prof, fvec)
    
    assert explanation.is_ambiguous == True
    assert "Behavioral expansion pattern identified" in explanation.narrative
    assert "role change or new project assignment" in explanation.narrative
    
def test_cold_start_ambiguity(engine, inputs):
    det, cls, prof, fvec = inputs
    prof.is_cold_start = True
    prof.cold_start_flag = True
    
    explanation = engine.explain(det, cls, prof, fvec)
    
    assert explanation.is_ambiguous == True
    assert "Note: entity profile is new" in explanation.narrative
    assert "entity is in cold-start mode" in explanation.ambiguity_reason.lower()

def test_consistency_validator():
    from src.explainability.consistency_validator import ConsistencyValidator
    validator = ConsistencyValidator(threshold=0.33)
    
    # Matching
    attrs = [FeatureAttribution(feature_name="geo_velocity_kmph", feature_value=1.0, attribution_score=1.0, direction="toward_anomaly", source_model="bpm", human_label="Speed")]
    res = validator.validate_explanation_consistency(attrs, "impossible_travel", top_n=3)
    assert res.is_consistent == True
    
    # Mismatch
    attrs = [FeatureAttribution(feature_name="failure_count_norm", feature_value=1.0, attribution_score=1.0, direction="toward_anomaly", source_model="bpm", human_label="Failures")]
    res = validator.validate_explanation_consistency(attrs, "impossible_travel", top_n=3)
    assert res.is_consistent == False

def test_attribution_merging():
    config = ExplainabilityConfig()
    engine = AttributionEngine(config)
    
    top_bpm = [
        FeatureAttribution(feature_name="failure_count_norm", feature_value=0.5, attribution_score=0.8, direction="toward_anomaly", source_model="bpm", human_label="A"),
        FeatureAttribution(feature_name="geo_velocity_kmph", feature_value=0.1, attribution_score=0.2, direction="toward_anomaly", source_model="bpm", human_label="B")
    ]
    
    top_sdm = [
        FeatureAttribution(feature_name="failure_count_norm", feature_value=0.6, attribution_score=0.4, direction="toward_anomaly", source_model="sdm", human_label="A"),
        FeatureAttribution(feature_name="ip_entity_ratio", feature_value=0.9, attribution_score=0.7, direction="toward_anomaly", source_model="sdm", human_label="C")
    ]
    
    merged = engine.merge_attributions(top_bpm, top_sdm)
    assert len(merged) == 3
    
    # failure_count_norm should be averaged: (0.8 + 0.4) / 2 = 0.6
    failure = next(f for f in merged if f.feature_name == "failure_count_norm")
    assert failure.attribution_score == pytest.approx(0.6)
    assert failure.source_model == "bpm+sdm"
