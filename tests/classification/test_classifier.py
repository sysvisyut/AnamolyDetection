import numpy as np
import pytest
from anomaly_detection.common.models.enums import AnomalyCategory
from anomaly_detection.common.models.features import EntityFeatureVector
from anomaly_detection.common.models.ml_io import DetectionOutput, UnifiedAnomalySignal

from src.classification.classifier import AnomalyClassifier
from src.classification.config import ClassifierConfig
from src.profiling.profile_model import ExtendedProfilingOutput


class MockModel:
    def __init__(self, pred_probs):
        self.pred_probs = pred_probs

    def predict(self, X):
        return self.pred_probs

@pytest.fixture
def config():
    return ClassifierConfig(
        confidence_threshold=0.6,
        cold_start_discount_factor=0.8,
        max_profile_age_for_norm=100
    )

@pytest.fixture
def classifier(config):
    clf = AnomalyClassifier(config)
    # Mocking LightGBM model that returns high confidence for index 4 (LATERAL_MOVEMENT)
    # index 4 maps to LATERAL_MOVEMENT in LABEL_TO_INT
    pred_probs = np.array([[0.05, 0.05, 0.05, 0.05, 0.7, 0.05, 0.02, 0.03]])
    clf.model = MockModel(pred_probs)
    return clf

@pytest.fixture
def inputs():
    fvec = EntityFeatureVector(root=[0.1] * 24)
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
        per_feature_deviations={"f2": 0.5},
        entity_status="warm",
        is_cold_start=False,
        profile_age=150
    )
    return det_out, prof_out, fvec


def test_classify_normal(classifier, inputs):
    det_out, prof_out, fvec = inputs

    out = classifier.classify(det_out, prof_out, fvec)

    assert out.predicted_class == AnomalyCategory.LATERAL_MOVEMENT
    assert out.classification_confidence == 0.7
    assert len(out.class_probabilities) == 9

    # Check that probabilities sum to 1.0
    assert pytest.approx(sum(out.class_probabilities.values())) == 1.0

def test_classify_unclassified_fallback(classifier, inputs):
    det_out, prof_out, fvec = inputs

    # Change config threshold to force fallback
    classifier.config.confidence_threshold = 0.8

    out = classifier.classify(det_out, prof_out, fvec)

    assert out.predicted_class == AnomalyCategory.UNCLASSIFIED
    # Maximum probability was 0.7, so UNCLASSIFIED gets 0.0 probability because 0.7 + other probs = 1.0
    # Wait, the fallback is just the category assignment. The distribution still sums to 1.0.
    # UNCLASSIFIED will have probability 0.0 since sum(LGBM probs) = 1.0.
    # classification_confidence for UNCLASSIFIED is the probability of UNCLASSIFIED, which is 0.0.
    assert out.predicted_class == AnomalyCategory.UNCLASSIFIED
    assert out.classification_confidence == 0.0

def test_classify_cold_start_discount(classifier, inputs):
    det_out, prof_out, fvec = inputs

    # Set cold start
    det_out.cold_start_flag = True

    out = classifier.classify(det_out, prof_out, fvec)

    # 0.7 * 0.8 (discount) = 0.56
    # This is below the default confidence_threshold of 0.6, so it falls back to UNCLASSIFIED!
    assert out.predicted_class == AnomalyCategory.UNCLASSIFIED

    # Check that UNCLASSIFIED got the remaining probability mass
    # Sum of LGBM preds was 1.0. Now it's 0.8. So 0.2 goes to UNCLASSIFIED.
    assert pytest.approx(out.class_probabilities[AnomalyCategory.UNCLASSIFIED.value]) == 0.2
    assert pytest.approx(out.classification_confidence) == 0.2
    assert pytest.approx(sum(out.class_probabilities.values())) == 1.0


def test_classify_signal_uses_boundary_g_and_preserves_anomaly_decision(classifier, inputs):
    """The repaired inference path consumes score fusion's boundary-G signal."""
    _, _, fvec = inputs
    signal = UnifiedAnomalySignal(
        entity_id="user_1",
        event_id="evt_1",
        fused_score=0.8,
        is_anomaly=True,
        bpm_score=0.7,
        sdm_score=0.8,
        cold_start_flag=False,
        contributing_features=["f1", "f2"],
    )

    out = classifier.classify_signal(signal, fvec)

    assert out.predicted_class == AnomalyCategory.LATERAL_MOVEMENT
    assert out.is_anomaly is True
    assert sum(out.class_probabilities.values()) == pytest.approx(1.0)


def test_classify_signal_overrides_normal_for_anomalous_signal(classifier, inputs):
    """A detector-approved anomaly cannot leave classification as normal."""
    _, _, fvec = inputs
    classifier.model = MockModel(
        np.array([[0.8, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02]])
    )
    signal = UnifiedAnomalySignal(
        entity_id="user_1",
        event_id="evt_1",
        fused_score=0.8,
        is_anomaly=True,
        bpm_score=0.7,
        sdm_score=0.8,
        cold_start_flag=False,
        contributing_features=[],
    )

    out = classifier.classify_signal(signal, fvec)

    assert out.predicted_class == AnomalyCategory.UNCLASSIFIED
