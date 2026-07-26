"""
Anomaly Classifier Implementation (M08).
"""


import lightgbm as lgb
import numpy as np
from anomaly_detection.common.models.enums import AnomalyCategory
from anomaly_detection.common.models.features import EntityFeatureVector
from anomaly_detection.common.models.ml_io import (
    ClassificationOutput,
    DetectionOutput,
    ProfilingOutput,
    UnifiedAnomalySignal,
)

from src.classification.config import ClassifierConfig
from src.classification.dataset import INT_TO_LABEL


class AnomalyClassifier:
    """
    Standalone LightGBM Multi-Class Classifier mapping Detection/Profiling 
    outputs to specific attack categories.
    """

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.model = None

    def load_model(self, model_path: str = None) -> None:
        """
        Load a trained LightGBM model from disk.
        """
        path = model_path or self.config.model_path
        self.model = lgb.Booster(model_file=path)

    def save_model(self, model_path: str = None) -> None:
        """
        Save the trained model to disk.
        """
        if self.model is None:
            raise ValueError("No model is currently loaded or trained.")
        path = model_path or self.config.model_path
        self.model.save_model(path)

    def classify(
        self,
        detection_output: DetectionOutput,
        profiling_output: ProfilingOutput,
        feature_vector: EntityFeatureVector
    ) -> ClassificationOutput:
        """
        Classifies an anomaly using the fused inputs and feature vector.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() or train first.")

        # 1. Assemble the 27-dimensional feature vector
        fvec = feature_vector.root

        anomaly_score = detection_output.anomaly_score

        # profiling_output might be an ExtendedProfilingOutput with deviation_score and profile_age,
        # or we fallback to anomaly_score if it's the standard ProfilingOutput.
        deviation_score = getattr(profiling_output, "deviation_score", profiling_output.anomaly_score)
        profile_age = getattr(profiling_output, "profile_age", 100) # assume fully mature if missing

        age_norm = min(profile_age / self.config.max_profile_age_for_norm, 1.0)

        X = np.array([fvec + [anomaly_score, deviation_score, age_norm]])

        # 2. Predict probabilities for the 8 classes
        # LGBM predict returns shape (1, 8)
        preds = self.model.predict(X)[0]

        # 3. Apply Cold-Start Discounting
        is_cold_start = profiling_output.cold_start_flag or detection_output.cold_start_flag
        if is_cold_start:
            preds = preds * self.config.cold_start_discount_factor

        # 4. Find the winning class among the 8
        max_idx = int(np.argmax(preds))
        max_prob = float(preds[max_idx])
        predicted_label = INT_TO_LABEL[max_idx]

        # 5. Handle UNCLASSIFIED fallback
        # If the highest confidence is below the threshold, fallback to unclassified
        if max_prob < self.config.confidence_threshold:
            predicted_category = AnomalyCategory.UNCLASSIFIED
        else:
            predicted_category = AnomalyCategory(predicted_label)

        # 6. Format the 9-class output distribution
        # Initialize probabilities with 0.0 for all 9 classes
        class_probabilities = {cat.value: 0.0 for cat in AnomalyCategory}

        # Populate the 8 classes from LGBM
        for i, prob in enumerate(preds):
            label = INT_TO_LABEL[i]
            class_probabilities[label] = float(prob)

        # The sum of LGBM preds might be < 1.0 due to the cold-start discount.
        # The remaining probability mass is assigned to 'unclassified'.
        sum_probs = sum(class_probabilities.values())
        class_probabilities[AnomalyCategory.UNCLASSIFIED.value] = max(0.0, 1.0 - sum_probs)

        # 7. Construct and return ClassificationOutput
        return ClassificationOutput(
            entity_id=detection_output.entity_id,
            event_id=detection_output.event_id,
            predicted_class=predicted_category,
            class_probabilities=class_probabilities,
            classification_confidence=max_prob if predicted_category != AnomalyCategory.UNCLASSIFIED else class_probabilities[AnomalyCategory.UNCLASSIFIED.value],
            is_anomaly=True  # Assuming it's an anomaly if it reached this stage, or infer from fused logic elsewhere
        )

    def classify_signal(
        self,
        signal: UnifiedAnomalySignal,
        feature_vector: EntityFeatureVector,
    ) -> ClassificationOutput:
        """Classify a boundary-G signal using the specified 27-feature layout.

        The public legacy ``classify`` method remains for already-trained
        M08 artifacts.  New orchestrator code uses this method so inference
        follows the frozen ML contract: three fused-model scores followed by
        the 24 engineered feature dimensions.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() or train first.")

        classifier_input = np.array(
            [[signal.fused_score, signal.bpm_score, signal.sdm_score, *feature_vector.root]]
        )
        probabilities = self._predict_probabilities(classifier_input)
        class_probabilities = {
            category.value: 0.0 for category in AnomalyCategory
        }
        for index, probability in enumerate(probabilities):
            class_probabilities[INT_TO_LABEL[index]] = float(probability)

        predicted_category = AnomalyCategory(INT_TO_LABEL[int(np.argmax(probabilities))])
        if signal.is_anomaly and predicted_category == AnomalyCategory.NORMAL:
            predicted_category = self._highest_attack_category(class_probabilities)
        elif not signal.is_anomaly:
            predicted_category = AnomalyCategory.NORMAL

        confidence = float(class_probabilities[predicted_category.value])
        if signal.cold_start_flag:
            confidence = min(confidence, self.config.cold_start_discount_factor)

        if signal.is_anomaly and confidence < self.config.confidence_threshold:
            predicted_category = AnomalyCategory.UNCLASSIFIED
            for category in AnomalyCategory:
                if category != AnomalyCategory.UNCLASSIFIED:
                    class_probabilities[category.value] *= confidence
            class_probabilities[AnomalyCategory.UNCLASSIFIED.value] = 1.0 - confidence
            confidence = class_probabilities[AnomalyCategory.UNCLASSIFIED.value]

        return ClassificationOutput(
            entity_id=signal.entity_id,
            event_id=signal.event_id,
            predicted_class=predicted_category,
            class_probabilities=class_probabilities,
            classification_confidence=confidence,
            is_anomaly=signal.is_anomaly,
        )

    def _predict_probabilities(self, classifier_input: np.ndarray) -> np.ndarray:
        """Validate one LightGBM multi-class probability prediction."""
        predicted = np.asarray(self.model.predict(classifier_input)[0], dtype=float)
        if predicted.shape != (len(INT_TO_LABEL),):
            raise ValueError("Classifier model must return probabilities for all 8 classes")
        if np.any(predicted < 0.0) or not np.isclose(predicted.sum(), 1.0):
            raise ValueError("Classifier model probabilities must be non-negative and sum to 1.0")
        return predicted

    @staticmethod
    def _highest_attack_category(class_probabilities: dict[str, float]) -> AnomalyCategory:
        """Choose the most probable non-normal category for a flagged event."""
        attack_categories = [
            category
            for category in AnomalyCategory
            if category not in {AnomalyCategory.NORMAL, AnomalyCategory.UNCLASSIFIED}
        ]
        return max(attack_categories, key=lambda category: class_probabilities[category.value])
