"""
Top-level Explainability Engine (M09).
"""

from typing import Optional
import shap

from anomaly_detection.common.models.ml_io import DetectionOutput, ProfilingOutput, ClassificationOutput, Explanation, AnomalyCategory
from anomaly_detection.common.models.features import EntityFeatureVector
from anomaly_detection.models.sequence_detection.attribution_interface import AttributionInterface

from src.explainability.config import ExplainabilityConfig
from src.explainability.attribution_engine import AttributionEngine
from src.explainability.narrative_translator import NarrativeTranslator
from src.explainability.consistency_validator import ConsistencyValidator


class ExplainabilityEngine:
    """
    Orchestrates the Explainability Layer sub-components.
    """
    def __init__(
        self, 
        config: ExplainabilityConfig, 
        bpm_explainer: Optional[shap.TreeExplainer] = None, 
        sdm_attribution_iface: Optional[AttributionInterface] = None
    ):
        self.config = config
        self.attribution_engine = AttributionEngine(config)
        self.narrative_translator = NarrativeTranslator(config)
        self.consistency_validator = ConsistencyValidator(config.consistency_threshold)
        self.bpm_explainer = bpm_explainer
        self.sdm_attribution_iface = sdm_attribution_iface

    def explain(
        self,
        detection_output: DetectionOutput,
        classification_output: ClassificationOutput,
        profiling_output: ProfilingOutput,
        feature_vector: EntityFeatureVector,
        sequence_tensor: Optional['torch.Tensor'] = None,
        sequence_mask: Optional['torch.Tensor'] = None,
        raw_snapshot: Optional[dict] = None
    ) -> Explanation:
        """
        Generates an Explanation object.
        """
        predicted_class_str = classification_output.predicted_class.value
        
        # 1. Compute Attributions
        import numpy as np
        fvec_array = np.array([feature_vector.root], dtype=np.float32)
        
        top_bpm = self.attribution_engine.compute_bpm_attributions(self.bpm_explainer, fvec_array)
        
        top_sdm = []
        if self.sdm_attribution_iface and sequence_tensor is not None and sequence_mask is not None:
            top_sdm = self.attribution_engine.compute_sdm_attributions(self.sdm_attribution_iface, sequence_tensor, sequence_mask)
            
        merged_attributions = self.attribution_engine.merge_attributions(top_bpm, top_sdm)
        
        # Fallback if both fail
        if not merged_attributions:
            # Create dummy to avoid crash
            # But the criteria says explainability must never crash the pipeline, fallback on per_feature_deviations
            deviations = getattr(profiling_output, "per_feature_deviations", {})
            for feat, val in sorted(deviations.items(), key=lambda item: abs(item[1]), reverse=True)[:5]:
                from src.explainability.feature_phrase_map import HUMAN_LABEL_MAP
                from anomaly_detection.common.models.ml_io import FeatureAttribution
                direction = "toward_anomaly" if val > 0 else "toward_normal"
                human = HUMAN_LABEL_MAP.get(feat, {}).get("human_label", feat)
                merged_attributions.append(FeatureAttribution(
                    feature_name=feat,
                    feature_value=0.0, # We don't have the original value easily here, just a fallback
                    attribution_score=float(val),
                    direction=direction,
                    source_model="bpm",
                    human_label=human
                ))

        # 2. Validate Consistency
        val_result = self.consistency_validator.validate_explanation_consistency(
            merged_attributions, predicted_class_str, self.config.top_n_features
        )
        
        # 3. Determine Ambiguity
        is_ambiguous = False
        ambiguity_reason = None
        
        cold_start = getattr(profiling_output, "is_cold_start", getattr(profiling_output, "cold_start_flag", False))
        
        if predicted_class_str == "insider_drift":
            is_ambiguous = True
            ambiguity_reason = "Predicted class is insider drift, which is inherently ambiguous between legitimate role changes and malicious activity."
        elif classification_output.classification_confidence < self.config.ambiguity_threshold:
            is_ambiguous = True
            ambiguity_reason = f"Low classification confidence ({classification_output.classification_confidence:.2f} < threshold {self.config.ambiguity_threshold})."
        elif cold_start:
            is_ambiguous = True
            ambiguity_reason = "Entity is in cold-start mode; anomaly scoring relies on population priors rather than established individual baselines."
            
        # 4. Generate Narrative
        narrative = self.narrative_translator.translate(
            feature_attributions=merged_attributions,
            predicted_class=predicted_class_str,
            classification_confidence=classification_output.classification_confidence,
            fused_score=getattr(classification_output, "fused_score", 1.0),
            cold_start_flag=cold_start,
            entity_id=classification_output.entity_id,
            raw_snapshot=raw_snapshot,
            consistency_warning=not val_result.is_consistent
        )
        
        # 5. Return Explanation
        return Explanation(
            narrative=narrative,
            feature_attributions=merged_attributions,
            predicted_category=classification_output.predicted_class,
            consistency_check_passed=val_result.is_consistent,
            is_ambiguous=is_ambiguous,
            ambiguity_reason=ambiguity_reason
        )
