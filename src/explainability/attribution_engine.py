"""
Attribution Engine (M09).
Computes feature-level attribution scores using BPM (SHAP) and SDM (Captum).
"""

import numpy as np
from typing import List
import shap
import torch

from anomaly_detection.common.models.ml_io import FeatureAttribution
from anomaly_detection.models.sequence_detection.attribution_interface import AttributionInterface
from src.explainability.config import ExplainabilityConfig
from src.explainability.feature_phrase_map import HUMAN_LABEL_MAP

# We use the keys of HUMAN_LABEL_MAP as our canonical FEATURE_NAMES
FEATURE_NAMES = list(HUMAN_LABEL_MAP.keys())


class AttributionEngine:
    """
    Computes and merges feature attributions from BPM and SDM.
    """
    def __init__(self, config: ExplainabilityConfig):
        self.config = config

    def compute_bpm_attributions(self, explainer: shap.TreeExplainer, feature_vector_array: np.ndarray) -> List[FeatureAttribution]:
        """
        Computes SHAP values for the BPM Isolation Forest.
        Expects a shap.TreeExplainer (initialized with the isolation forest).
        """
        if explainer is None:
            return []
            
        try:
            # feature_vector_array should be shape (1, 24)
            shap_values = explainer.shap_values(feature_vector_array)
            # shap_values shape (1, 24)
            # IsolationForest SHAP: positive means higher raw score (more normal).
            # We negate it so positive means pushes anomaly score higher (toward anomaly).
            sv = -1.0 * shap_values[0]
            
            attributions = []
            for d in range(24):
                feat_name = FEATURE_NAMES[d]
                feat_value = float(feature_vector_array[0, d])
                attr_score = float(sv[d])
                direction = "toward_anomaly" if attr_score > 0 else "toward_normal"
                
                attributions.append(FeatureAttribution(
                    feature_name=feat_name,
                    feature_value=feat_value,
                    attribution_score=attr_score,
                    direction=direction,
                    source_model="bpm",
                    human_label=HUMAN_LABEL_MAP[feat_name]["human_label"]
                ))
                
            # Sort by absolute magnitude descending
            attributions.sort(key=lambda x: abs(x.attribution_score), reverse=True)
            return attributions[:5]
        except Exception as e:
            # Fallback on failure
            return []

    def compute_sdm_attributions(self, sdm_attribution_iface: AttributionInterface, sequence_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> List[FeatureAttribution]:
        """
        Computes Integrated Gradients for the SDM via the provided interface.
        """
        if sdm_attribution_iface is None:
            return []
            
        try:
            # returns Dict[str, float] with signed mean IG values
            attr_dict = sdm_attribution_iface.get_feature_attributions(sequence_tensor, sequence_mask)
            
            # Extract last valid position values for feature_value
            last_idx = (sequence_mask.sum(dim=1) - 1)[0].item()
            last_idx = max(0, int(last_idx))
            
            attributions = []
            for d, feat_name in enumerate(FEATURE_NAMES):
                attr_score = float(attr_dict.get(feat_name, 0.0))
                feat_value = float(sequence_tensor[0, last_idx, d].item())
                direction = "toward_anomaly" if attr_score > 0 else "toward_normal"
                
                attributions.append(FeatureAttribution(
                    feature_name=feat_name,
                    feature_value=feat_value,
                    attribution_score=attr_score,
                    direction=direction,
                    source_model="sdm",
                    human_label=HUMAN_LABEL_MAP[feat_name]["human_label"]
                ))
                
            attributions.sort(key=lambda x: abs(x.attribution_score), reverse=True)
            return attributions[:5]
        except Exception as e:
            # Fallback on failure
            return []

    def merge_attributions(self, top_bpm: List[FeatureAttribution], top_sdm: List[FeatureAttribution]) -> List[FeatureAttribution]:
        """
        Merges BPM and SDM attributions, deduplicating and averaging scores where both cite a feature.
        """
        merged_map = {}
        
        # Add BPM features
        for attr in top_bpm:
            merged_map[attr.feature_name] = attr
            
        # Add or merge SDM features
        for attr in top_sdm:
            if attr.feature_name in merged_map:
                existing = merged_map[attr.feature_name]
                # Average the scores
                avg_score = (existing.attribution_score + attr.attribution_score) / 2.0
                new_direction = "toward_anomaly" if avg_score > 0 else "toward_normal"
                
                merged_map[attr.feature_name] = FeatureAttribution(
                    feature_name=attr.feature_name,
                    feature_value=existing.feature_value, # take from BPM (current event)
                    attribution_score=avg_score,
                    direction=new_direction,
                    source_model="bpm+sdm",
                    human_label=attr.human_label
                )
            else:
                merged_map[attr.feature_name] = attr
                
        # Sort by absolute magnitude descending
        merged_list = list(merged_map.values())
        merged_list.sort(key=lambda x: abs(x.attribution_score), reverse=True)
        
        # Enforce max 10 entries
        return merged_list[:10]
