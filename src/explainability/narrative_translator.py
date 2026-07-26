"""
Narrative Translator (M09).
Translates ranked feature attributions into human-readable narrative text.
"""

from typing import List, Optional
import math

from anomaly_detection.common.models.ml_io import FeatureAttribution
from src.explainability.config import ExplainabilityConfig
from src.explainability.feature_phrase_map import HUMAN_LABEL_MAP
from src.explainability.templates import PREFIXES, INSIDER_DRIFT_TEMPLATE, COLD_START_MODIFIER


class NarrativeTranslator:
    def __init__(self, config: ExplainabilityConfig):
        self.config = config

    def _decode_circular_feature(self, sin_val: float, cos_val: float, period: int = 24) -> str:
        """
        Decodes a sin/cos encoded feature back into its original scale.
        """
        val = (math.atan2(sin_val, cos_val) * period / (2 * math.pi)) % period
        
        if period == 24:
            hour = round(val) % 24
            return f"{hour:02d}:00"
        elif period == 7:
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_idx = round(val) % 7
            return days[day_idx]
        return str(round(val))

    def _denormalize_value(self, feature_name: str, value: float, raw_snapshot: Optional[dict] = None) -> Any:
        """
        Denormalizes values for narrative inclusion based on the feature type.
        """
        if feature_name == "geo_velocity_kmph":
            return round(value * 2000)
        elif feature_name == "failure_count_norm":
            return round(value * 20)
        elif feature_name == "session_event_count_norm":
            return round(value * 200)
        elif feature_name == "ip_entity_ratio":
            return round(value * 10)
        elif feature_name == "entity_ip_ratio":
            return round(value * 10)
        elif feature_name == "resource_breadth_norm":
            return round(value * 50)
        elif feature_name == "command_seq_length_norm":
            return round(value * 50)
        elif feature_name == "inter_event_gap_norm":
            return round(value * 24)
        elif feature_name == "session_duration_norm":
            return f"{round(value * 3600)} sec"
        elif feature_name == "is_new_geo" and raw_snapshot:
            # Try to get city/country from raw snapshot if available
            return raw_snapshot.get("geo_location", {}).get("country", "unknown country")
        return value

    def _format_phrase(self, attr: FeatureAttribution, raw_snapshot: Optional[dict] = None) -> str:
        """
        Formats a single feature phrase using the HUMAN_LABEL_MAP template.
        """
        template_str = HUMAN_LABEL_MAP[attr.feature_name][attr.direction]
        
        # Handle special binary device matches
        if attr.feature_name in ["fingerprint_mac_match", "fingerprint_os_match", "fingerprint_protocol_match"]:
            if not raw_snapshot:
                # Fallback if no snapshot provided
                return template_str.format(device_id="unknown", expected_os="unknown", actual_os="unknown", expected_mac="unknown", actual_mac="unknown", expected_proto="unknown", actual_proto="unknown")
            
            fingerprint = raw_snapshot.get("device_fingerprint", {})
            return template_str.format(
                device_id=fingerprint.get("device_id", "unknown"),
                expected_os="registered profile",
                actual_os=f"{fingerprint.get('os_family', 'unknown')}/{fingerprint.get('os_version', 'unknown')}",
                expected_mac="registered profile",
                actual_mac=fingerprint.get("mac_address", "unknown"),
                expected_proto="registered profile",
                actual_proto=fingerprint.get("protocol", "unknown")
            )
            
        denorm_val = self._denormalize_value(attr.feature_name, attr.feature_value, raw_snapshot)
        return template_str.format(value=denorm_val)

    def _build_insider_drift(self, entity_id: str, classification_confidence: float, feature_attributions: List[FeatureAttribution]) -> str:
        """
        Builds the specific override template for Insider Drift.
        """
        # Find resource breadth
        rb = next((fa for fa in feature_attributions if fa.feature_name == "resource_breadth_norm"), None)
        new_resource_count = round(rb.feature_value * 50) if rb else "several"
        
        # Find rarity
        rr = next((fa for fa in feature_attributions if fa.feature_name == "resource_rarity_score"), None)
        rarity = rr.feature_value if rr else 0.5
        
        # Default category and drift_days (could be extracted from raw event snapshot or feature sequence length)
        category = "general"
        drift_days = "recent"
        
        return INSIDER_DRIFT_TEMPLATE.format(
            prefix=PREFIXES.get("insider_drift"),
            entity_id=entity_id,
            new_resource_count=new_resource_count,
            drift_days=drift_days,
            category=category,
            rarity=rarity,
            confidence=classification_confidence
        )

    def translate(
        self, 
        feature_attributions: List[FeatureAttribution], 
        predicted_class: str, 
        classification_confidence: float, 
        fused_score: float, 
        cold_start_flag: bool,
        entity_id: str,
        raw_snapshot: Optional[dict] = None,
        consistency_warning: bool = False
    ) -> str:
        """
        Translates attributes into a narrative.
        """
        if predicted_class == "normal" or predicted_class not in PREFIXES:
            return ""

        if predicted_class == "insider_drift":
            narrative = self._build_insider_drift(entity_id, classification_confidence, feature_attributions)
            # Cap at max length
            if len(narrative) > self.config.max_narrative_length:
                narrative = narrative[:self.config.max_narrative_length - 3] + "..."
            return narrative

        # 1. Select top features
        filtered_attrs = [fa for fa in feature_attributions if fa.direction == "toward_anomaly" and abs(fa.attribution_score) > self.config.attribution_threshold]
        
        # Merge circular encodings (hour and day)
        hour_sin = next((fa for fa in filtered_attrs if fa.feature_name == "hour_of_day_sin"), None)
        hour_cos = next((fa for fa in filtered_attrs if fa.feature_name == "hour_of_day_cos"), None)
        
        merged_attrs = []
        skip = set()
        
        for attr in filtered_attrs:
            if attr.feature_name in skip:
                continue
                
            if attr.feature_name in ["hour_of_day_sin", "hour_of_day_cos"] and hour_sin and hour_cos:
                hour_str = self._decode_circular_feature(hour_sin.feature_value, hour_cos.feature_value, 24)
                # Create a merged phrase manually
                phrase = f"an off-hours login at {hour_str} UTC"
                merged_attrs.append(phrase)
                skip.update(["hour_of_day_sin", "hour_of_day_cos"])
            else:
                merged_attrs.append(self._format_phrase(attr, raw_snapshot))
                
        # Take top N
        top_phrases = merged_attrs[:self.config.top_n_features]
        
        if not top_phrases:
            top_phrases = ["anomalous signal values"]
            
        # 3. Compose sentence
        if len(top_phrases) == 1:
            evidence = f"Flagged due to {top_phrases[0]}."
        elif len(top_phrases) == 2:
            evidence = f"Flagged due to {top_phrases[0]} combined with {top_phrases[1]}."
        elif len(top_phrases) == 3:
            evidence = f"Flagged due to {top_phrases[0]}, {top_phrases[1]}, and {top_phrases[2]}."
        else:
            evidence = f"Flagged due to {top_phrases[0]}, {top_phrases[1]}, {top_phrases[2]}, and {top_phrases[3]}."
            
        # 4. Prepend context
        prefix = PREFIXES.get(predicted_class, PREFIXES["unclassified"])
        narrative = f"{prefix}. {evidence}"
        
        # 5. Append qualifiers
        if classification_confidence < self.config.ambiguity_threshold:
            narrative += f" (classification confidence: {classification_confidence:.0%})"
            
        if consistency_warning:
            top_feature = feature_attributions[0].feature_name if feature_attributions else "unknown"
            narrative += f" [Note: primary attribution signal ({top_feature}) is atypical for this classification — analyst verification recommended.]"
            
        if cold_start_flag:
            narrative += COLD_START_MODIFIER
            
        # 6. Truncate
        if len(narrative) > self.config.max_narrative_length:
            narrative = narrative[:self.config.max_narrative_length - 3] + "..."
            
        return narrative
