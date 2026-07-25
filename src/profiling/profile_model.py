"""
Behavioral Profiling Model (BPM) Implementation (M06).

Scores incoming EntityFeatureVectors using a statistical Z-score approach.
Provides graduation logic from cold-start to warm.
"""

import math
from typing import Dict, List, Literal, Any

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pydantic import Field

from anomaly_detection.common.models.ml_io import ModelScore
from anomaly_detection.common.models.features import EntityFeatureVector
from anomaly_detection.common.models.enums import EntityStatus, EntityType
from anomaly_detection.common.models.entities import EntityProfile

from src.profiling.config import ProfilingConfig, FEATURE_NAMES
from src.profiling.population_prior import PopulationPrior
from src.profiling.profile_store import ProfileStore


class ExtendedProfilingOutput(ModelScore):
    """
    Output of the Behavioral Profiling Model.
    Inherits from M02's ModelScore and adds fields specifically required
    by the M09 Explainability layer.
    """
    model_id: Literal["bpm"] = Field("bpm", description="BPM model identifier")
    
    deviation_score: float = Field(..., ge=0.0, le=1.0, description="Normalized deviation score [0, 1]")
    per_feature_deviations: Dict[str, float] = Field(..., description="Signed Z-scores keyed by feature name")
    entity_status: EntityStatus = Field(..., description="cold_start or warm")
    is_cold_start: bool = Field(..., description="True if entity has < graduation_threshold events")
    profile_age: int = Field(..., ge=0, description="Number of historical events seen for this entity")


class BehavioralProfilingModel:
    """
    Statistical anomaly detection model.
    Scores events based on their deviation from the entity's historical mean,
    using population-level statistics as a fallback for cold-start entities.
    """

    def __init__(
        self,
        config: ProfilingConfig,
        store: ProfileStore,
        prior: PopulationPrior
    ) -> None:
        self.config = config
        self.store = store
        self.prior = prior

    def fit(self, training_data: pd.DataFrame) -> None:
        """
        Builds per-entity profiles and the population prior from a training DataFrame.
        
        Args:
            training_data: A DataFrame containing at least 'entity_id', 'entity_type', 
                           and 'feature_vector' (a 24-element list/array of floats) columns.
        """
        if training_data.empty:
            return

        all_vectors: List[EntityFeatureVector] = []
        all_types: List[EntityType] = []

        now_str = datetime.now(timezone.utc).isoformat()

        # Group by entity_id to compute personal baselines
        for entity_id, group in training_data.groupby('entity_id'):
            # All events for this entity should have the same entity_type
            etype_str = group['entity_type'].iloc[0]
            if isinstance(etype_str, EntityType):
                etype = etype_str
            else:
                etype = EntityType(etype_str)

            # Extract feature vectors as a 2D numpy array
            vectors = np.stack(group['feature_vector'].values)
            
            mean_vec = np.mean(vectors, axis=0).tolist()
            std_vec = np.std(vectors, axis=0).tolist()
            
            # Apply epsilon to std
            std_vec = [max(s, self.config.variance_epsilon) for s in std_vec]
            
            event_count = len(group)
            
            # For cold_start_flag in EntityProfile, we use MIN_PROFILE_EVENTS.
            # Using graduation_threshold from our config.
            is_cold_start = event_count < self.config.graduation_threshold

            profile = EntityProfile(
                entity_id=str(entity_id),
                entity_type=etype,
                baseline_vector=mean_vec,
                baseline_std=std_vec,
                sequence_history=[],  # Not managed by BPM directly, empty for now
                most_frequent_country="US",  # Placeholder, as it requires raw event access
                known_mac_addresses=[],
                known_os_profiles=[],
                known_protocols=[],
                resource_access_counts={},
                command_frequency={},
                event_count=event_count,
                cold_start_flag=is_cold_start,
                last_updated=now_str,
                profile_version=1
            )
            self.store.upsert_profile(profile)

            # Collect all vectors for population prior
            for vec in vectors:
                all_vectors.append(EntityFeatureVector(root=vec.tolist()))
                all_types.append(etype)

        # Persist all newly built profiles
        self.store.save()

        # Build and save population prior
        self.prior.fit(all_vectors, all_types)
        self.prior.save()


    def score(
        self,
        entity_id: str,
        event_id: str,
        entity_type: EntityType,
        feature_vector: EntityFeatureVector
    ) -> ExtendedProfilingOutput:
        """
        Score a new feature vector against the entity's profile or population prior.
        """
        profile = self.store.get_profile(entity_id)
        
        # 1. Determine cold-start status and graduation
        if profile is None:
            profile_age = 0
            is_cold = True
        else:
            profile_age = profile.event_count
            is_cold = profile_age < self.config.graduation_threshold

        # 2. Select baseline statistics (Personalized vs Population)
        if is_cold:
            entity_status = EntityStatus.COLD_START
            try:
                mean_vec, std_vec = self.prior.get_prior(entity_type)
            except KeyError:
                # Fallback if prior not fitted for this type (e.g. edge case)
                mean_vec = [0.0] * len(FEATURE_NAMES)
                std_vec = [1.0] * len(FEATURE_NAMES)
            confidence = 0.6  # Lower confidence for population prior
        else:
            entity_status = EntityStatus.WARM
            mean_vec = profile.baseline_vector
            std_vec = profile.baseline_std
            confidence = 1.0  # High confidence for personalized profile

        # 3. Compute per-feature Z-scores
        raw_vec = feature_vector.root
        per_feature_deviations: Dict[str, float] = {}
        
        for i, fname in enumerate(FEATURE_NAMES):
            val = raw_vec[i]
            mean = mean_vec[i]
            std = max(std_vec[i], self.config.variance_epsilon)
            
            # Signed Z-score (positive = above normal)
            z_score = (val - mean) / std
            per_feature_deviations[fname] = z_score

        # 4. Compute overall deviation score
        max_abs_z = max(abs(z) for z in per_feature_deviations.values())
        deviation_score = min(max_abs_z / self.config.z_score_cap, 1.0)

        # 5. Determine top contributing features
        sorted_features = sorted(
            per_feature_deviations.keys(),
            key=lambda k: abs(per_feature_deviations[k]),
            reverse=True
        )
        top_contributing_features = sorted_features[:5]

        # 6. Construct output
        return ExtendedProfilingOutput(
            entity_id=entity_id,
            event_id=event_id,
            model_id="bpm",
            anomaly_score=deviation_score,
            confidence=confidence,
            cold_start_flag=is_cold,
            top_contributing_features=top_contributing_features,
            deviation_score=deviation_score,
            per_feature_deviations=per_feature_deviations,
            entity_status=entity_status,
            is_cold_start=is_cold,
            profile_age=profile_age
        )
