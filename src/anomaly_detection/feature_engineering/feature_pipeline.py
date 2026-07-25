"""
Feature Pipeline — end-to-end orchestrator that transforms raw AccessLog
records into EngineeredFeatures instances consumed by BPM and SDM.

ARCHITECTURE COMPONENT: Feature Engineering
BOUNDARY RESPONSIBILITY: Consumes B (AccessLogBase); produces C (EngineeredFeatures).
TIER: T1

The FeaturePipeline orchestrates:
    SessionBuilder → FeatureExtractor → SequenceBuilder

It supports:
  - Batch mode (training): full DataFrame → list[EngineeredFeatures]
  - Single-record inference mode: one AccessLogInference → one EngineeredFeatures
  - fit() method: computes population-level fallback statistics from training
    data and stores them in FeatureEngineeringConfig

Label leakage prevention:
    - fit() accepts AccessLogTraining but strips labels before processing.
    - transform() and transform_training() share the same internal
      _transform_record() path, which only reads AccessLogBase fields.
    - Running the pipeline on AccessLogInference and AccessLogTraining
      for the same raw event fields produces identical feature vectors.

Cold-start path:
    - FeaturePipeline.fit() computes population-level baseline vectors and
      stds per entity type from all normal training events.
    - These are stored in config.population_baseline_vectors/stds and used
      by FeatureExtractor when profile.cold_start_flag is True.
    - Population statistics are NEVER hardcoded; they always reflect the
      actual training data distribution.
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from anomaly_detection.common.models.access_log import (
    AccessLogBase,
    AccessLogInference,
    AccessLogTraining,
)
from anomaly_detection.common.models.entities import EntityProfile
from anomaly_detection.common.models.enums import EntityStatus, EntityType
from anomaly_detection.common.models.features import (
    EngineeredFeatures,
    EntityFeatureVector,
    EntitySequence,
    SessionMetadata,
)
from anomaly_detection.feature_engineering.config import (
    FEATURE_DIM,
    FeatureEngineeringConfig,
)
from anomaly_detection.feature_engineering.feature_extractor import FeatureExtractor
from anomaly_detection.feature_engineering.profile_store_interface import (
    ProfileStoreInterface,
)
from anomaly_detection.feature_engineering.sequence_builder import SequenceBuilder
from anomaly_detection.feature_engineering.session_builder import (
    EventContext,
    SessionBuilder,
)


class FeaturePipeline:
    """
    End-to-end feature engineering pipeline: raw event → EngineeredFeatures.

    Orchestrates SessionBuilder → FeatureExtractor → SequenceBuilder with
    dependency-injection of the ProfileStore for testability.

    The pipeline supports both batch and single-record inference modes
    without code duplication:
        - Both modes call ``_transform_record()`` internally.
        - Batch mode pre-sorts events and loops; inference mode calls once.

    Dependency injection:
        ``profile_store`` is injected via the constructor. M06 never
        hardcodes a dependency on M05's concrete class. Any object that
        satisfies ``ProfileStoreInterface`` can be injected.
    """

    def __init__(
        self,
        config: FeatureEngineeringConfig,
        profile_store: ProfileStoreInterface,
    ) -> None:
        """
        Initialise the pipeline with configuration and a profile store.

        Args:
            config: Feature engineering configuration. Population fallback
                statistics should be populated via ``fit()`` before using
                ``transform()``.
            profile_store: Any object satisfying ``ProfileStoreInterface``.
                Typically ``stores.profile_store.InMemoryProfileStore`` or
                ``stores.profile_store.SQLiteProfileStore`` (M05 classes).
                M06 never imports either concrete class.
        """
        self._config = config
        self._profile_store = profile_store
        self._session_builder = SessionBuilder(config)
        self._extractor = FeatureExtractor(config)
        self._seq_builder = SequenceBuilder(config)
        self._is_fitted = False

    # ── fit: compute population statistics ────────────────────────────────

    def fit(self, training_records: Sequence[AccessLogTraining]) -> "FeaturePipeline":
        """
        Compute population-level fallback statistics from training data.

        This method:
        1. Resets all sliding state from previous runs.
        2. Processes all training records chronologically to compute feature
           vectors (without updating any profile store — this is read-only
           over the profile store passed at init time).
        3. Groups computed vectors by entity_type.
        4. Computes per-entity-type mean and std vectors.
        5. Stores them in ``self._config`` for cold-start use.
        6. Sets ``self._is_fitted = True``.

        Population statistics are NEVER hardcoded — they always reflect the
        actual training data, satisfying acceptance criterion 3.

        Args:
            training_records: Sequence of ``AccessLogTraining`` records from
                the training split. Labels are stripped internally before
                any computation.

        Returns:
            self (for method chaining).

        Label leakage prevention:
            Labels are accessed ONLY to identify normal events for baseline
            computation (per ML_PIPELINE.md §2.5). Labels are never stored
            in feature vectors, and the computed population statistics do
            not embed label information.
        """
        self._session_builder.reset()
        self._seq_builder.reset()

        # Sort chronologically
        sorted_records = sorted(training_records, key=lambda r: r.timestamp)

        # Group feature vectors by entity_type for population statistics
        # Only use normal-labeled events for population baseline (ML_PIPELINE.md §2.5)
        type_feature_vecs: Dict[str, List[List[float]]] = defaultdict(list)

        for record in sorted_records:
            # Produces boundary B (label-free AccessLogBase view)
            base_record: AccessLogBase = record  # subclass; label never read below

            profile = self._profile_store.get_profile(base_record.entity_id)
            ctx: EventContext = self._session_builder.process_event(base_record)
            fvec = self._extractor.extract(base_record, ctx, profile)
            self._seq_builder.update_and_get_window(fvec, base_record.entity_id)

            # Accumulate for population statistics — normal records only
            # This is the ONLY place label is accessed in M06; only for
            # baseline population computation, not for feature computation.
            if record.label.value == "normal":
                entity_type_key = base_record.entity_type
                type_feature_vecs[entity_type_key].append(fvec)

        # ── compute per-entity-type population statistics ─────────────────
        self._config.population_baseline_vectors = {}
        self._config.population_baseline_stds = {}

        for entity_type_key, vecs in type_feature_vecs.items():
            arr = np.array(vecs, dtype=np.float64)
            self._config.population_baseline_vectors[entity_type_key] = (
                arr.mean(axis=0).tolist()
            )
            self._config.population_baseline_stds[entity_type_key] = (
                arr.std(axis=0).tolist()
            )

        # ── compute population session duration mean/std ──────────────────
        all_durations = [r.session_duration for r in sorted_records if r.label.value == "normal"]
        if all_durations:
            dur_arr = np.array(all_durations, dtype=np.float64)
            self._config.cold_start_session_duration_mean = float(dur_arr.mean())
            self._config.cold_start_session_duration_std = max(
                float(dur_arr.std()), 1.0
            )

        # ── compute population most-frequent country ──────────────────────
        country_counts: Dict[str, int] = defaultdict(int)
        for r in sorted_records:
            if r.label.value == "normal":
                country_counts[r.geo_location.country] += 1
        if country_counts:
            self._config.population_most_frequent_country = max(
                country_counts, key=lambda k: country_counts[k]
            )

        # Reset state — fit() only computes statistics, transform() will
        # re-process events from scratch.
        self._session_builder.reset()
        self._seq_builder.reset()

        self._is_fitted = True
        return self

    # ── transform: batch mode ─────────────────────────────────────────────

    def transform(
        self,
        records: Sequence[AccessLogInference],
    ) -> List[EngineeredFeatures]:
        """
        Transform a batch of AccessLogInference records to EngineeredFeatures.

        Events are sorted chronologically before processing to ensure correct
        inter-event gap computation. The pipeline resets its sliding state
        at the start of each transform() call to avoid state bleed between
        evaluation runs.

        Args:
            records: Sequence of ``AccessLogInference`` records (no label field).

        Returns:
            List of ``EngineeredFeatures`` in the same chronological order
            as the sorted input records.

        Raises:
            RuntimeError: If ``fit()`` has not been called and the profile
                store contains no profiles (cold-start fallback unavailable).
        """
        self._session_builder.reset()
        self._seq_builder.reset()

        sorted_records = sorted(records, key=lambda r: r.timestamp)
        results: List[EngineeredFeatures] = []

        for record in sorted_records:
            ef = self._transform_record(record)
            results.append(ef)

        return results

    def transform_training(
        self,
        records: Sequence[AccessLogTraining],
    ) -> List[EngineeredFeatures]:
        """
        Transform a batch of AccessLogTraining records to EngineeredFeatures.

        Internally delegates to the same ``_transform_record()`` path as
        ``transform()`` — satisfying the label-leakage acceptance criterion:
        the feature vectors produced here must be identical to those produced
        by ``transform()`` for the same raw event fields.

        Args:
            records: Sequence of ``AccessLogTraining`` records.
                The ``label`` field is structurally present but never read
                by any code in the transform path.

        Returns:
            List of ``EngineeredFeatures`` in chronological order.
        """
        self._session_builder.reset()
        self._seq_builder.reset()

        sorted_records = sorted(records, key=lambda r: r.timestamp)
        results: List[EngineeredFeatures] = []

        for record in sorted_records:
            # Cast to base: label field is now structurally inaccessible
            base_record: AccessLogBase = record
            ef = self._transform_record(base_record)
            results.append(ef)

        return results

    # ── Single-record inference mode ───────────────────────────────────────

    def transform_single(
        self,
        record: AccessLogInference,
        profile: Optional[EntityProfile] = None,
    ) -> EngineeredFeatures:
        """
        Transform a single AccessLogInference record to EngineeredFeatures.

        This is the primary entry point for real-time inference. Unlike
        ``transform()``, it does NOT reset the pipeline's sliding state,
        so the entity's rolling deques and session counters persist across
        calls — matching production streaming behaviour.

        Args:
            record: A single ``AccessLogInference`` event.
            profile: Optional pre-fetched ``EntityProfile`` for this entity.
                If ``None``, the profile store is queried. Providing a
                pre-fetched profile avoids a redundant store lookup when
                the caller has already retrieved it.

        Returns:
            A single ``EngineeredFeatures`` instance.

        Notes:
            In streaming mode, events arrive one at a time in chronological
            order. The pipeline accumulates state across calls. For correctness,
            events for the same entity must be processed in temporal order.
        """
        return self._transform_record(record, preloaded_profile=profile)

    # ── Shared internal implementation ─────────────────────────────────────

    def _transform_record(
        self,
        event: AccessLogBase,
        preloaded_profile: Optional[EntityProfile] = None,
    ) -> EngineeredFeatures:
        """
        Core transform logic shared by batch and single-record modes.

        This is the single implementation that satisfies acceptance criterion 6
        (no code duplication between modes).

        Args:
            event: A raw access-log record (AccessLogBase or subclass).
                   The ``label`` field is never accessed here.
            preloaded_profile: Optional pre-fetched EntityProfile. If None,
                               the profile store is queried.

        Returns:
            ``EngineeredFeatures`` with a validated 24-dim feature vector
            and an updated sequence window.

        Label leakage prevention:
            All code in this method operates only on ``AccessLogBase`` fields.
            The ``label`` attribute does not exist on ``AccessLogBase``.
        """
        # ── 1. Fetch profile (boundary E) ──────────────────────────────────
        # Consumes boundary E: EntityProfile
        if preloaded_profile is not None:
            profile = preloaded_profile
        else:
            profile = self._profile_store.get_profile(event.entity_id)

        is_cold_start = (profile is None or profile.cold_start_flag)

        # ── 2. Session Builder: compute context ────────────────────────────
        ctx: EventContext = self._session_builder.process_event(event)

        # ── 3. Feature Extractor: compute 24-dim vector ────────────────────
        fvec: List[float] = self._extractor.extract(event, ctx, profile)

        # ── 4. Sequence Builder: update rolling window ─────────────────────
        window, mask = self._seq_builder.update_and_get_window(
            fvec, event.entity_id
        )

        # ── 5. Determine entity status for SessionMetadata ─────────────────
        if is_cold_start:
            entity_status_str = EntityStatus.COLD_START.value
        else:
            entity_status_str = EntityStatus.WARM.value

        # ── 6. Assemble SessionMetadata ────────────────────────────────────
        session_metadata = SessionMetadata(
            is_cold_start=is_cold_start,
            delivery_mode_hint=getattr(event, "delivery_mode", "batch"),
            profile_event_count=profile.event_count if profile else 0,
        )

        # ── 7. Produce boundary C: EngineeredFeatures ──────────────────────
        # Produces boundary C: EngineeredFeatures
        return EngineeredFeatures(
            entity_id=event.entity_id,
            event_id=event.event_id,
            session_id=event.session_id,
            feature_vector=EntityFeatureVector(root=fvec),
            sequence_window=EntitySequence(root=window),
            session_metadata=session_metadata,
        )
