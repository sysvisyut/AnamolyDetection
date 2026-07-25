"""
Sequence Builder — constructs fixed-length, left-padded sliding sequence
windows of feature vectors for SDM input.

ARCHITECTURE COMPONENT: Feature Engineering
BOUNDARY RESPONSIBILITY: Contributes to producing C (EngineeredFeatures),
                         specifically the sequence_window component.
TIER: T1

The SequenceBuilder takes an ordered list of feature vectors per entity
and constructs sliding windows of length W (default 20), left-padded with
zero vectors for entities with fewer than W historical events.

Both batch mode (full entity history → list of windows) and single-record
inference mode (one new vector → one updated window) share the same
``_build_window()`` helper, satisfying the no-code-duplication acceptance
criterion.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterator, List, Tuple

from anomaly_detection.feature_engineering.config import (
    DEFAULT_SEQUENCE_WINDOW,
    DEFAULT_STRIDE,
    FEATURE_DIM,
    FeatureEngineeringConfig,
)


class SequenceBuilder:
    """
    Constructs fixed-length, zero-padded sliding sequence windows for SDM input.

    The SDM requires tensors of shape ``(1, W, 24)`` (DATA_SCHEMA.md §3.3).
    This class produces ``(W, 24)`` lists-of-lists; the pipeline assembles
    them into tensors downstream.

    Maintains a per-entity deque of the last W feature vectors for
    single-record inference mode (``update_and_get_window()``).

    Usage — batch mode (training):
        windows = list(builder.sliding_windows(fvecs, entity_id="usr_001"))

    Usage — single-record inference mode:
        window, mask = builder.update_and_get_window(fvec, entity_id="usr_001")
    """

    def __init__(self, config: FeatureEngineeringConfig) -> None:
        """
        Initialise the SequenceBuilder with the pipeline configuration.

        Args:
            config: Fully constructed ``FeatureEngineeringConfig``.
        """
        self._window_size = config.sequence_window_size
        self._stride = config.stride
        self._feature_dim = FEATURE_DIM

        # Per-entity rolling deque for inference mode (keyed by entity_id)
        self._entity_windows: Dict[str, Deque[List[float]]] = {}

    def reset(self) -> None:
        """Clear all accumulated per-entity deques."""
        self._entity_windows.clear()

    # ── Shared helper ──────────────────────────────────────────────────────

    @staticmethod
    def _build_window(
        history: List[List[float]],
        window_size: int,
    ) -> Tuple[List[List[float]], List[bool]]:
        """
        Build a single fixed-length window from a history of feature vectors.

        This is the **shared helper used by both batch and single-record modes**
        to avoid code duplication (acceptance criterion 6).

        Args:
            history: Ordered list of feature vectors (oldest first).
                     May contain fewer than ``window_size`` elements.
            window_size: Target window length W (default 20).

        Returns:
            A 2-tuple:
                - ``window``: list of ``window_size`` feature vectors,
                  left-padded with zero vectors if ``len(history) < window_size``.
                - ``mask``: list of ``window_size`` bools, where
                  ``True`` = real event position, ``False`` = zero-padding.

        The window ordering is: index 0 = oldest event, index W-1 = most recent.
        Zero-padding occupies the leftmost positions (index 0 onward).
        """
        feature_dim = len(history[0]) if history else FEATURE_DIM
        zero_vec: List[float] = [0.0] * feature_dim

        n_real = min(len(history), window_size)
        n_padding = window_size - n_real

        # Take the most recent n_real vectors
        recent = history[-n_real:] if n_real > 0 else []

        window = [zero_vec] * n_padding + recent
        mask = [False] * n_padding + [True] * n_real

        return (window, mask)

    # ── Batch mode ─────────────────────────────────────────────────────────

    def sliding_windows(
        self,
        feature_vectors: List[List[float]],
        entity_id: str = "",
    ) -> Iterator[Tuple[List[List[float]], List[bool]]]:
        """
        Generate all sliding sequence windows for a chronological list of vectors.

        Used in training mode to produce all training examples for the SDM.

        Args:
            feature_vectors: Chronologically ordered list of 24-dim feature
                vectors for a single entity.
            entity_id: Optional entity identifier (used for logging only).

        Yields:
            2-tuples of ``(window, mask)`` where:
                - ``window``: shape ``(W, 24)`` list-of-lists
                - ``mask``: shape ``(W,)`` bool list

        Notes:
            - The first window uses only the first event (heavily padded).
            - Subsequent windows step by ``self._stride`` events.
            - For entities with zero events, yields nothing.

        Derives from:
            DATA_SCHEMA.md §3.3 (sequence_window specification).
        """
        if not feature_vectors:
            return

        n = len(feature_vectors)
        # Generate one window ending at each valid position stepping by stride
        for end_idx in range(0, n, self._stride):
            history = feature_vectors[: end_idx + 1]
            yield self._build_window(history, self._window_size)

    def build_batch_windows(
        self,
        feature_vectors: List[List[float]],
    ) -> Tuple[List[List[List[float]]], List[List[bool]]]:
        """
        Build all sliding windows for a single entity and return as parallel lists.

        Convenience wrapper around ``sliding_windows()`` that collects
        all outputs into two lists.

        Args:
            feature_vectors: Chronologically ordered list of 24-dim feature vectors.

        Returns:
            A 2-tuple:
                - ``windows``: list of ``(W, 24)`` windows
                - ``masks``: list of ``(W,)`` bool masks

        Derives from:
            ML_PIPELINE.md §3.7 (SDM training data preparation).
        """
        windows: List[List[List[float]]] = []
        masks: List[List[bool]] = []
        for window, mask in self.sliding_windows(feature_vectors):
            windows.append(window)
            masks.append(mask)
        return (windows, masks)

    # ── Single-record inference mode ───────────────────────────────────────

    def update_and_get_window(
        self,
        feature_vector: List[float],
        entity_id: str,
    ) -> Tuple[List[List[float]], List[bool]]:
        """
        Append a new feature vector to the entity's rolling deque and return
        the current window.

        This is the primary method for single-record inference mode. It updates
        the entity's history with the new vector, then calls ``_build_window()``
        — the same helper used by batch mode.

        Args:
            feature_vector: The 24-dim feature vector for the current event.
            entity_id: Entity identifier string.

        Returns:
            A 2-tuple ``(window, mask)`` ready for SDM input:
                - ``window``: shape ``(W, 24)`` list-of-lists
                - ``mask``: shape ``(W,)`` bool list

        Raises:
            ValueError: If ``feature_vector`` has incorrect length.
        """
        if len(feature_vector) != FEATURE_DIM:
            raise ValueError(
                f"Feature vector length {len(feature_vector)} != {FEATURE_DIM}"
            )

        if entity_id not in self._entity_windows:
            self._entity_windows[entity_id] = deque(maxlen=self._window_size)

        self._entity_windows[entity_id].append(feature_vector)

        # Use the shared helper (same code path as batch mode)
        history = list(self._entity_windows[entity_id])
        return self._build_window(history, self._window_size)

    def get_current_window(
        self,
        entity_id: str,
    ) -> Tuple[List[List[float]], List[bool]]:
        """
        Retrieve the current sequence window for an entity without adding a new vector.

        Useful for building the initial window from profile.sequence_history
        before any new events arrive.

        Args:
            entity_id: Entity identifier string.

        Returns:
            A 2-tuple ``(window, mask)`` using whatever vectors are in the deque.
            Returns an all-padding window if the entity has no deque yet.
        """
        if entity_id not in self._entity_windows:
            # Empty history → fully padded window
            zero_window: List[List[float]] = [[0.0] * FEATURE_DIM] * self._window_size
            mask: List[bool] = [False] * self._window_size
            return (zero_window, mask)

        history = list(self._entity_windows[entity_id])
        return self._build_window(history, self._window_size)

    def seed_from_profile(
        self,
        entity_id: str,
        sequence_history: List[List[float]],
    ) -> None:
        """
        Pre-populate an entity's rolling deque from its EntityProfile.sequence_history.

        Called at pipeline startup to seed the inference-mode window with
        the entity's last W vectors from the Profile Store, avoiding a
        cold-restart of the sequence context.

        Args:
            entity_id: Entity identifier string.
            sequence_history: List of feature vectors from
                ``EntityProfile.sequence_history`` (up to W vectors, oldest first).
        """
        self._entity_windows[entity_id] = deque(
            sequence_history[-self._window_size :], maxlen=self._window_size
        )

    @staticmethod
    def window_to_flat(window: List[List[float]]) -> List[float]:
        """
        Flatten a ``(W, 24)`` window to a ``(W*24,)`` 1-D list.

        Utility method for implementations that need a flat representation
        rather than a 2-D list.

        Args:
            window: ``(W, 24)`` list-of-lists.

        Returns:
            Flattened list of floats of length ``W * 24``.
        """
        return [val for row in window for val in row]
