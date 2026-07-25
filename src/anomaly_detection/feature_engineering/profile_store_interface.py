"""
Abstract interface (Protocol) that the Feature Engineering Pipeline uses
to retrieve and update entity behavioral profiles.

ARCHITECTURE COMPONENT: Feature Engineering
BOUNDARY RESPONSIBILITY: Consumes Boundary E (EntityProfile) via abstract methods.
                         Does NOT implement Boundary E — that is M05 (stores/).
TIER: T1

Dependency direction: M06 (feature_engineering) DEFINES this interface.
                      M05 (stores/profile_store.py) IMPLEMENTS it.
                      M06 must NOT import from M05 — only from this file.

M06 defines the contract; M06 also depends on it via dependency injection
in FeaturePipeline and FeatureExtractor. This is the classic
Dependency Inversion Principle pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Protocol, runtime_checkable

from anomaly_detection.common.models.entities import EntityProfile


@runtime_checkable
class ProfileStoreInterface(Protocol):
    """
    Protocol (structural typing contract) that any ProfileStore must satisfy.

    M06 references this interface; M05 provides a concrete class
    (InMemoryProfileStore, SQLiteProfileStore) that satisfies it.
    Neither M05 nor any other module needs to explicitly inherit from this
    class — Python's structural typing checks compliance at runtime
    via isinstance(store, ProfileStoreInterface).

    Dependency direction:
        feature_engineering.profile_store_interface   ← defines
        stores.profile_store                          ← implements
        M06 never imports from stores/ (CODING_GUIDELINES §2.2)
    """

    def get_profile(self, entity_id: str) -> Optional[EntityProfile]:
        """
        Retrieve the current behavioral profile for a given entity.

        Args:
            entity_id: The unique entity identifier (e.g. ``usr_4d8e21bc``).

        Returns:
            The entity's ``EntityProfile`` if it exists, or ``None`` if the
            entity is brand-new (no events ever seen). The Feature Engineering
            layer treats ``None`` as a cold-start signal and uses the
            population-level fallback profile.

        Note:
            Implementations must be safe to call concurrently from multiple
            inference threads. Read-only — this method must not mutate state.
        """
        ...

    def get_profiles_batch(
        self, entity_ids: List[str]
    ) -> Dict[str, Optional[EntityProfile]]:
        """
        Retrieve profiles for multiple entities in a single call.

        Provides a batched read interface for training-mode pipeline
        execution where many entities are processed simultaneously.

        Args:
            entity_ids: List of entity identifiers to look up.

        Returns:
            Dictionary mapping each ``entity_id`` to its ``EntityProfile``
            (or ``None`` if not found).

        Note:
            Default implementation in AbstractProfileStore iterates
            ``get_profile``; concrete implementations may override with
            a batch SQL query or cache multi-get.
        """
        ...

    def upsert_profile(self, profile: EntityProfile) -> None:
        """
        Insert or update an entity's behavioral profile.

        Called by the profile updater (M11) after each event that passes
        the Gated EWMA threshold (fused_score < 0.4).

        Args:
            profile: The updated ``EntityProfile`` object.  The caller
                     is responsible for incrementing ``profile_version``
                     before passing it here.

        Note:
            Implementations must be idempotent — calling upsert with the
            same profile twice must not corrupt state. Write atomicity is
            expected (no partial updates visible to concurrent readers).
        """
        ...

    def list_entity_ids(self) -> List[str]:
        """
        Return all known entity IDs in the store.

        Used by ``FeaturePipeline.fit()`` to enumerate entities during
        training-time population statistics computation.

        Returns:
            Sorted list of entity identifier strings.
        """
        ...


class AbstractProfileStore(ABC):
    """
    Optional abstract base class for ProfileStore implementors.

    Provides a default ``get_profiles_batch`` implementation that
    iterates ``get_profile``. Concrete stores (SQLite, in-memory) may
    override this for performance.

    Inheriting from this class satisfies ``ProfileStoreInterface``
    automatically, but inheritance is not required — structural typing
    via the Protocol above is sufficient.
    """

    @abstractmethod
    def get_profile(self, entity_id: str) -> Optional[EntityProfile]:
        """
        Retrieve the current behavioral profile for a given entity.

        Args:
            entity_id: The unique entity identifier.

        Returns:
            The entity's ``EntityProfile`` or ``None`` if brand-new.
        """

    def get_profiles_batch(
        self, entity_ids: List[str]
    ) -> Dict[str, Optional[EntityProfile]]:
        """
        Retrieve profiles for multiple entities.

        Default implementation calls ``get_profile`` for each entity.
        Override in subclasses for batch-optimised lookups.

        Args:
            entity_ids: List of entity identifiers.

        Returns:
            Dictionary mapping entity_id to profile or None.
        """
        return {eid: self.get_profile(eid) for eid in entity_ids}

    @abstractmethod
    def upsert_profile(self, profile: EntityProfile) -> None:
        """
        Insert or update an entity's behavioral profile.

        Args:
            profile: The updated ``EntityProfile``.
        """

    @abstractmethod
    def list_entity_ids(self) -> List[str]:
        """
        Return all known entity IDs in the store.

        Returns:
            Sorted list of entity identifier strings.
        """
