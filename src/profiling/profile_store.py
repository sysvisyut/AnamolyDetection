"""
Profile Store Implementation (M06).

Implements the ProfileStoreInterface defined by M05 Feature Engineering.
Provides persistence for EntityProfile objects.
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime, timezone

from anomaly_detection.feature_engineering.profile_store_interface import AbstractProfileStore
from anomaly_detection.common.models.entities import EntityProfile


class ProfileStore(AbstractProfileStore):
    """
    Concrete implementation of ProfileStoreInterface.
    Uses an in-memory dictionary backed by a JSON file on disk for persistence.
    """

    def __init__(self, persistence_path: str = "data/profiles/store.json") -> None:
        self.persistence_path = persistence_path
        self._profiles: Dict[str, EntityProfile] = {}
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.persistence_path)), exist_ok=True)

    def get_profile(self, entity_id: str) -> Optional[EntityProfile]:
        """
        Retrieve the current behavioral profile for a given entity.
        Returns None if the entity is brand-new.
        """
        return self._profiles.get(entity_id)

    def upsert_profile(self, profile: EntityProfile) -> None:
        """
        Insert or update an entity's behavioral profile.
        M11 is responsible for incrementing profile_version before calling this.
        """
        self._profiles[profile.entity_id] = profile

    def list_entity_ids(self) -> List[str]:
        """
        Return all known entity IDs in the store.
        """
        return sorted(self._profiles.keys())

    def save(self) -> None:
        """
        Persist all profiles to disk atomically.
        """
        temp_path = f"{self.persistence_path}.tmp"
        
        # Serialize Pydantic models to dicts
        data = {
            eid: profile.model_dump(mode="json")
            for eid, profile in self._profiles.items()
        }
        
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
            
        # Atomic rename to prevent corruption
        os.replace(temp_path, self.persistence_path)

    def load(self) -> None:
        """
        Load all profiles from disk into memory.
        """
        if os.path.exists(self.persistence_path):
            with open(self.persistence_path, "r") as f:
                data = json.load(f)
            
            self._profiles = {
                eid: EntityProfile.model_validate(profile_dict)
                for eid, profile_dict in data.items()
            }
