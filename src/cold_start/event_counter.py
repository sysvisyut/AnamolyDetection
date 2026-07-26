"""
Persistent Event Counter for the Cold-Start Handler (M10).
"""

import json
import os
from typing import Dict, List, Any


class EventCounter:
    """
    Maintains a persistent count of events and accumulated feature vectors
    for entities currently in the cold-start phase.
    """

    def __init__(self, persistence_path: str) -> None:
        """
        Initialize the event counter.

        Args:
            persistence_path: File path where the counter state is persisted as JSON.
        """
        self.persistence_path = persistence_path
        # Schema: {entity_id: {"count": int, "vectors": List[List[float]]}}
        self._state: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Loads state from disk if the file exists."""
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, "r", encoding="utf-8") as f:
                    self._state = json.load(f)
            except (json.JSONDecodeError, IOError):
                # If file is corrupt or unreadable, start fresh
                self._state = {}
        else:
            self._state = {}

    def _save(self) -> None:
        """Saves state to disk atomically."""
        os.makedirs(os.path.dirname(os.path.abspath(self.persistence_path)), exist_ok=True)
        temp_path = f"{self.persistence_path}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._state, f)
            os.replace(temp_path, self.persistence_path)
        except IOError:
            # Swallow save errors gracefully in production, though in a real system
            # this would be logged.
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def increment(self, entity_id: str, vector: List[float]) -> None:
        """
        Increment the event count for an entity and store the feature vector.
        Saves the state atomically to disk.
        """
        if entity_id not in self._state:
            self._state[entity_id] = {"count": 0, "vectors": []}
        
        self._state[entity_id]["count"] += 1
        self._state[entity_id]["vectors"].append(vector)
        self._save()

    def get_count(self, entity_id: str) -> int:
        """Get the current event count for an entity."""
        if entity_id in self._state:
            return self._state[entity_id]["count"]
        return 0

    def get_vectors(self, entity_id: str) -> List[List[float]]:
        """Get the accumulated feature vectors for an entity."""
        if entity_id in self._state:
            return self._state[entity_id]["vectors"]
        return []

    def reset(self, entity_id: str) -> None:
        """
        Remove the entity from the event counter (used upon graduation).
        """
        if entity_id in self._state:
            del self._state[entity_id]
            self._save()
