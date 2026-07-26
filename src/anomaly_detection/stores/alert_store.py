"""
Alert Store abstract interface (M05).
Defines the contract for alert persistence and retrieval (Boundary J).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import yaml

from anomaly_detection.common.models.alerts import Alert, AlertSummary
from anomaly_detection.common.models.entities import EntityHistoryEntry


class AbstractAlertStore(ABC):
    """
    Interface for the Alert & Result Store.
    Manages persistence for alerts and entity historical events.
    """

    @abstractmethod
    def save_alert(self, alert: Alert) -> None:
        """
        Persist a new alert.
        """
        pass

    @abstractmethod
    def save_history_entry(self, entry: EntityHistoryEntry, entity_id: str) -> None:
        """
        Persist an event history entry (normal or alert).
        """
        pass

    @abstractmethod
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """
        Retrieve a single alert by its ID.
        """
        pass

    @abstractmethod
    def get_alerts(
        self,
        page: int = 1,
        page_size: int = 50,
        risk_tier: Optional[List[str]] = None,
        attack_class: Optional[List[str]] = None,
        entity_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None
    ) -> Tuple[List[AlertSummary], int]:
        """
        Retrieve a paginated, filtered, and ranked list of alerts.
        Returns:
            Tuple containing the list of AlertSummary objects for the requested page,
            and the total count of alerts matching the filters.
        """
        pass

    @abstractmethod
    def update_feedback(self, alert_id: str, decision: str, notes: str) -> bool:
        """
        Update analyst feedback on a specific alert.
        Returns True if the alert was found and updated, False otherwise.
        """
        pass

    @abstractmethod
    def get_entity_history(self, entity_id: str, limit: int = 50) -> List[EntityHistoryEntry]:
        """
        Retrieve chronological timeline of events for a specific entity.
        """
        pass


def get_alert_store(config_path: str = "config/default.yaml") -> AbstractAlertStore:
    """
    Factory function to retrieve the configured AlertStore backend.
    """
    import os
    from anomaly_detection.stores.backends.in_memory import InMemoryAlertStore
    from anomaly_detection.stores.backends.sqlite import SQLiteAlertStore
    
    # Load config
    backend_type = "in_memory"
    sqlite_path = "data/alerts.db"
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
            store_config = config.get("stores", {}).get("alert_store", {})
            backend_type = store_config.get("backend", "in_memory")
            sqlite_path = store_config.get("sqlite_path", "data/alerts.db")
            
    if backend_type == "sqlite":
        return SQLiteAlertStore(db_path=sqlite_path)
    elif backend_type == "in_memory":
        return InMemoryAlertStore()
    else:
        raise ValueError(f"Unknown alert_store backend: {backend_type}")

