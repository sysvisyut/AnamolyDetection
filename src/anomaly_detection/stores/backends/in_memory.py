"""
In-memory backend for the Alert Store.
Useful for fast unit testing.
"""

from typing import List, Optional, Tuple, Dict

from anomaly_detection.common.models.alerts import Alert, AlertSummary
from anomaly_detection.common.models.entities import EntityHistoryEntry
from anomaly_detection.stores.alert_store import AbstractAlertStore


class InMemoryAlertStore(AbstractAlertStore):
    """
    In-memory implementation of AbstractAlertStore.
    """

    def __init__(self):
        # Maps alert_id to Alert
        self._alerts: Dict[str, Alert] = {}
        # Stores historical events to simulate Alert & Result store
        self._history: List[EntityHistoryEntry] = []

    def save_alert(self, alert: Alert) -> None:
        self._alerts[alert.alert_id] = alert

    def save_history_entry(self, entry: EntityHistoryEntry, entity_id: str) -> None:
        self._history.append((entity_id, entry))
        # Sort history by timestamp DESC
        self._history.sort(key=lambda x: x[1].timestamp, reverse=True)

        # Also push to history. We don't do this automatically anymore, caller does it.

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        return self._alerts.get(alert_id)

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
        
        filtered_alerts = list(self._alerts.values())

        if risk_tier:
            filtered_alerts = [a for a in filtered_alerts if a.risk.risk_tier in risk_tier]
        
        if attack_class:
            filtered_alerts = [a for a in filtered_alerts if a.attack_class.value in attack_class]
            
        if entity_id:
            filtered_alerts = [a for a in filtered_alerts if a.entity_id == entity_id]
            
        if since:
            filtered_alerts = [a for a in filtered_alerts if a.timestamp >= since]
            
        if until:
            filtered_alerts = [a for a in filtered_alerts if a.timestamp <= until]

        # Sort: Primary Sort: risk_score DESCENDING. Secondary Sort: timestamp DESCENDING.
        filtered_alerts.sort(key=lambda a: (a.risk.risk_score, a.timestamp), reverse=True)
        
        total_count = len(filtered_alerts)
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_alerts = filtered_alerts[start_idx:end_idx]

        summaries = []
        for a in paginated_alerts:
            summaries.append(AlertSummary(
                alert_id=a.alert_id,
                entity_id=a.entity_id,
                timestamp=a.timestamp,
                risk_score=a.risk.risk_score,
                risk_tier=a.risk.risk_tier,
                attack_class=a.attack_class,
                classification_confidence=a.classification_confidence,
                cold_start_flag=a.cold_start_flag,
                human_readable_explanation=a.explanation.human_readable_explanation[:150]
            ))

        return summaries, total_count

    def update_feedback(self, alert_id: str, decision: str, notes: str) -> bool:
        alert = self._alerts.get(alert_id)
        if not alert:
            return False
        alert.analyst_decision = decision
        alert.analyst_notes = notes
        return True

    def get_entity_history(self, entity_id: str, limit: int = 50) -> List[EntityHistoryEntry]:
        filtered = [entry for eid, entry in self._history if eid == entity_id]
        return filtered[:limit]


