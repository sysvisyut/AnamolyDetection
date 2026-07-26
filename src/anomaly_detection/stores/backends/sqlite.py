"""
SQLite backend for the Alert Store (M05).
Handles schema initialization, persistence, and JSON serialization.
"""

import json
import sqlite3
import os
from typing import List, Optional, Tuple, Any

from anomaly_detection.common.models.alerts import Alert, AlertSummary
from anomaly_detection.common.models.entities import EntityHistoryEntry
from anomaly_detection.stores.alert_store import AbstractAlertStore


class SQLiteAlertStore(AbstractAlertStore):
    """
    SQLite implementation of AbstractAlertStore.
    Persists data in a local SQLite database.
    """

    def __init__(self, db_path: str = "data/alerts.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    entity_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    risk_score INTEGER NOT NULL,
                    risk_tier TEXT NOT NULL,
                    attack_class TEXT NOT NULL,
                    classification_confidence REAL NOT NULL,
                    fused_score REAL NOT NULL,
                    bpm_score REAL NOT NULL,
                    sdm_score REAL NOT NULL,
                    cold_start_flag BOOLEAN NOT NULL,
                    human_readable_explanation TEXT NOT NULL,
                    feature_attributions TEXT NOT NULL,
                    raw_event_snapshot TEXT NOT NULL,
                    analyst_decision TEXT,
                    analyst_notes TEXT
                )
            """)
            # Index for fast querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts (timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_risk ON alerts (risk_score DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_entity ON alerts (entity_id)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    resource_accessed TEXT NOT NULL,
                    auth_outcome TEXT NOT NULL,
                    risk_score INTEGER,
                    attack_class TEXT NOT NULL,
                    has_alert BOOLEAN NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_entity ON entity_history (entity_id, timestamp DESC)")

    def save_alert(self, alert: Alert) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alerts (
                    alert_id, entity_id, event_id, session_id, timestamp, detected_at,
                    risk_score, risk_tier, attack_class, classification_confidence,
                    fused_score, bpm_score, sdm_score, cold_start_flag,
                    human_readable_explanation, feature_attributions, raw_event_snapshot,
                    analyst_decision, analyst_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id,
                alert.entity_id,
                alert.event_id,
                alert.session_id,
                alert.timestamp,
                alert.detected_at,
                alert.risk.risk_score,
                alert.risk.risk_tier,
                alert.attack_class.value,
                alert.classification_confidence,
                alert.fused_score,
                alert.bpm_score,
                alert.sdm_score,
                alert.cold_start_flag,
                alert.explanation.human_readable_explanation,
                json.dumps([fa.model_dump() for fa in alert.explanation.feature_attributions]),
                json.dumps(alert.raw_event_snapshot),
                alert.analyst_decision,
                alert.analyst_notes
            ))

    def save_history_entry(self, entry: EntityHistoryEntry, entity_id: str) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO entity_history (
                    entity_id, event_id, timestamp, resource_accessed, auth_outcome,
                    risk_score, attack_class, has_alert
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entity_id,
                entry.event_id,
                entry.timestamp,
                entry.resource_accessed,
                entry.auth_outcome,
                entry.risk_score,
                entry.attack_class.value,
                entry.has_alert
            ))

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)).fetchone()
            if not row:
                return None
            
            # Reconstruct Alert object
            return Alert(
                alert_id=row['alert_id'],
                entity_id=row['entity_id'],
                event_id=row['event_id'],
                session_id=row['session_id'],
                timestamp=row['timestamp'],
                detected_at=row['detected_at'],
                risk={
                    "risk_score": row['risk_score'],
                    "risk_tier": row['risk_tier']
                },
                explanation={
                    "human_readable_explanation": row['human_readable_explanation'],
                    "feature_attributions": json.loads(row['feature_attributions'])
                },
                attack_class=row['attack_class'],
                classification_confidence=row['classification_confidence'],
                fused_score=row['fused_score'],
                bpm_score=row['bpm_score'],
                sdm_score=row['sdm_score'],
                cold_start_flag=bool(row['cold_start_flag']),
                raw_event_snapshot=json.loads(row['raw_event_snapshot']),
                analyst_decision=row['analyst_decision'],
                analyst_notes=row['analyst_notes']
            )

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
        
        query = "SELECT * FROM alerts WHERE 1=1"
        count_query = "SELECT COUNT(*) as total FROM alerts WHERE 1=1"
        params: List[Any] = []
        
        if risk_tier:
            placeholders = ",".join(["?"] * len(risk_tier))
            query += f" AND risk_tier IN ({placeholders})"
            count_query += f" AND risk_tier IN ({placeholders})"
            params.extend(risk_tier)
            
        if attack_class:
            placeholders = ",".join(["?"] * len(attack_class))
            query += f" AND attack_class IN ({placeholders})"
            count_query += f" AND attack_class IN ({placeholders})"
            params.extend(attack_class)
            
        if entity_id:
            query += " AND entity_id = ?"
            count_query += " AND entity_id = ?"
            params.append(entity_id)
            
        if since:
            query += " AND timestamp >= ?"
            count_query += " AND timestamp >= ?"
            params.append(since)
            
        if until:
            query += " AND timestamp <= ?"
            count_query += " AND timestamp <= ?"
            params.append(until)
            
        query += " ORDER BY risk_score DESC, timestamp DESC LIMIT ? OFFSET ?"
        
        with self._get_connection() as conn:
            total_count = conn.execute(count_query, params).fetchone()['total']
            
            # Add pagination params
            offset = (page - 1) * page_size
            paginated_params = params + [page_size, offset]
            
            rows = conn.execute(query, paginated_params).fetchall()
            
            summaries = []
            for row in rows:
                summaries.append(AlertSummary(
                    alert_id=row['alert_id'],
                    entity_id=row['entity_id'],
                    timestamp=row['timestamp'],
                    risk_score=row['risk_score'],
                    risk_tier=row['risk_tier'],
                    attack_class=row['attack_class'],
                    classification_confidence=row['classification_confidence'],
                    cold_start_flag=bool(row['cold_start_flag']),
                    human_readable_explanation=row['human_readable_explanation'][:150]
                ))
                
            return summaries, total_count

    def update_feedback(self, alert_id: str, decision: str, notes: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("""
                UPDATE alerts
                SET analyst_decision = ?, analyst_notes = ?
                WHERE alert_id = ?
            """, (decision, notes, alert_id))
            return cursor.rowcount > 0

    def get_entity_history(self, entity_id: str, limit: int = 50) -> List[EntityHistoryEntry]:
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM entity_history
                WHERE entity_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (entity_id, limit)).fetchall()
            
            return [
                EntityHistoryEntry(
                    event_id=row['event_id'],
                    timestamp=row['timestamp'],
                    resource_accessed=row['resource_accessed'],
                    auth_outcome=row['auth_outcome'],
                    risk_score=row['risk_score'],
                    attack_class=row['attack_class'],
                    has_alert=bool(row['has_alert'])
                )
                for row in rows
            ]
