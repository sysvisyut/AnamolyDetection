"""
Tests for Alert Store backends (M05).
"""

import pytest
import os
import uuid
import tempfile

from anomaly_detection.common.models.alerts import Alert, AlertSummary, Explanation, RiskScore, FeatureAttribution
from anomaly_detection.common.models.enums import AnomalyCategory
from anomaly_detection.common.models.entities import EntityHistoryEntry
from anomaly_detection.stores.backends.in_memory import InMemoryAlertStore
from anomaly_detection.stores.backends.sqlite import SQLiteAlertStore


def create_dummy_alert(risk_score: int, timestamp: str, attack_class: AnomalyCategory, alert_id: str = None) -> Alert:
    if not alert_id:
        alert_id = str(uuid.uuid4())
        
    return Alert(
        alert_id=alert_id,
        entity_id="user_123",
        event_id="evt_123",
        session_id="sess_123",
        timestamp=timestamp,
        detected_at="2026-07-26T10:00:00Z",
        risk=RiskScore(risk_score=risk_score, risk_tier="high" if risk_score >= 50 else "low"),
        explanation=Explanation(
            human_readable_explanation="Dummy explanation",
            feature_attributions=[
                FeatureAttribution(
                    feature_name="test_feature",
                    feature_value=0.9,
                    attribution_score=0.8,
                    direction="toward_anomaly",
                    source_model="bpm",
                    human_label="Test Feature"
                )
            ]
        ),
        attack_class=attack_class,
        classification_confidence=0.95,
        fused_score=0.8,
        bpm_score=0.7,
        sdm_score=0.9,
        cold_start_flag=False,
        raw_event_snapshot={"resource_accessed": "file.txt", "auth_outcome": "success"},
        analyst_decision=None,
        analyst_notes=None
    )


@pytest.fixture(params=["memory", "sqlite"])
def store(request):
    if request.param == "memory":
        yield InMemoryAlertStore()
    else:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        s = SQLiteAlertStore(db_path=path)
        yield s
        os.remove(path)


def test_save_and_get_alert(store):
    alert = create_dummy_alert(85, "2026-07-26T09:00:00Z", AnomalyCategory.BRUTE_FORCE)
    store.save_alert(alert)
    
    retrieved = store.get_alert(alert.alert_id)
    assert retrieved is not None
    assert retrieved.alert_id == alert.alert_id
    assert retrieved.risk.risk_score == 85
    assert retrieved.explanation.feature_attributions[0].feature_name == "test_feature"


def test_get_alerts_pagination_and_sorting(store):
    # Insert 3 alerts with different scores/times
    alert1 = create_dummy_alert(60, "2026-07-26T08:00:00Z", AnomalyCategory.BRUTE_FORCE, "a1")
    alert2 = create_dummy_alert(90, "2026-07-26T09:00:00Z", AnomalyCategory.IMPOSSIBLE_TRAVEL, "a2")
    alert3 = create_dummy_alert(60, "2026-07-26T10:00:00Z", AnomalyCategory.BRUTE_FORCE, "a3")
    
    store.save_alert(alert1)
    store.save_alert(alert2)
    store.save_alert(alert3)
    
    summaries, total = store.get_alerts(page=1, page_size=2)
    
    assert total == 3
    assert len(summaries) == 2
    
    # Sort order should be risk_score DESC, then timestamp DESC
    # Expected order: a2 (90), a3 (60, 10:00), a1 (60, 08:00)
    assert summaries[0].alert_id == "a2"
    assert summaries[1].alert_id == "a3"
    
    # Page 2
    summaries_p2, _ = store.get_alerts(page=2, page_size=2)
    assert len(summaries_p2) == 1
    assert summaries_p2[0].alert_id == "a1"


def test_get_alerts_filtering(store):
    store.save_alert(create_dummy_alert(40, "2026-07-25T00:00:00Z", AnomalyCategory.NORMAL, "a1")) # low tier
    store.save_alert(create_dummy_alert(85, "2026-07-26T00:00:00Z", AnomalyCategory.BRUTE_FORCE, "a2")) # high tier
    store.save_alert(create_dummy_alert(95, "2026-07-27T00:00:00Z", AnomalyCategory.IMPOSSIBLE_TRAVEL, "a3")) # high tier
    
    # Filter by risk tier
    summaries, total = store.get_alerts(risk_tier=["high"])
    assert total == 2
    assert all(s.risk_tier == "high" for s in summaries)
    
    # Filter by attack class
    summaries, total = store.get_alerts(attack_class=["impossible_travel"])
    assert total == 1
    assert summaries[0].alert_id == "a3"
    
    # Filter by since
    summaries, total = store.get_alerts(since="2026-07-26T00:00:00Z")
    assert total == 2
    
    # Filter by until
    summaries, total = store.get_alerts(until="2026-07-26T00:00:00Z")
    assert total == 2 # 25th and 26th


def test_update_feedback(store):
    alert = create_dummy_alert(85, "2026-07-26T09:00:00Z", AnomalyCategory.BRUTE_FORCE)
    store.save_alert(alert)
    
    # Update feedback
    success = store.update_feedback(alert.alert_id, "true_positive", "Investigated and confirmed.")
    assert success is True
    
    # Retrieve and verify
    retrieved = store.get_alert(alert.alert_id)
    assert retrieved.analyst_decision == "true_positive"
    assert retrieved.analyst_notes == "Investigated and confirmed."
    
    # Update non-existent
    success = store.update_feedback("does-not-exist", "false_positive", "")
    assert success is False


def test_entity_history(store):
    entry1 = EntityHistoryEntry(
        event_id="evt1",
        timestamp="2026-07-26T08:00:00Z",
        resource_accessed="file1.txt",
        auth_outcome="success",
        risk_score=None,
        attack_class=AnomalyCategory.NORMAL,
        has_alert=False
    )
    entry2 = EntityHistoryEntry(
        event_id="evt2",
        timestamp="2026-07-26T09:00:00Z",
        resource_accessed="file2.txt",
        auth_outcome="success",
        risk_score=85,
        attack_class=AnomalyCategory.BRUTE_FORCE,
        has_alert=True
    )
    
    store.save_history_entry(entry1, "user_1")
    store.save_history_entry(entry2, "user_1")
    
    # Should sort descending by timestamp
    history = store.get_entity_history("user_1")
    assert len(history) == 2
    assert history[0].event_id == "evt2"
    assert history[1].event_id == "evt1"
    
    assert store.get_entity_history("user_2") == []


def test_get_alert_store_factory(tmp_path):
    from anomaly_detection.stores.alert_store import get_alert_store
    import yaml
    
    # Test sqlite backend
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "stores": {
            "alert_store": {
                "backend": "sqlite",
                "sqlite_path": str(tmp_path / "test.db")
            }
        }
    }))
    
    store = get_alert_store(str(config_file))
    assert isinstance(store, SQLiteAlertStore)
    
    # Test in_memory backend
    config_file.write_text(yaml.dump({
        "stores": {
            "alert_store": {
                "backend": "in_memory"
            }
        }
    }))
    
    store = get_alert_store(str(config_file))
    assert isinstance(store, InMemoryAlertStore)
