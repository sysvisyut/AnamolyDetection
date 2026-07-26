#!/usr/bin/env python3
"""
Seed script to generate test data for the M14 Analyst Dashboard.
Creates fake alerts in the SQLite database to allow testing of the frontend.
"""

import os
import uuid
import random
from datetime import datetime, timedelta

from anomaly_detection.stores.backends.sqlite import SQLiteAlertStore
from anomaly_detection.common.models.alerts import Alert, RiskScore, Explanation, FeatureAttribution
from anomaly_detection.common.models.enums import AnomalyCategory

def generate_fake_alert(entity_id: str, minutes_ago: int, is_cold_start: bool = False) -> Alert:
    categories = list(AnomalyCategory)
    
    # Remove normal from the choices
    categories = [c for c in categories if c != AnomalyCategory.NORMAL]
    attack_class = random.choice(categories)
    
    score = random.randint(65, 99)
    if score >= 90:
        tier = "critical"
    elif score >= 80:
        tier = "high"
    elif score >= 70:
        tier = "medium"
    else:
        tier = "low"
        
    ts = (datetime.utcnow() - timedelta(minutes=minutes_ago)).isoformat() + "Z"
    
    features = [
        FeatureAttribution(
            feature_name="login_failures",
            feature_value=random.uniform(0.1, 1.0),
            attribution_score=random.uniform(1.0, 5.0),
            direction="toward_anomaly",
            source_model="bpm",
            human_label="High number of login failures"
        ),
        FeatureAttribution(
            feature_name="geo_velocity",
            feature_value=random.uniform(0.1, 1.0),
            attribution_score=random.uniform(0.5, 3.0),
            direction="toward_anomaly",
            source_model="sdm",
            human_label="Impossible travel speed detected"
        )
    ]
    
    explanation = f"Detected {attack_class.value.replace('_', ' ')} based on abnormal behavior."
    if is_cold_start:
        explanation = "[COLD START HEURISTIC] " + explanation
        
    alert = Alert(
        alert_id=str(uuid.uuid4()),
        entity_id=entity_id,
        event_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
        timestamp=ts,
        detected_at=ts,
        risk=RiskScore(risk_score=score, risk_tier=tier),
        explanation=Explanation(
            human_readable_explanation=explanation,
            feature_attributions=features
        ),
        attack_class=attack_class,
        classification_confidence=random.uniform(0.7, 0.99),
        fused_score=score / 100.0,
        bpm_score=random.uniform(0.5, 1.0),
        sdm_score=random.uniform(0.5, 1.0),
        cold_start_flag=is_cold_start,
        raw_event_snapshot={"foo": "bar", "resource_accessed": "/api/admin"},
        analyst_decision=None,
        analyst_notes=None
    )
    return alert

def main():
    db_path = "data/alerts.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    store = SQLiteAlertStore(db_path=db_path)
    
    print(f"Seeding database at {db_path}...")
    
    entities = [f"user_{i}" for i in range(1, 6)]
    
    count = 0
    for entity_id in entities:
        # Create 3-5 alerts per entity
        num_alerts = random.randint(3, 5)
        is_cold = random.random() < 0.3
        
        for i in range(num_alerts):
            alert = generate_fake_alert(
                entity_id=entity_id, 
                minutes_ago=random.randint(1, 1440),
                is_cold_start=is_cold
            )
            store.save_alert(alert)
            count += 1
            
            # Also insert history for testing EntityView
            with store._get_connection() as conn:
                conn.execute("""
                    INSERT INTO entity_history (entity_id, event_id, timestamp, resource_accessed, auth_outcome, risk_score, attack_class, has_alert)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity_id, alert.event_id, alert.timestamp, "/api/data", "success", 
                    alert.risk.risk_score, alert.attack_class.value, True
                ))
    
    print(f"Successfully seeded {count} alerts.")

if __name__ == "__main__":
    main()
