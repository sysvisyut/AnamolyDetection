"""Tests for the M12 alert-construction boundary.

Tier: T1
Pytest mark: @pytest.mark.tier1
"""

from anomaly_detection.common.models.enums import AnomalyCategory

from src.orchestrator.alert_builder import MAX_INSIDER_DRIFT_RISK_SCORE, AlertBuilder


def test_risk_tier_assigns_documented_score_bands() -> None:
    """Alert risk tiers match the low, medium, high, and critical ranges."""
    assert AlertBuilder._risk_tier(24) == "low"
    assert AlertBuilder._risk_tier(25) == "medium"
    assert AlertBuilder._risk_tier(50) == "high"
    assert AlertBuilder._risk_tier(75) == "critical"


def test_risk_score_caps_insider_drift_at_medium_tier() -> None:
    """Insider Drift cannot reach high or critical analyst-routing tiers."""
    risk_score = AlertBuilder._risk_score(0.95, AnomalyCategory.INSIDER_DRIFT)

    assert risk_score == MAX_INSIDER_DRIFT_RISK_SCORE
    assert AlertBuilder._risk_tier(risk_score) == "medium"
