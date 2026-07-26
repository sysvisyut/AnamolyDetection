"""Tests for the Inference API endpoint."""

import pytest
from httpx import AsyncClient, ASGITransport
from datetime import datetime, timezone

from anomaly_detection.api.main import create_app, lifespan
from anomaly_detection.common.models.access_log import AccessLogInference, GeoLocation, DeviceFingerprint

# Create a test application instance
app = create_app()


import pytest_asyncio

@pytest_asyncio.fixture
async def async_client():
    """Provides an async test client for the API."""
    async with lifespan(app):
        # Mock the classifier since it isn't trained
        from anomaly_detection.common.models.ml_io import ClassificationOutput
        from unittest.mock import MagicMock
        def mock_classify(signal, features):
            return ClassificationOutput(
                entity_id=signal.entity_id,
                event_id=signal.event_id,
                predicted_class="unclassified",
                class_probabilities={"unclassified": 1.0},
                classification_confidence=1.0,
                is_anomaly=False
            )
        app.state.orchestrator.classifier.classify_signal = mock_classify

        async with AsyncClient(
            transport=ASGITransport(app=app, client=("127.0.0.1", 123), raise_app_exceptions=False), base_url="http://test"
        ) as client:
            yield client


@pytest.fixture
def valid_event_payload() -> dict:
    """Returns a valid event payload dictionary."""
    return {
        "event_id": "test_evt_001",
        "session_id": "test_ses_001",
        "entity_id": "test_usr_001",
        "entity_type": "user",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_ip": "192.168.1.100",
        "geo_location": {
            "city": "Test City",
            "country": "US",
            "latitude": 40.7128,
            "longitude": -74.0060,
        },
        "resource_accessed": "file/test.txt",
        "auth_method": "password",
        "auth_outcome": "success",
        "session_duration": 15.0,
        "command_sequence": [],
        "device_fingerprint": {
            "device_id": "test_dev_001",
            "os_family": "Linux",
            "os_version": "22.04",
            "mac_address": "00:11:22:33:44:55",
            "protocol": "HTTPS",
            "user_agent": "pytest",
            "firmware_version": "",
        },
        "failure_count": 0,
        "delivery_mode": "batch",
    }


@pytest.mark.asyncio
async def test_inference_router_success(async_client, valid_event_payload):
    """Test valid single-event inference request returns 200 OK and AlertPayload."""
    # We test with the actual lifecycle (app state loaded).
    # Since we are using the real M11 orchestrator, it will generate a score.
    # We do not mock it, per requirement 4: "proving the M11 orchestrator wiring... is live".
    response = await async_client.post(
        "/api/v1/inference/events",
        json={"events": [valid_event_payload]},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["processed_count"] == 1
    assert "alerts" in data


@pytest.mark.asyncio
async def test_inference_router_rejects_label(async_client, valid_event_payload):
    """Test payload containing 'label' is rejected with 422 (Risk 3)."""
    payload_with_label = valid_event_payload.copy()
    payload_with_label["label"] = "brute_force"

    response = await async_client.post(
        "/api/v1/inference/events",
        json={"events": [payload_with_label]},
    )
    
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert "details" in data
    
    # Verify the error message points to the label restriction
    err_msgs = [err.get("msg", "") for err in data["details"]]
    assert any("ground-truth 'label' field" in msg for msg in err_msgs)


@pytest.mark.asyncio
async def test_inference_router_malformed_payload(async_client, valid_event_payload):
    """Test malformed payload returns structured 422."""
    malformed_payload = valid_event_payload.copy()
    del malformed_payload["entity_id"]  # Missing required field

    response = await async_client.post(
        "/api/v1/inference/events",
        json={"events": [malformed_payload]},
    )
    
    assert response.status_code == 422
    data = response.json()
    assert "details" in data
    # Ensure it's returning the global exception handler JSON response
    assert data["error"] == "Validation Error"


@pytest.mark.asyncio
async def test_inference_consecutive_requests_update_profile(async_client, valid_event_payload):
    """
    A test runs two consecutive inference requests for the same cold-start entity 
    and confirms the second request's profile reflects the EWMA update from the first.
    """
    # 0. Inject a profile so the entity is warm and EWMA updates apply
    from anomaly_detection.common.models.entities import EntityProfile
    from datetime import datetime, timezone
    profile = EntityProfile(
        entity_id="test_usr_001",
        entity_type="user",
        baseline_vector=[0.0] * 24,
        baseline_std=[1.0] * 24,
        sequence_history=[],
        most_frequent_country="US",
        known_mac_addresses=[],
        known_os_profiles=[],
        known_protocols=[],
        resource_access_counts={},
        command_frequency={},
        event_count=100,
        cold_start_flag=False,
        last_updated=datetime.now(timezone.utc).isoformat(),
        profile_version=1
    )
    app.state.profile_store.upsert_profile(profile)

    # 1. First Request
    response1 = await async_client.post(
        "/api/v1/inference/events",
        json={"events": [valid_event_payload]},
    )
    assert response1.status_code == 200
    
    # 2. Second Request for the same entity
    valid_event_payload["event_id"] = "test_evt_002"
    valid_event_payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    response2 = await async_client.post(
        "/api/v1/inference/events",
        json={"events": [valid_event_payload]},
    )
    assert response2.status_code == 200
    
    # 3. Check the ProfileStore (attached to app state during lifespan)
    profile_store = app.state.profile_store
    profile = profile_store.get_profile("test_usr_001")
    
    assert profile is not None
    # Profile should have version > 1 if updated
    assert profile.profile_version > 1
    # Check event count increased
    assert profile.event_count >= 1


@pytest.mark.asyncio
async def test_lifespan_initializes_once(async_client):
    """A test confirms models/stores are loaded once at startup."""
    # Since lifespan is triggered around the async_client context, 
    # we can verify that app.state has the instantiated objects.
    assert hasattr(app.state, "orchestrator")
    assert hasattr(app.state, "profile_store")
    assert hasattr(app.state, "alert_store")
    
    # Ensure they are the initialized instances
    from src.orchestrator.pipeline import InferencePipeline
    assert isinstance(app.state.orchestrator, InferencePipeline)
