"""Tests for Simulated Streaming (M13)."""

import os
import asyncio
import time
from datetime import datetime, timezone
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from anomaly_detection.streaming.simulated_stream import SimulatedStreamReader
from anomaly_detection.common.models.access_log import AccessLogInference


@pytest.fixture
def dummy_parquet_file(tmp_path):
    filepath = tmp_path / "test_stream.parquet"
    
    data = [
        {
            "event_id": "e1",
            "session_id": "s1",
            "entity_id": "usr_1",
            "entity_type": "user",
            "timestamp": "2026-07-26T10:00:00Z",
            "source_ip": "1.1.1.1",
            "geo_location": {"city": "City", "country": "US", "latitude": 0.0, "longitude": 0.0},
            "resource_accessed": "file/1",
            "auth_method": "password",
            "auth_outcome": "success",
            "session_duration": 10.0,
            "command_sequence": [],
            "device_fingerprint": {
                "device_id": "dev1", "os_family": "Linux", "os_version": "1.0", 
                "mac_address": "00", "protocol": "HTTP", "user_agent": "", "firmware_version": ""
            },
            "failure_count": 0,
            "label": "normal"
        },
        {
            "event_id": "e2",
            "session_id": "s1",
            "entity_id": "usr_1",
            "entity_type": "user",
            "timestamp": "2026-07-26T10:01:00Z", # 60 seconds later
            "source_ip": "1.1.1.1",
            "geo_location": {"city": "City", "country": "US", "latitude": 0.0, "longitude": 0.0},
            "resource_accessed": "file/2",
            "auth_method": "password",
            "auth_outcome": "success",
            "session_duration": 15.0,
            "command_sequence": [],
            "device_fingerprint": {
                "device_id": "dev1", "os_family": "Linux", "os_version": "1.0", 
                "mac_address": "00", "protocol": "HTTP", "user_agent": "", "firmware_version": ""
            },
            "failure_count": 0,
            "label": "brute_force"
        }
    ]
    
    df = pd.DataFrame(data)
    df.to_parquet(filepath)
    return str(filepath)


@pytest.mark.asyncio
async def test_simulated_stream_label_stripping(dummy_parquet_file):
    reader = SimulatedStreamReader(config_path="nonexistent.yaml") # defaults to 60.0
    
    events = []
    async for event in reader.read_stream(dummy_parquet_file):
        events.append(event)
        
    assert len(events) == 2
    # Ensure label is not in the object (pydantic will exclude it if it's not in the model, but let's check the dict)
    dumped = events[0].model_dump()
    assert "label" not in dumped
    assert events[0].delivery_mode == "simulated_stream"


@pytest.mark.asyncio
async def test_simulated_stream_time_compression(dummy_parquet_file):
    reader = SimulatedStreamReader(config_path="nonexistent.yaml")
    reader.compression_factor = 600.0 # 60s / 600 = 0.1s sleep
    
    # We mock asyncio.sleep to check if it's called correctly
    with patch("asyncio.sleep") as sleep_mock:
        events = []
        async for event in reader.read_stream(dummy_parquet_file):
            events.append(event)
            
        assert len(events) == 2
        
        # First event doesn't trigger sleep
        # Second event is 60 simulated seconds later. 60 / 600 = 0.1 seconds
        # Let's check the arguments passed to asyncio.sleep
        assert sleep_mock.call_count == 1
        args, kwargs = sleep_mock.call_args
        sleep_time = args[0]
        
        # With a small overhead of time.monotonic(), sleep_time should be approximately 0.1
        assert 0.05 < sleep_time <= 0.1


def test_t1_only_guard():
    import sys
    import importlib
    
    # Unload modules if present
    if "anomaly_detection.streaming.simulated_stream" in sys.modules:
        del sys.modules["anomaly_detection.streaming.simulated_stream"]
    if "anomaly_detection.api.routers.stream" in sys.modules:
        del sys.modules["anomaly_detection.api.routers.stream"]
    if "anomaly_detection.api.main" in sys.modules:
        del sys.modules["anomaly_detection.api.main"]
        
    # Python caches submodules as attributes on the parent package. We must remove it there too.
    if "anomaly_detection.api.routers" in sys.modules:
        routers_pkg = sys.modules["anomaly_detection.api.routers"]
        if hasattr(routers_pkg, "stream"):
            delattr(routers_pkg, "stream")
        
    with patch.dict(sys.modules, {"anomaly_detection.streaming.simulated_stream": None, "anomaly_detection.api.routers.stream": None}):
        # Simulate app startup with a clean module to avoid cached create_app missing things
        from anomaly_detection.api.main import create_app
        app = create_app()
        
        # Verify that read endpoints are registered
        routes = app.openapi()["paths"].keys()
        assert any(r.startswith("/api/v1/alerts") for r in routes)
        assert any(r.startswith("/api/v1/entities") for r in routes)
        
        # T2 stream route shouldn't crash the app if missing, but it might not be registered
        # because of the guard in main.py
        assert not any(r.startswith("/api/v1/stream") for r in routes)
