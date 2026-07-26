#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import uuid
import json
from datetime import datetime, timedelta

def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/labeled", exist_ok=True)
    
    run_id = "testrun123"
    
    # Generate 1000 events, over 30 days
    start_date = datetime(2023, 1, 1)
    
    events = []
    labels = []
    
    for i in range(1000):
        evt_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        # Spread across 30 days
        ts = start_date + timedelta(days=(i / 1000.0) * 30)
        
        is_anomaly = (i % 10 == 0)
        label = "brute_force" if is_anomaly else "normal"
        
        evt = {
            "event_id": evt_id,
            "session_id": session_id,
            "entity_id": "usr_00000001",
            "entity_type": "user",
            "timestamp": ts.isoformat() + "Z",
            "source_ip": "192.168.1.1",
            "geo_location": json.dumps({"city": "New York", "country": "US", "latitude": 40.7, "longitude": -74.0}),
            "resource_accessed": "file/test.txt",
            "auth_method": "password",
            "auth_outcome": "failure" if is_anomaly else "success",
            "session_duration": 0.0 if is_anomaly else 300.0,
            "command_sequence": json.dumps([]),
            "device_fingerprint": json.dumps({"device_id": "dev1", "os_family": "Windows", "os_version": "10", "mac_address": "00:11:22:33:44:55", "protocol": "HTTPS", "user_agent": "Chrome", "firmware_version": ""}),
            "failure_count": 5 if is_anomaly else 0,
        }
        events.append(evt)
        labels.append({"event_id": evt_id, "label": label})
        
    pd.DataFrame(events).to_parquet(f"data/raw/synthetic_logs_{run_id}.parquet")
    pd.DataFrame(labels).to_parquet(f"data/labeled/labels_{run_id}.parquet")
    
    print(f"Mock datasets created for run_id {run_id}")

if __name__ == "__main__":
    main()
