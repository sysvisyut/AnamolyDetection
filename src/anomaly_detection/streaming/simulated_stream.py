"""
Simulated Stream implementation (T2).
"""

import asyncio
import time
from typing import AsyncGenerator
from datetime import datetime, timezone

import pandas as pd
import yaml

from anomaly_detection.common.models.access_log import AccessLogInference
from anomaly_detection.streaming.stream_interface import StreamReader


class SimulatedStreamReader(StreamReader):
    """
    Reads a dataset and replays it in timestamp order with time-compression.
    """

    def __init__(self, config_path: str = "config/streaming.yaml"):
        self.compression_factor = 60.0
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if config and "streaming" in config and "compression_factor" in config["streaming"]:
                    self.compression_factor = float(config["streaming"]["compression_factor"])
        except (FileNotFoundError, yaml.YAMLError):
            pass

    async def read_stream(self, filepath: str) -> AsyncGenerator[AccessLogInference, None]:
        """
        Reads the stream from the given dataset and yields inference-ready events.
        Strips 'label' and adds 'delivery_mode="simulated_stream"'.
        Time compression is applied by sleeping between events based on their timestamps.
        """
        df = pd.read_parquet(filepath)
        # Ensure it's sorted by timestamp
        df = df.sort_values(by="timestamp")
        
        last_event_time = None
        last_wall_time = None

        for _, row in df.iterrows():
            event_dict = row.to_dict()
            
            # Strip label for boundary B compliance
            if "label" in event_dict:
                del event_dict["label"]
            
            event_dict["delivery_mode"] = "simulated_stream"
            
            # Parse timestamp to apply time-compression sleep
            event_time = datetime.fromisoformat(event_dict["timestamp"].replace("Z", "+00:00"))
            current_wall_time = time.monotonic()
            
            if last_event_time is not None and last_wall_time is not None:
                simulated_delta = (event_time - last_event_time).total_seconds()
                if simulated_delta > 0:
                    sleep_time = simulated_delta / self.compression_factor
                    
                    # Account for the time spent processing
                    elapsed = current_wall_time - last_wall_time
                    sleep_time -= elapsed
                    
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
            
            last_event_time = event_time
            
            # Create the inference event model
            inference_event = AccessLogInference(**event_dict)
            yield inference_event
            
            # Update last wall time after yield (processing might have happened)
            last_wall_time = time.monotonic()
