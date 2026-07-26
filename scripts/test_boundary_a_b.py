import asyncio
from anomaly_detection.data_generator.generator import DataGenerator
from anomaly_detection.data_generator.config import GeneratorConfig
from anomaly_detection.streaming.simulated_stream import SimulatedStreamReader
import os

async def main():
    config = GeneratorConfig(num_entities=2, days_of_data=1, attack_injection_rate=0.0)
    gen = DataGenerator(config)
    run_id = gen.generate(output_dir="data")
    
    filepath = f"data/raw/synthetic_logs_{run_id}.parquet"
    
    reader = SimulatedStreamReader()
    # just grab the first event
    async for event in reader.read_stream(filepath):
        print(f"Successfully consumed event: {event.event_id}")
        break

if __name__ == "__main__":
    asyncio.run(main())
