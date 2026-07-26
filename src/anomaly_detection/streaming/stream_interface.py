"""
Stream Interface (M13 - Simulated Streaming T2).
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from anomaly_detection.common.models.access_log import AccessLogInference


class StreamReader(ABC):
    """
    Abstract base class for reading the event stream.
    Implementations must yield `AccessLogInference` events.
    """

    @abstractmethod
    async def read_stream(self, filepath: str) -> AsyncGenerator[AccessLogInference, None]:
        """
        Reads the stream from the given dataset and yields inference-ready events.
        
        Args:
            filepath: Path to the dataset (e.g. parquet file).
            
        Yields:
            AccessLogInference events.
        """
        pass
        yield  # type: ignore
