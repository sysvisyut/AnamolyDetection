"""
Population-level prior statistical profiles for cold-start entities (M06).
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np

from anomaly_detection.common.models.features import EntityFeatureVector
from anomaly_detection.common.models.enums import EntityType
from src.profiling.config import ProfilingConfig


class PopulationPrior:
    """
    Builds and serves population-level statistical baselines.
    Used as the fallback profile for entities with 'cold_start' status.
    """

    def __init__(self, config: ProfilingConfig) -> None:
        self.config = config
        # Mapping from EntityType (as string) to a tuple of (mean_vector, std_vector)
        self._priors: Dict[str, Tuple[List[float], List[float]]] = {}

    def fit(
        self,
        vectors: List[EntityFeatureVector],
        entity_types: List[EntityType]
    ) -> None:
        """
        Compute population mean and std dev across all provided feature vectors.
        Groups calculations by EntityType so users, service_accounts, and
        edge_devices have independent population priors.

        Args:
            vectors: List of EntityFeatureVectors from the training population.
            entity_types: Parallel list of EntityType for each vector.
        """
        if not vectors:
            return

        grouped_data = defaultdict(list)
        for vec, etype in zip(vectors, entity_types):
            grouped_data[etype.value].append(vec.root)

        for etype_str, data_list in grouped_data.items():
            arr = np.array(data_list, dtype=np.float64)
            mean_vec = np.mean(arr, axis=0).tolist()
            std_vec = np.std(arr, axis=0).tolist()
            
            # Apply minimum variance epsilon
            std_vec = [max(s, self.config.variance_epsilon) for s in std_vec]
            
            self._priors[etype_str] = (mean_vec, std_vec)

    def get_prior(self, entity_type: EntityType) -> Tuple[List[float], List[float]]:
        """
        Retrieve the population prior for the given entity type.

        Args:
            entity_type: The type of entity (user, service_account, edge_device).

        Returns:
            Tuple of (mean_vector, std_vector).
            
        Raises:
            KeyError: If no prior exists for the given entity type (e.g. fit not called).
        """
        return self._priors[entity_type.value]

    def save(self) -> None:
        """
        Persist the population priors to disk atomically.
        """
        path = self.config.population_prior_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = f"{path}.tmp"
        
        with open(temp_path, "w") as f:
            json.dump(self._priors, f, indent=2)
            
        # Atomic rename
        os.replace(temp_path, path)

    def load(self) -> None:
        """
        Load the population priors from disk.
        """
        path = self.config.population_prior_path
        if os.path.exists(path):
            with open(path, "r") as f:
                self._priors = json.load(f)
