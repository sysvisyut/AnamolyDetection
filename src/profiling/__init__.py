"""
Behavioral Profiling Module (M06).

Provides the ProfileStore, PopulationPrior, and BehavioralProfilingModel
for scoring entity behavioral anomalies using statistical deviations.
"""

from src.profiling.config import ProfilingConfig, FEATURE_NAMES
from src.profiling.population_prior import PopulationPrior
from src.profiling.profile_store import ProfileStore
from src.profiling.profile_model import BehavioralProfilingModel, ExtendedProfilingOutput

__all__ = [
    "ProfilingConfig",
    "FEATURE_NAMES",
    "PopulationPrior",
    "ProfileStore",
    "BehavioralProfilingModel",
    "ExtendedProfilingOutput",
]
