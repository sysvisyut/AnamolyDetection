"""
Enumerations for data schema models.

Matches DATA_SCHEMA.md constraints.
"""

from enum import Enum


class EntityType(str, Enum):
    """The type of entity being modeled."""
    USER = "user"
    SERVICE_ACCOUNT = "service_account"
    EDGE_DEVICE = "edge_device"


class AuthMethod(str, Enum):
    """The method used for authentication."""
    PASSWORD = "password"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    NONE = "none"


class AnomalyCategory(str, Enum):
    """
    The taxonomy of attack classes plus normal.
    M02 requirements include 'unclassified'.
    """
    NORMAL = "normal"
    BRUTE_FORCE = "brute_force"
    IMPOSSIBLE_TRAVEL = "impossible_travel"
    CREDENTIAL_STUFFING = "credential_stuffing"
    LATERAL_MOVEMENT = "lateral_movement"
    DEVICE_SPOOFING = "device_spoofing"
    LOW_AND_SLOW = "low_and_slow"
    INSIDER_DRIFT = "insider_drift"
    UNCLASSIFIED = "unclassified"


class EntityStatus(str, Enum):
    """The cold-start or drift status of the entity."""
    WARM = "warm"
    COLD_START = "cold_start"
    DRIFT_ADAPTED = "drift_adapted"
