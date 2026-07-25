"""
Raw Access Log models and related sub-structures.
"""

from typing import List
from pydantic import BaseModel, Field, ConfigDict

from anomaly_detection.common.models.enums import AnomalyCategory


class GeoLocation(BaseModel):
    """Geographical location information."""
    city: str = Field(..., description='e.g. "Mumbai" (Faker-generated)')
    country: str = Field(..., description='ISO 3166-1 alpha-2, e.g. "IN"')
    latitude: float = Field(..., ge=-90.0, le=90.0, description="WGS84, range [-90.0, 90.0]")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="WGS84, range [-180.0, 180.0]")
    
    model_config = ConfigDict(from_attributes=True)


class CommandEntry(BaseModel):
    """An individual command executed within a privileged session."""
    sequence_position: int = Field(..., ge=0, description="0-indexed position within the session")
    command: str = Field(..., description='e.g. "sudo", "ssh", "curl", "scp", "grep"')
    target: str = Field(..., description="resource or host the command acted on; empty string if N/A")
    outcome: str = Field(..., description='enum: "success", "failure", "denied"')
    elapsed_seconds: float = Field(..., ge=0.0, description="seconds since session start at time of command")

    model_config = ConfigDict(from_attributes=True)


class DeviceFingerprint(BaseModel):
    """Stable fingerprint for edge devices and known endpoints."""
    device_id: str = Field(..., description='stable device identifier, e.g. "dev_3f8a21bc"')
    os_family: str = Field(..., description='e.g. "Windows", "Linux", "iOS", "Embedded/RTU"')
    os_version: str = Field(..., description='e.g. "11.0", "22.04", "16.3"')
    mac_address: str = Field(..., description='format: "AA:BB:CC:DD:EE:FF" (Faker-generated, not real)')
    protocol: str = Field(..., description='primary protocol used, e.g. "HTTPS", "Modbus", "MQTT", "RDP"')
    user_agent: str = Field(..., description="for HTTP-based access; empty string for non-HTTP protocols")
    firmware_version: str = Field(..., description="for edge devices; empty string for standard OS endpoints")

    model_config = ConfigDict(from_attributes=True)


class AccessLogBase(BaseModel):
    """Base fields shared by all access logs."""
    event_id: str = Field(..., description="UUID v4; globally unique across all runs")
    session_id: str = Field(..., description="UUID v4; groups related events")
    entity_id: str = Field(..., description="The entity whose behavior is being modeled")
    entity_type: str = Field(..., description='Enum: "user", "service_account", "edge_device"')
    timestamp: str = Field(..., description="ISO-8601 with UTC timezone; microsecond precision")
    source_ip: str = Field(..., description="IPv4 dotted-decimal")
    geo_location: GeoLocation = Field(..., description="Geographical location sub-structure")
    resource_accessed: str = Field(..., description="Format: <category>/<identifier>")
    auth_method: str = Field(..., description='Enum: "password", "token", "certificate", "biometric", "none"')
    auth_outcome: str = Field(..., description='Enum: "success", "failure", "mfa_required"')
    session_duration: float = Field(..., ge=0.0, description="Seconds; 0.0 for failed auth events")
    command_sequence: List[CommandEntry] = Field(..., description="Empty list [] for non-privileged sessions")
    device_fingerprint: DeviceFingerprint = Field(..., description="Device fingerprint sub-structure")
    failure_count: int = Field(..., ge=0, description="Count of consecutive auth failures immediately preceding this event")

    model_config = ConfigDict(from_attributes=True)


class AccessLogTraining(AccessLogBase):
    """Training schema for access logs containing the ground-truth label."""
    label: AnomalyCategory = Field(..., description="Ground-truth class; never crosses boundary B")


class AccessLogInference(AccessLogBase):
    """
    Inference schema for access logs with label explicitly removed.
    Also adds delivery_mode metadata.
    Note: Label is structurally absent per requirements.
    """
    delivery_mode: str = Field(..., description="Enum: batch or simulated_stream")

    model_config = ConfigDict(from_attributes=True, extra="forbid")
