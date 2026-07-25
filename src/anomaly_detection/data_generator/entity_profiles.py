"""
Per-entity behavioral profile model for the Synthetic Data Generator.

Implements the profile architecture defined in
SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Resource pool constants
# Section 2b: pre-defined resource pools for sampling.
# ---------------------------------------------------------------------------

FILE_RESOURCES: List[str] = (
    [f"file/finance/{name}" for name in [
        "payroll_2026.xlsx", "budget_q1.xlsx", "expenses_q2.csv", "audit_report.pdf",
        "invoice_3456.pdf", "contracts_2026.docx", "tax_filings.zip", "board_minutes.pdf",
        "acquisition_plan.pptx", "vendor_payments.csv", "accounts_payable.xlsx",
        "revenue_forecast.xlsx", "cash_flow_2026.pdf", "securities_holdings.pdf",
        "merger_doc.docx", "loan_agreements.pdf", "insurance_policy.pdf",
        "balance_sheet_q3.pdf", "profit_loss.xlsx", "annual_report_2025.pdf",
    ]]
    + [f"file/hr/{name}" for name in [
        "employee_records.csv", "salary_bands.xlsx", "performance_reviews_2026.pdf",
        "org_chart.pdf", "hiring_plan.docx", "pii_export.csv", "onboarding_checklist.pdf",
        "termination_log.csv", "benefits_enrollment.xlsx", "training_completion.csv",
        "headcount_report.xlsx", "succession_plan.docx", "disciplinary_records.pdf",
        "leave_balances.csv", "contractor_agreements.pdf",
    ]]
    + [f"file/engineering/{name}" for name in [
        "architecture_design.pdf", "source_code.tar.gz", "deploy_config.yaml",
        "api_keys_vault.json", "db_schema.sql", "ci_pipeline_config.yml",
        "security_scan_report.pdf", "release_notes_v3.md", "roadmap_2027.pptx",
        "postmortem_incident42.md", "runbook_prod.md", "ssl_certs.zip",
        "infra_diagram.drawio", "monitoring_alerts.json", "capacity_plan.xlsx",
    ]]
    + [f"file/legal/{name}" for name in [
        "nda_template.docx", "patent_application.pdf", "litigation_notes.pdf",
        "contract_acme.docx", "compliance_report.pdf", "gdpr_audit.pdf",
        "terms_of_service.pdf", "privacy_policy.pdf", "ip_assignment.docx",
        "shareholder_agreement.pdf",
    ]]
)

API_RESOURCES: List[str] = [
    f"api/{svc}"
    for svc in [
        "admin/users", "admin/roles", "admin/config",
        "auth/login", "auth/refresh", "auth/logout",
        "data/export", "data/import", "data/query",
        "reports/generate", "reports/download", "reports/schedule",
        "users/profile", "users/preferences", "users/activity",
        "billing/invoices", "billing/subscriptions", "billing/payments",
        "integrations/webhook", "integrations/oauth", "integrations/saml",
        "audit/logs", "audit/events", "audit/alerts",
        "search/full", "search/entities", "search/resources",
        "notifications/send", "notifications/subscribe",
        "monitoring/health", "monitoring/metrics",
    ]
]

PORT_RESOURCES: List[str] = [
    f"port/{p}" for p in [22, 80, 443, 3306, 5432, 8080, 8443, 9200, 6379, 5000]
]

DB_RESOURCES: List[str] = [
    f"db/{schema}/{table}"
    for schema, table in [
        ("core", "users"), ("core", "sessions"), ("core", "audit_log"),
        ("finance", "transactions"), ("finance", "accounts"), ("finance", "invoices"),
        ("hr", "employees"), ("hr", "departments"), ("hr", "salaries"),
        ("analytics", "events"), ("analytics", "metrics"), ("analytics", "reports"),
        ("security", "alerts"), ("security", "policies"), ("security", "roles"),
        ("ops", "deployments"), ("ops", "services"), ("ops", "configs"),
        ("data", "raw_logs"), ("data", "processed_events"),
    ]
]

ALL_RESOURCES: List[str] = FILE_RESOURCES + API_RESOURCES + PORT_RESOURCES + DB_RESOURCES

# Exfil command set per Section 3.6
EXFIL_COMMANDS: List[str] = ["scp", "rsync", "ftp", "curl", "wget", "nc"]

# Full command vocabulary per Section 2b
COMMAND_VOCABULARY: List[str] = [
    "ls", "cat", "grep", "sudo", "ssh", "scp", "rsync", "curl", "wget",
    "ps", "netstat", "chmod", "find", "tar", "vim",
]

# OS family distributions for user devices per Section 2b
OS_FAMILY_CHOICES: List[str] = ["Windows", "Linux", "macOS", "iOS"]
OS_FAMILY_WEIGHTS: List[float] = [0.60, 0.20, 0.15, 0.05]

# OS version samples per family
OS_VERSION_MAP: Dict[str, List[str]] = {
    "Windows": ["10.0", "11.0", "Server 2022"],
    "Linux": ["22.04", "20.04", "18.04"],
    "macOS": ["13.5", "14.0", "12.6"],
    "iOS": ["16.3", "17.0", "15.7"],
    "Embedded/RTU": ["FW-2.3.1", "FW-3.0.0", "FW-1.9.7"],
}

# Named city/geo pools
CITY_GEO_POOL: List[Dict[str, Any]] = [
    {"city": "Mumbai", "country": "IN", "lat": 19.0760, "lon": 72.8777},
    {"city": "Bangalore", "country": "IN", "lat": 12.9716, "lon": 77.5946},
    {"city": "Delhi", "country": "IN", "lat": 28.7041, "lon": 77.1025},
    {"city": "Kolkata", "country": "IN", "lat": 22.5726, "lon": 88.3639},
    {"city": "Hyderabad", "country": "IN", "lat": 17.3850, "lon": 78.4867},
    {"city": "Chennai", "country": "IN", "lat": 13.0827, "lon": 80.2707},
    {"city": "New York", "country": "US", "lat": 40.7128, "lon": -74.0060},
    {"city": "San Francisco", "country": "US", "lat": 37.7749, "lon": -122.4194},
    {"city": "London", "country": "GB", "lat": 51.5074, "lon": -0.1278},
    {"city": "Berlin", "country": "DE", "lat": 52.5200, "lon": 13.4050},
    {"city": "Singapore", "country": "SG", "lat": 1.3521, "lon": 103.8198},
    {"city": "Tokyo", "country": "JP", "lat": 35.6762, "lon": 139.6503},
    {"city": "Sydney", "country": "AU", "lat": -33.8688, "lon": 151.2093},
    {"city": "Dubai", "country": "AE", "lat": 25.2048, "lon": 55.2708},
    {"city": "Paris", "country": "FR", "lat": 48.8566, "lon": 2.3522},
]

# Persona definitions
USER_PERSONAS: List[str] = ["executive", "developer", "analyst", "support", "remote_worker"]
SERVICE_ACCOUNT_PERSONAS: List[str] = ["cicd_pipeline", "monitoring_agent", "etl_job", "api_integration"]
EDGE_DEVICE_PERSONAS: List[str] = ["iot_sensor", "plc_controller", "security_camera", "rtu_device"]

# Auth method distribution for users per Section 2b
USER_AUTH_METHOD_CHOICES: List[str] = ["password", "token", "certificate", "biometric"]
USER_AUTH_METHOD_WEIGHTS: List[float] = [0.60, 0.25, 0.10, 0.05]


@dataclass
class GeoPoint:
    """A geographic city/coordinate pair."""
    city: str
    country: str
    lat: float
    lon: float


@dataclass
class DeviceRecord:
    """A registered device fingerprint for an entity."""
    device_id: str
    os_family: str
    os_version: str
    mac_address: str
    protocol: str
    user_agent: str
    firmware_version: str


@dataclass
class EntityProfile:
    """Per-entity behavioral profile.

    This dataclass holds all static parameters that govern event generation
    for a single entity. It is the source of truth for what 'normal' looks
    like for that entity, implementing the profile hierarchy described in
    SYNTHETIC_DATA_GENERATOR_DESIGN.md Section 2a.

    The profile is serializable to JSON/dict for reproducibility and later
    use by the concept-drift simulation.
    """

    # Core identity
    entity_id: str
    entity_type: str  # "user", "service_account", "edge_device"
    persona: str

    # Late Joiner flag per Section 2f
    is_late_joiner: bool = False

    # Temporal profile — Section 2b login timing
    active_hour_center: float = 12.0   # μ_h: mean login hour
    active_hour_spread: float = 2.0    # σ_h: std of login hour distribution
    has_schedule_drift: bool = False    # Tier 3: 10% of users have slow drift
    drift_rate_hours_per_week: float = 0.0  # ρ_h: hours per 7 days
    has_role_expansion: bool = False    # Tier 3: 5% of users expand ResourceSet
    role_expansion_resources_per_10d: int = 0  # new resources per 10 days

    # Geographic profile — Section 2b
    home_geo_set: List[GeoPoint] = field(default_factory=list)
    # Weights for sampling from home_geo_set; parallel to home_geo_set
    home_geo_weights: List[float] = field(default_factory=list)

    # Resource profile — Section 2b
    resource_set: List[str] = field(default_factory=list)
    # Sampling weights (Dirichlet-drawn); parallel to resource_set
    resource_weights: List[float] = field(default_factory=list)

    # Device profile — Section 2b
    device_set: List[DeviceRecord] = field(default_factory=list)
    # Sampling weights; parallel to device_set ([0.85, 0.15] for 2 devices)
    device_weights: List[float] = field(default_factory=list)

    # Auth profile — Section 2b
    primary_auth_method: str = "password"

    # Session profile — Section 2b / 2c / 2d
    session_duration_mu: float = 7.0    # log-normal μ_s
    session_duration_sigma: float = 0.8  # log-normal σ_s

    # Command profile — Section 2b (users only)
    command_pool: List[str] = field(default_factory=list)
    privileged_session_prob: float = 0.0  # probability of non-empty command_sequence

    # Service account / edge device specifics
    event_interval_seconds: float = 300.0  # near-deterministic interval
    event_interval_jitter: float = 5.0     # ε std in seconds

    # Normal subnet prefix for IP generation (e.g., "192.168.1")
    normal_ip_subnet: str = "192.168.1"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the profile to a JSON-compatible dictionary.

        The profile is serializable so that it can be saved alongside generated
        data for reproducibility and later use by the drift simulation.
        """
        d = asdict(self)
        # Convert GeoPoint and DeviceRecord objects (dataclasses) are handled by asdict
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityProfile":
        """Reconstruct an EntityProfile from its serialized dict form."""
        # Reconstruct nested dataclasses
        home_geo_set = [GeoPoint(**g) for g in data.pop("home_geo_set", [])]
        device_set = [DeviceRecord(**d) for d in data.pop("device_set", [])]
        return cls(home_geo_set=home_geo_set, device_set=device_set, **data)

    def to_json(self) -> str:
        """Serialize the profile to a JSON string."""
        return json.dumps(self.to_dict())
