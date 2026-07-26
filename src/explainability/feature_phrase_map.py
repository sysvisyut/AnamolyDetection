"""
Feature phrase mapping for the Explainability Layer (M09).
Contains the HUMAN_LABEL_MAP used to translate technical features into narrative phrases.
"""

# Mapping dictionary for all 24 feature dimensions
HUMAN_LABEL_MAP = {
    "hour_of_day_sin": {
        "human_label": "Login time (hour of day)",
        "toward_anomaly": "an off-hours login at {value} UTC",
        "toward_normal": "login during normal hours"
    },
    "hour_of_day_cos": {
        "human_label": "Login time (hour cycle)",
        "toward_anomaly": "an off-hours login at {value} UTC",
        "toward_normal": "login during normal hours"
    },
    "day_of_week_sin": {
        "human_label": "Day of week",
        "toward_anomaly": "unusual {value} access",
        "toward_normal": "access on a normal workday"
    },
    "day_of_week_cos": {
        "human_label": "Day of week",
        "toward_anomaly": "unusual {value} access",
        "toward_normal": "access on a normal workday"
    },
    "session_duration_norm": {
        "human_label": "Session length",
        "toward_anomaly": "an unusually long session ({value})",
        "toward_normal": "normal session length"
    },
    "failure_count_norm": {
        "human_label": "Authentication failure count",
        "toward_anomaly": "{value} consecutive authentication failures",
        "toward_normal": "no unusual authentication failures"
    },
    "geo_velocity_kmph": {
        "human_label": "Speed between logins (km/h)",
        "toward_anomaly": "a geo-velocity of {value} km/h between consecutive logins",
        "toward_normal": "normal login locations"
    },
    "is_new_geo": {
        "human_label": "New geographic location",
        "toward_anomaly": "a new country ({value}) not in this entity's location history",
        "toward_normal": "a known login location"
    },
    "resource_category_enc": {
        "human_label": "Resource category accessed",
        "toward_anomaly": "access to an unusual resource category ({value})",
        "toward_normal": "access to a normal resource category"
    },
    "resource_rarity_score": {
        "human_label": "Resource access rarity",
        "toward_anomaly": "access to a resource rarely or never accessed before (rarity score {value:.2f})",
        "toward_normal": "access to a frequently-used resource"
    },
    "auth_method_enc": {
        "human_label": "Authentication method",
        "toward_anomaly": "use of an unusual authentication method ({value})",
        "toward_normal": "use of the entity's normal authentication method"
    },
    "auth_outcome_enc": {
        "human_label": "Authentication outcome",
        "toward_anomaly": "authentication failure",
        "toward_normal": "successful authentication"
    },
    "command_seq_length_norm": {
        "human_label": "Command sequence length",
        "toward_anomaly": "an unusually long command sequence ({value} commands)",
        "toward_normal": "normal command sequence length"
    },
    "command_rarity_score": {
        "human_label": "Command rarity",
        "toward_anomaly": "use of commands rarely issued by this entity (rarity score {value:.2f})",
        "toward_normal": "use of the entity's normal command set"
    },
    "has_exfil_command": {
        "human_label": "Data transfer command detected",
        "toward_anomaly": "a data transfer command (scp/rsync/curl/wget) was issued",
        "toward_normal": "no data transfer commands detected"
    },
    "fingerprint_os_match": {
        "human_label": "OS profile match",
        "toward_anomaly": "an OS mismatch on device {device_id} (expected {expected_os}, saw {actual_os})",
        "toward_normal": "OS matches the registered device profile"
    },
    "fingerprint_mac_match": {
        "human_label": "MAC address match",
        "toward_anomaly": "a MAC address mismatch on device {device_id} (expected {expected_mac}, saw {actual_mac})",
        "toward_normal": "MAC address matches the registered device profile"
    },
    "fingerprint_protocol_match": {
        "human_label": "Protocol match",
        "toward_anomaly": "a protocol mismatch on device {device_id} (expected {expected_proto}, saw {actual_proto})",
        "toward_normal": "protocol matches the registered device profile"
    },
    "entity_type_enc": {
        "human_label": "Entity type",
        "toward_anomaly": "an unusual entity type ({value})",
        "toward_normal": "a standard entity type"
    },
    "inter_event_gap_norm": {
        "human_label": "Time since last login",
        "toward_anomaly": "an unusually long gap ({value} hours) since this entity's previous login",
        "toward_normal": "normal login frequency"
    },
    "session_event_count_norm": {
        "human_label": "Events in current session",
        "toward_anomaly": "{value} events in the current session (unusually high volume)",
        "toward_normal": "normal session depth"
    },
    "resource_breadth_norm": {
        "human_label": "Resource variety in session",
        "toward_anomaly": "access to {value} distinct resources in a single session (unusually broad)",
        "toward_normal": "access to a normal number of resources"
    },
    "ip_entity_ratio": {
        "human_label": "Entities reached from this IP",
        "toward_anomaly": "this IP address was used to attempt access to {value} distinct entities in the past 24 hours",
        "toward_normal": "this IP address is associated with a normal number of entities"
    },
    "entity_ip_ratio": {
        "human_label": "IPs used by this entity",
        "toward_anomaly": "this entity used {value} distinct IP addresses in the past 24 hours",
        "toward_normal": "this entity used a normal number of IP addresses"
    }
}
