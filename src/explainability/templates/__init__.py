from src.explainability.templates.brute_force import PREFIX as BRUTE_FORCE_PREFIX
from src.explainability.templates.impossible_travel import PREFIX as IMPOSSIBLE_TRAVEL_PREFIX
from src.explainability.templates.credential_stuffing import PREFIX as CREDENTIAL_STUFFING_PREFIX
from src.explainability.templates.lateral_movement import PREFIX as LATERAL_MOVEMENT_PREFIX
from src.explainability.templates.device_spoofing import PREFIX as DEVICE_SPOOFING_PREFIX
from src.explainability.templates.low_and_slow import PREFIX as LOW_AND_SLOW_PREFIX, SEQUENCE_PREFIX as LOW_AND_SLOW_SEQ_PREFIX
from src.explainability.templates.insider_drift import PREFIX as INSIDER_DRIFT_PREFIX, TEMPLATE as INSIDER_DRIFT_TEMPLATE
from src.explainability.templates.unclassified import PREFIX as UNCLASSIFIED_PREFIX
from src.explainability.templates.cold_start_modifier import MODIFIER as COLD_START_MODIFIER

PREFIXES = {
    "brute_force": BRUTE_FORCE_PREFIX,
    "impossible_travel": IMPOSSIBLE_TRAVEL_PREFIX,
    "credential_stuffing": CREDENTIAL_STUFFING_PREFIX,
    "lateral_movement": LATERAL_MOVEMENT_PREFIX,
    "device_spoofing": DEVICE_SPOOFING_PREFIX,
    "low_and_slow": LOW_AND_SLOW_PREFIX,
    "insider_drift": INSIDER_DRIFT_PREFIX,
    "unclassified": UNCLASSIFIED_PREFIX
}
