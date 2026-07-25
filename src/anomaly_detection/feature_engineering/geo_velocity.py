"""
Geo-velocity computation utilities.

ARCHITECTURE COMPONENT: Feature Engineering
BOUNDARY RESPONSIBILITY: Internal helper; does not cross a named boundary directly.
TIER: T1

Provides the Haversine formula for computing great-circle distance between
two geographic coordinates, and derives km/h geo-velocity from a pair of
(GeoLocation, timestamp) tuples.

All geo features are critical for Impossible Travel detection (dim 6, 7).

Edge cases handled per the acceptance criteria:
  - Same location consecutive logins: velocity = 0.0
  - Missing geo data (None): returns 0.0 velocity
  - Wrap-around for coordinates near ±180° longitude: Haversine handles
    this natively via the atan2 formula — no special casing needed.
  - Time delta of zero seconds: returns 0.0 velocity (cannot divide by 0)
  - Very short time intervals with large distance: capped at
    MAX_GEO_VELOCITY_KMPH (2000 km/h) normalised to 1.0
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from anomaly_detection.common.models.access_log import GeoLocation
from anomaly_detection.feature_engineering.config import MAX_GEO_VELOCITY_KMPH


# ---------------------------------------------------------------------------
# Physical / mathematical constants
# ---------------------------------------------------------------------------

#: Mean radius of the Earth in kilometres (WGS84 semi-major axis ≈ 6378 km,
#: mean radius ≈ 6371 km — using 6371 km per standard Haversine convention).
EARTH_RADIUS_KM: float = 6371.0


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Compute the great-circle distance between two points on the Earth.

    Uses the Haversine formula, which correctly handles the wrap-around
    case for coordinates near ±180° longitude without special casing.

    Args:
        lat1: Latitude of point 1 in decimal degrees (WGS84).
        lon1: Longitude of point 1 in decimal degrees (WGS84).
        lat2: Latitude of point 2 in decimal degrees (WGS84).
        lon2: Longitude of point 2 in decimal degrees (WGS84).

    Returns:
        Great-circle distance in kilometres.  Returns 0.0 when the two
        points are identical (same location consecutive logins).

    Attack relevance:
        dim 6 (geo_velocity_kmph) — Impossible Travel detection.
    """
    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    # Haversine naturally handles ±180° longitude wrap-around via modular
    # arithmetic implicit in math.sin/cos on the difference.
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    sin_half_dphi = math.sin(delta_phi / 2.0)
    sin_half_dlambda = math.sin(delta_lambda / 2.0)
    a = (
        sin_half_dphi * sin_half_dphi
        + math.cos(phi1) * math.cos(phi2) * sin_half_dlambda * sin_half_dlambda
    )
    # Clamp to [0, 1] to guard against floating-point rounding producing a
    # value marginally above 1 for identical coordinates.
    a = max(0.0, min(1.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def compute_geo_velocity(
    prev_geo: Optional[GeoLocation],
    curr_geo: Optional[GeoLocation],
    elapsed_seconds: float,
) -> Tuple[float, float]:
    """
    Compute geo-velocity between two consecutive events for the same entity.

    Returns both the raw km/h value and its normalised [0, 1] form
    (capped at MAX_GEO_VELOCITY_KMPH).

    Args:
        prev_geo: GeoLocation of the previous event for this entity, or
                  ``None`` if this is the first event (cold-start).
        curr_geo: GeoLocation of the current event.
        elapsed_seconds: Time difference in seconds between the two events.
                         Must be non-negative.

    Returns:
        A 2-tuple ``(raw_kmph, normalised_velocity)`` where:
            - ``raw_kmph``: actual speed in km/h (uncapped).
            - ``normalised_velocity``: ``min(raw_kmph / MAX_GEO_VELOCITY_KMPH, 1.0)``
              This is the value stored in dim 6 of the feature vector.

    Edge cases:
        - ``prev_geo`` is ``None`` (first event / cold-start): returns (0.0, 0.0).
        - ``elapsed_seconds == 0`` (same timestamp or extremely close): returns
          (0.0, 0.0) to avoid division-by-zero; the zero-velocity assumption
          is conservative (prevents false positives on same-instant duplicates).
        - Same location, different time: returns (0.0, 0.0).
        - Very high velocity (supersonic): capped at 1.0 in normalised output.

    Attack relevance:
        dim 6 (geo_velocity_kmph) — primary signal for Impossible Travel.
        Any value > 800 km/h is physically impossible for ground/air travel,
        making this a near-perfect decision rule for that attack class.
    """
    # Cold-start: no previous geo-location available
    if prev_geo is None or curr_geo is None:
        return (0.0, 0.0)

    # Zero elapsed time: cannot compute meaningful velocity
    if elapsed_seconds <= 0.0:
        return (0.0, 0.0)

    distance_km = haversine_distance_km(
        prev_geo.latitude,
        prev_geo.longitude,
        curr_geo.latitude,
        curr_geo.longitude,
    )

    # Convert seconds → hours, then km/h
    elapsed_hours = elapsed_seconds / 3600.0
    raw_kmph = distance_km / elapsed_hours

    # Cap and normalise
    normalised = min(raw_kmph / MAX_GEO_VELOCITY_KMPH, 1.0)
    return (raw_kmph, normalised)
