"""Explicit DRAD feature profiles for the comprehensive baseline experiment."""
from math import isclose

FEATURE_ORDER = (
    "comm_degree", "comm_betweenness", "power_degree", "power_capacity",
    "infection_risk", "uncertainty",
)
DEFAULT_RISK_PROFILE = "distributed-feature"
RISK_PROFILES = {
    "distributed-feature": (0.20, 0.15, 0.15, 0.15, 0.20, 0.15),
    "degree-prioritized": (0.36, 0.18, 0.08, 0.08, 0.20, 0.10),
}


def get_risk_weights(profile=DEFAULT_RISK_PROFILE):
    """Return a fresh mapping; keep the frozen supplementary core independent."""
    try:
        values = RISK_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown risk profile {profile!r}; choose {tuple(RISK_PROFILES)}") from exc
    if len(values) != len(FEATURE_ORDER) or not isclose(sum(values), 1.0):
        raise ValueError(f"Invalid risk weights for {profile}")
    return dict(zip(FEATURE_ORDER, values))
