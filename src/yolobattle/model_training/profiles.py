"""Stable public profile API.

Definitions, registry entries, and legacy equalization live in focused sibling
modules; existing callers can continue importing from this module.
"""
from yolobattle.model_training.benchmark_definitions import DatasetSpec
from yolobattle.model_training.equalization import equalize_for_split, read_split_counts_from_data
from yolobattle.model_training.profile_models import TrainProfile, effective_policy
from yolobattle.model_training.profile_registry import (
    BENCHMARK_PROFILES,
    LEGACY_SWEEP_PROFILES,
    PROFILES,
    UNCANONICALIZED_PROFILES,
)

__all__ = [
    "BENCHMARK_PROFILES",
    "DatasetSpec",
    "LEGACY_SWEEP_PROFILES",
    "PROFILES",
    "TrainProfile",
    "UNCANONICALIZED_PROFILES",
    "effective_policy",
    "equalize_for_split",
    "get_profile",
    "read_split_counts_from_data",
]


def get_profile(key: str) -> TrainProfile:
    if key not in PROFILES:
        raise KeyError(f"Unknown profile '{key}'. Available: {', '.join(PROFILES)}")
    return PROFILES[key]
