"""Regression tests for the split profile-module public API."""
from __future__ import annotations

import unittest

from yolobattle.model_training import profiles
from yolobattle.model_training.profile_models import TrainProfile
from yolobattle.model_training.profile_registry import (
    BENCHMARK_PROFILES,
    LEGACY_SWEEP_PROFILES,
    PROFILES,
)


class ProfileModuleTest(unittest.TestCase):
    def test_public_facade_preserves_registry_identity(self):
        self.assertIs(profiles.PROFILES, PROFILES)
        self.assertIs(profiles.BENCHMARK_PROFILES, BENCHMARK_PROFILES)
        self.assertIs(profiles.LEGACY_SWEEP_PROFILES, LEGACY_SWEEP_PROFILES)
        self.assertIs(profiles.TrainProfile, TrainProfile)

    def test_profile_categories_are_disjoint_and_cover_the_registry(self):
        benchmark_names = set(BENCHMARK_PROFILES)
        legacy_names = set(LEGACY_SWEEP_PROFILES)
        self.assertFalse(benchmark_names & legacy_names)
        self.assertEqual(set(PROFILES), benchmark_names | legacy_names)


if __name__ == "__main__":
    unittest.main()
