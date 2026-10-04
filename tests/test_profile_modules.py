"""Regression tests for the split profile-module public API."""
from __future__ import annotations

import unittest
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

from yolobattle.model_training import profiles
from yolobattle.model_training.profile_models import TrainProfile
from yolobattle.model_training.profile_registry import (
    BENCHMARK_PROFILES,
    BENCHMARK_MODULES,
    LEGACY_SWEEP_PROFILES,
    PROFILES,
    _collect_profiles,
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

    def test_old_constant_imports_work_in_fresh_processes(self):
        # Start with each old/new entry point to catch circular imports hidden
        # by this test process's already-imported registry.
        entry_points = [
            "benchmark_policy", "benchmark_definitions", "profile_models", "profiles",
            *(f"benchmarks.{module.__name__.rsplit('.', 1)[-1]}" for module in BENCHMARK_MODULES),
        ]
        for entry_point in entry_points:
            with self.subTest(entry_point=entry_point):
                result = subprocess.run(
                    [sys.executable, "-c", '''
import importlib
import sys
base = "yolobattle.model_training"
first = importlib.import_module(f"{base}.{sys.argv[1]}")
for old_name in ("benchmark_policy", "benchmark_definitions"):
    old = importlib.import_module(f"{base}.{old_name}")
    for name, owner in old._LEGACY_EXPORTS.items():
        current = importlib.import_module(f"{base}.benchmarks.{owner}")
        assert getattr(old, name) is getattr(current, name), name
from yolobattle.model_training.profile_models import lego_gears_profile
from yolobattle.model_training.benchmarks.lego_gears import BENCHMARK_PROFILES
profile = BENCHMARK_PROFILES["LegoGearsDarknetBenchmark"]
copy = lego_gears_profile(root=profile.dataset.root, name=profile.name,
    backend=profile.backend, data_path=profile.data_path, cfg_out=profile.cfg_out,
    batch_size=profile.batch_size, subdivisions=profile.subdivisions,
    learning_rate=profile.learning_rate, templates=profile.templates,
    sweep_keys=profile.sweep_keys, sweep_values=profile.sweep_values)
assert copy == profile
''', entry_point], capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_dataset_modules_own_their_benchmarks_and_policies(self):
        for module in BENCHMARK_MODULES:
            for name, profile in module.BENCHMARK_PROFILES.items():
                with self.subTest(profile=name):
                    self.assertIs(PROFILES[name], profile)
                    self.assertTrue(any(value is profile.benchmark for value in vars(module).values()))
                    self.assertTrue(any(value is profile.policy for value in vars(module).values()))

    def test_duplicate_names_across_modules_or_categories_are_rejected(self):
        name, profile = next(iter(BENCHMARK_PROFILES.items()))
        first = SimpleNamespace(__name__="first", BENCHMARK_PROFILES={name: profile}, LEGACY_SWEEP_PROFILES={})
        for category in ("BENCHMARK_PROFILES", "LEGACY_SWEEP_PROFILES"):
            second = SimpleNamespace(__name__="second", BENCHMARK_PROFILES={}, LEGACY_SWEEP_PROFILES={})
            getattr(second, category)[name] = profile
            with self.subTest(category=category), self.assertRaisesRegex(ValueError, "Duplicate profile"):
                _collect_profiles((first, second))

    def test_registry_key_must_match_profile_name(self):
        name, profile = next(iter(BENCHMARK_PROFILES.items()))
        module = SimpleNamespace(
            __name__="mismatch", BENCHMARK_PROFILES={name: replace(profile, name="different")},
            LEGACY_SWEEP_PROFILES={},
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            _collect_profiles((module,))


if __name__ == "__main__":
    unittest.main()
