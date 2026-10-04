"""Collect profiles from dataset modules; benchmark settings live in benchmarks/."""
from types import ModuleType

from .benchmarks import (
    arabic_handwriting,
    cards,
    cubes,
    fisheye8k,
    fisheye_traffic,
    leather,
    lego_gears,
)
from .profile_models import TrainProfile


# Add a dataset module here after defining its policy, recipe, and profiles.
BENCHMARK_MODULES = (
    lego_gears,
    leather,
    fisheye_traffic,
    fisheye8k,
    cubes,
    cards,
    arabic_handwriting,
)


def _collect_profiles(modules: tuple[ModuleType, ...]):
    benchmarks: dict[str, TrainProfile] = {}
    legacy: dict[str, TrainProfile] = {}
    owners: dict[str, str] = {}
    for module in modules:
        for category, destination in (
            ("BENCHMARK_PROFILES", benchmarks),
            ("LEGACY_SWEEP_PROFILES", legacy),
        ):
            for name, profile in getattr(module, category).items():
                if name in owners:
                    raise ValueError(
                        f"Duplicate profile {name!r} in {owners[name]} and {module.__name__}"
                    )
                if name != profile.name:
                    raise ValueError(
                        f"Profile key {name!r} does not match {profile.name!r} in {module.__name__}"
                    )
                owners[name] = module.__name__
                destination[name] = profile
    return benchmarks, legacy


BENCHMARK_PROFILES, LEGACY_SWEEP_PROFILES = _collect_profiles(BENCHMARK_MODULES)
UNCANONICALIZED_PROFILES: dict[str, TrainProfile] = {}
PROFILES: dict[str, TrainProfile] = {
    **BENCHMARK_PROFILES,
    **LEGACY_SWEEP_PROFILES,
    **UNCANONICALIZED_PROFILES,
}
