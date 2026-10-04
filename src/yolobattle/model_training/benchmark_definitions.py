"""Dataset/benchmark types; concrete settings live in benchmarks/<dataset>.py."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Tuple

from yolobattle.model_training.benchmark_policy import BenchmarkPolicy


@dataclass(frozen=True)
class DatasetSpec:
    """A dataset recipe resolved to a concrete runtime mount location."""

    root: str
    sets: Tuple[str, ...]
    classes: int
    names: str
    prefix: str
    split_seed: int = 9001
    neg_subdirs: Tuple[str, ...] = tuple()
    exts: Tuple[str, ...] = (".jpg",)
    flat_dir: str | None = None
    legos: bool = False
    url: str | None = None
    sha256: str | None = None
    require_existing: bool = False
    predefined_train_dir: str | None = None
    predefined_valid_dir: str | None = None
    class_names: Tuple[str, ...] = tuple()
    # Ground-truth source for framework-independent COCO evaluation.  ``auto``
    # retains legacy JSON-first detection; known datasets should declare their
    # actual format so unrelated metadata cannot change the evaluation path.
    annotation_format: str = "auto"

    # Path inside the unmodified archive; root remains the extraction directory.
    archive_subdir: str | None = None

    @property
    def content_root(self) -> Path:
        """Directory containing names, images, and labels after extraction."""
        root = Path(self.root).resolve()
        if self.archive_subdir is None:
            return root
        subdir = Path(self.archive_subdir)
        if subdir.is_absolute() or ".." in subdir.parts:
            raise ValueError("archive_subdir must be a relative path within the dataset root")
        content = (root / subdir).resolve()
        if not content.is_relative_to(root):
            raise ValueError("archive_subdir resolves outside the dataset root")
        return content


@dataclass(frozen=True)
class DatasetRecipe:
    """Framework-independent dataset identity, excluding its runtime root."""

    sets: Tuple[str, ...]
    classes: int
    names: str
    prefix: str
    neg_subdirs: Tuple[str, ...] = tuple()
    exts: Tuple[str, ...] = (".jpg",)
    flat_dir: str | None = None
    legos: bool = False
    url: str | None = None
    sha256: str | None = None
    require_existing: bool = False
    predefined_train_dir: str | None = None
    predefined_valid_dir: str | None = None
    class_names: Tuple[str, ...] = tuple()
    annotation_format: str = "auto"
    archive_subdir: str | None = None

    def at(self, root: str, *, split_seed: int = 9001) -> DatasetSpec:
        return DatasetSpec(root=root, split_seed=split_seed, **self.__dict__)


@dataclass(frozen=True)
class BenchmarkDefinition:
    """One canonical policy and one canonical dataset identity."""

    name: str
    policy: BenchmarkPolicy
    dataset_recipe: DatasetRecipe

    def dataset_at(self, root: str) -> DatasetSpec:
        return self.policy.dataset(self.dataset_recipe.at(root))

    def fingerprint(self) -> str:
        """Stable policy-and-dataset identity, excluding a runtime mount path."""
        payload = json.dumps({
            "name": self.name,
            "policy": asdict(self.policy),
            "dataset_recipe": asdict(self.dataset_recipe),
        }, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()[:16]


# Compatibility for previous imports; new settings live in benchmarks/<dataset>.py.
_LEGACY_EXPORTS = {
    "LEGO_GEARS_V1": "lego_gears",
    "LEATHER_V1": "leather",
    "FISHEYE_TRAFFIC_LOCAL_V1": "fisheye_traffic",
    "FISHEYE8K_OFFICIAL_V1": "fisheye8k",
    "CUBES_V1": "cubes",
    "CARDS_V1": "cards",
    "ARABIC_HANDWRITING_V1": "arabic_handwriting",
}

__all__ = ["DatasetSpec", "DatasetRecipe", "BenchmarkDefinition"] + list(_LEGACY_EXPORTS)


def __getattr__(name: str):
    # Lazy forwarding avoids a cycle: benchmark modules use the classes above.
    if name in _LEGACY_EXPORTS:
        from importlib import import_module

        module = import_module(f".benchmarks.{_LEGACY_EXPORTS[name]}", __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LEGACY_EXPORTS))
