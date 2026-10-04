"""Shared policy type; concrete policies live in benchmarks/<dataset>.py."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yolobattle.model_training.benchmark_definitions import DatasetSpec


@dataclass(frozen=True)
class BenchmarkPolicy:
    """Framework-neutral split, geometry, and evaluation rules."""

    name: str
    width: int
    height: int
    split_seed: int
    validation_fraction: float
    iterations: int
    # Optional supported split sweep.  ``validation_fraction`` remains the
    # single canonical split used for framework comparisons.
    validation_fractions: tuple[float, ...] = tuple()
    # "random" means the split is derived with split_seed.  "official" means
    # the dataset's supplied train/validation partition is used verbatim.
    split_strategy: str = "random"
    export_confidence: float = 0.01
    export_nms_iou: float = 0.45
    coco_iou_thresholds: tuple[float, ...] = tuple(round(0.50 + index * 0.05, 2) for index in range(10))
    confusion_confidence: float = 0.50
    confusion_iou: float = 0.50
    checkpoint_selector: str = "final"

    def fingerprint(self) -> str:
        """Stable identifier recorded with benchmark artifacts and tests."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()[:16]

    def dataset(self, dataset: "DatasetSpec") -> "DatasetSpec":
        """Apply the policy's split seed without mutating a dataset recipe."""
        return replace(dataset, split_seed=self.split_seed)


# Compatibility for previous imports; new settings live in benchmarks/<dataset>.py.
_LEGACY_EXPORTS = {
    "LEGO_GEARS_224X160_V1": "lego_gears",
    "LEATHER_256X256_V1": "leather",
    "FISHEYE_TRAFFIC_960X736_V1": "fisheye_traffic",
    "FISHEYE8K_OFFICIAL_1280X1280_V1": "fisheye8k",
    "CUBES_224X160_V1": "cubes",
    "CARDS_768X576_V1": "cards",
    "ARABIC_HANDWRITING_352X256_V1": "arabic_handwriting",
}

__all__ = ["BenchmarkPolicy"] + list(_LEGACY_EXPORTS)


def __getattr__(name: str):
    # Lazy forwarding avoids a cycle: benchmark modules use the classes above.
    if name in _LEGACY_EXPORTS:
        from importlib import import_module

        module = import_module(f".benchmarks.{_LEGACY_EXPORTS[name]}", __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LEGACY_EXPORTS))
