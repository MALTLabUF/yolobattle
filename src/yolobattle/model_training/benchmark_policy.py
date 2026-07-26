"""Immutable rules that make results comparable across training frameworks."""
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


LEGO_GEARS_224X160_V1 = BenchmarkPolicy(
    name="legogears_224x160_v1",
    width=224,
    height=160,
    split_seed=9001,
    validation_fraction=0.20,
    iterations=7000,
    validation_fractions=(0.10, 0.15, 0.20, 0.80),
)

LEATHER_256X256_V1 = BenchmarkPolicy(
    name="leather_256x256_v1",
    width=256,
    height=256,
    split_seed=9001,
    validation_fraction=0.20,
    iterations=7000,
)

FISHEYE_TRAFFIC_960X736_V1 = BenchmarkPolicy(
    name="fisheye_traffic_960x736_v1",
    width=960,
    height=736,
    split_seed=9001,
    validation_fraction=0.10,
    iterations=8000,
)

FISHEYE8K_OFFICIAL_1280X1280_V1 = BenchmarkPolicy(
    name="fisheye8k_official_1280x1280_v1",
    width=1280,
    height=1280,
    split_seed=9001,
    validation_fraction=0.30,
    iterations=8000,
    split_strategy="official",
)

CUBES_224X160_V1 = BenchmarkPolicy(
    name="cubes_224x160_v1",
    width=224,
    height=160,
    split_seed=9001,
    validation_fraction=0.20,
    iterations=7000,
    validation_fractions=(0.10, 0.15, 0.20),
)

CARDS_768X576_V1 = BenchmarkPolicy(
    name="cards_768x576_v1",
    width=768,
    height=576,
    split_seed=9001,
    validation_fraction=0.20,
    iterations=6000,
)
