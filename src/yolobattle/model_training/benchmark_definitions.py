"""Canonical dataset identities paired with framework-neutral benchmark policy."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Tuple

from yolobattle.model_training.benchmark_policy import (
    BenchmarkPolicy,
    CARDS_768X576_V1,
    CUBES_224X160_V1,
    FISHEYE8K_OFFICIAL_1280X1280_V1,
    FISHEYE_TRAFFIC_960X736_V1,
    LEATHER_256X256_V1,
    LEGO_GEARS_224X160_V1,
)


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


LEGO_GEARS_V1 = BenchmarkDefinition(
    name="legogears_v1",
    policy=LEGO_GEARS_224X160_V1,
    dataset_recipe=DatasetRecipe(
        sets=("set_01", "set_02_empty", "set_03"),
        classes=5,
        names="LegoGears.names",
        prefix="LegoGears",
        neg_subdirs=("set_02_empty",),
        exts=(".jpg",),
        url="https://www.ccoderun.ca/programming/2024-05-01_LegoGears/legogears_2_dataset.zip",
        sha256="126980d3e43986bbd3d785ac16f6430e9bf3b726e65a30574bb3c9ba06a4462e",
    ),
)

LEATHER_V1 = BenchmarkDefinition(
    name="leather_v1",
    policy=LEATHER_256X256_V1,
    dataset_recipe=DatasetRecipe(
        sets=("color", "cut", "fold", "glue", "poke", "good_1", "good_2"),
        classes=5,
        names="leather.names",
        prefix="leather",
        neg_subdirs=("good_1", "good_2"),
        exts=(".jpg", ".png"),
        url="https://g-665dcc.55ba.08cc.data.globus.org/leather_oct_25.zip",
        sha256="87fba3c49bce7342af51e1fe6df5a470862f201c0e8e25bf3ea80a0c6f238d8c",
        flat_dir="darkmark_image_cache/resize",
    ),
)

FISHEYE_TRAFFIC_LOCAL_V1 = BenchmarkDefinition(
    name="fisheye_traffic_local_v1",
    policy=FISHEYE_TRAFFIC_960X736_V1,
    dataset_recipe=DatasetRecipe(
        sets=tuple(), classes=5, names="obj.names", prefix="combined",
        exts=(".jpg", ".png"), require_existing=True,
        flat_dir="darkmark_image_cache/resize",
    ),
)

FISHEYE8K_OFFICIAL_V1 = BenchmarkDefinition(
    name="fisheye8k_official_v1",
    policy=FISHEYE8K_OFFICIAL_1280X1280_V1,
    dataset_recipe=DatasetRecipe(
        sets=tuple(), classes=5, names="FishEye8K.names", prefix="FishEye8K_official",
        exts=(".jpg", ".jpeg", ".png"), require_existing=True,
        predefined_train_dir="train/images", predefined_valid_dir="test/images",
        class_names=("Bus", "Bike", "Car", "Pedestrian", "Truck"),
    ),
)

CUBES_V1 = BenchmarkDefinition(
    name="cubes_v1",
    policy=CUBES_224X160_V1,
    dataset_recipe=DatasetRecipe(
        sets=tuple(), classes=4, names="cubes.names", prefix="cubes",
        exts=(".jpg", ".png"),
        url="https://g-665dcc.55ba.08cc.data.globus.org/refinedcubes.zip",
        sha256="8764c5086e1cada0b66de5198df11655009315873bc9245fd44741ff6e31f4e0",
        flat_dir="darkmark_image_cache/resize",
    ),
)

CARDS_V1 = BenchmarkDefinition(
    name="cards_v1",
    policy=CARDS_768X576_V1,
    dataset_recipe=DatasetRecipe(
        sets=tuple(), classes=19, names="ccr_playing_cards.names", prefix="ccr_playing_cards",
        exts=(".jpg", ".png"),
        url="https://g-665dcc.55ba.08cc.data.globus.org/playing_cards.zip",
        sha256="432d6da3a2fbec5d1dadd3278b5c4c21ccbaa2dbcd72e087daf193e9bdaf3cc4",
        flat_dir="darkmark_image_cache/resize",
    ),
)
