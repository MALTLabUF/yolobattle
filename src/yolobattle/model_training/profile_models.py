"""Framework-specific training profile model and construction helpers."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

from yolobattle.model_training.benchmark_definitions import BenchmarkDefinition, DatasetSpec, LEGO_GEARS_V1
from yolobattle.model_training.benchmark_policy import BenchmarkPolicy


@dataclass(frozen=True)
class TrainProfile:
    name: str
    backend: str
    data_path: str
    cfg_out: str
    width: int
    height: int
    batch_size: int
    subdivisions: int
    iterations: int
    learning_rate: float
    # Darknet-compatible image augmentation.  Backends that expose these
    # controls receive the values explicitly instead of relying on their
    # implementation defaults.
    mosaic: int = 0
    jitter: float = 0.3
    hue: float = 0.1
    saturation: float = 1.5
    exposure: float = 1.5
    flip: int = 0
    template: str | None = None
    templates: Tuple[str, ...] = tuple()
    val_fracs: Tuple[float, ...] = (0.20,)
    color_preset: Optional[str] = None
    color_presets: Tuple[Optional[str], ...] = (None,)
    tag_color_preset: bool = False
    map_thresh: float | None = None
    iou_thresh: float | None = None
    map_points: int | None = None
    dataset: DatasetSpec | None = None
    epochs: int | None = None
    ultra_data: str = "LG_v2.yaml"
    ultra_model: str = "yolo11n.pt"
    pytorch_cfg: str = "cfg/yolov4-tiny.cfg"
    policy: BenchmarkPolicy | None = None
    benchmark: BenchmarkDefinition | None = None
    training_seed: Optional[int] = None
    num_gpus: Optional[int] = None
    sweep_keys: Tuple[str, ...] = tuple()
    sweep_values: Dict[str, Tuple[Any, ...]] = field(default_factory=dict)


def benchmark_profile(
    *, definition: BenchmarkDefinition, root: str,
    val_fracs: Tuple[float, ...] | None = None, **kwargs: Any,
) -> TrainProfile:
    """Build a profile from one canonical dataset/policy definition."""
    policy = definition.policy
    return TrainProfile(
        width=policy.width,
        height=policy.height,
        iterations=policy.iterations,
        val_fracs=val_fracs or (policy.validation_fraction,),
        dataset=definition.dataset_at(root),
        policy=policy,
        benchmark=definition,
        **kwargs,
    )


def lego_gears_profile(
    *, root: str, val_fracs: Tuple[float, ...] | None = None,
    definition: BenchmarkDefinition = LEGO_GEARS_V1, **kwargs: Any,
) -> TrainProfile:
    return benchmark_profile(definition=definition, root=root, val_fracs=val_fracs, **kwargs)


def effective_policy(profile: TrainProfile) -> BenchmarkPolicy | None:
    """Resolve a run's effective split and update budget for provenance."""
    if profile.policy is None or not profile.val_fracs:
        return profile.policy
    return replace(
        profile.policy,
        validation_fraction=float(profile.val_fracs[0]),
        iterations=profile.iterations,
    )


def legacy_variant(base: TrainProfile, name: str, **changes: Any) -> TrainProfile:
    """Keep an old profile name while inheriting only its experimental delta."""
    return replace(base, name=name, policy=None, benchmark=None, **changes)
