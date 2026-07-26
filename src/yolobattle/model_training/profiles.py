from __future__ import annotations
from dataclasses import dataclass, replace, field
from typing import Tuple, Dict, Optional, Any
from pathlib import Path
import json
import math

from yolobattle.model_training.benchmark_policy import (
    BenchmarkPolicy,
)
from yolobattle.model_training.benchmark_definitions import (
    BenchmarkDefinition,
    CARDS_V1,
    CUBES_V1,
    DatasetSpec,
    FISHEYE8K_OFFICIAL_V1,
    FISHEYE_TRAFFIC_LOCAL_V1,
    LEATHER_V1,
    LEGO_GEARS_V1,
)

# Immutable policies and canonical dataset identities are defined in
# benchmark_policy.py and benchmark_definitions.py.

@dataclass(frozen=True)
class TrainProfile:
    name: str
    backend: str              # "darknet" or "ultralytics"
    data_path: str
    cfg_out: str

    # training knobs
    width: int
    height: int
    batch_size: int
    subdivisions: int
    iterations: int
    learning_rate: float

    # darknet template selection
    template: str | None = None
    templates: Tuple[str, ...] = tuple()
    val_fracs: Tuple[float, ...] = (0.20,)

    # SINGLE color knob: None -> keep template HSV; otherwise a preset name or "s,e,h"
    color_preset: Optional[str] = None
    color_presets: Tuple[Optional[str], ...] = (None,) # sweep list, e.g. (None, "preserve")

    tag_color_preset: bool = False

    # mAP evaluation knobs (darknet)
    map_thresh: float | None = None
    iou_thresh: float | None = None
    map_points: int | None = None

    # dataset recipe for split regen
    dataset: DatasetSpec | None = None

    # ultralytics only
    epochs: int | None = None
    ultra_data: str = "LG_v2.yaml"
    ultra_model: str = "yolo11n.pt"

    # Tianxiaomo/PyTorch-YOLOv4 only
    pytorch_cfg: str = "cfg/yolov4-tiny.cfg"

    # Shared, framework-neutral rules for comparable benchmarks.
    policy: BenchmarkPolicy | None = None
    benchmark: BenchmarkDefinition | None = None

    # ultralytics-only training RNG; ignored by Darknet
    training_seed: Optional[int] = None

    # request N GPUs (slice of visible devices). None => use all visible.
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
    """Compose a LegoGears profile from its canonical definition."""
    return benchmark_profile(
        definition=definition,
        root=root,
        val_fracs=val_fracs,
        **kwargs,
    )


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
    """Keep an old profile name while inheriting its canonical base recipe.

    Legacy variants intentionally have no policy: their historical sweeps retain
    their original runner behavior, while only the varying fields live here.
    """
    return replace(base, name=name, policy=None, benchmark=None, **changes)


# ---------------- equalization helpers (profiles-level policy) ----------------

# cache target epochs so the first encountered split sets a shared baseline
# for each (profile, template, color_preset) group across val_fracs.
_equalize_cache: Dict[tuple, float] = {}

def _manifest_from_data_path(data_path: str) -> Path:
    """'/path/LegoGears_v15.data' -> '/path/LegoGears_v15_split.json'."""
    p = Path(data_path)
    return p.with_name(p.stem + "_split.json")

def read_split_counts_from_data(data_path: str) -> Tuple[int, int]:
    """
    Return (n_train, n_valid) using the split manifest produced by dataset_setup.py.
    """
    mpath = _manifest_from_data_path(data_path)
    js = json.loads(Path(mpath).read_text(encoding="utf-8"))
    c = js.get("counts", {})
    return int(c.get("train_total", 0)), int(c.get("valid_total", 0))

def equalize_for_split(
    profile: TrainProfile,
    *,
    data_path: str,
    mode: str = "iterations",
    target_epochs: float | None = None,
) -> TrainProfile:
    """
    Returns a new TrainProfile where either iterations or batch_size has been
    adjusted so that approx_epochs ≈ constant across splits.

      approx_epochs ≈ (iterations * batch_size) / train_images

    mode:
      - "iterations" (recommended): keep batch the same, solve iterations.
      - "batch": keep iterations the same, solve batch (must be multiple of subdivisions).
    """
    global _equalize_cache

    # Count training images for this split (via manifest)
    try:
        T, _ = read_split_counts_from_data(data_path)
    except Exception:
        T = 0
    if T <= 0:
        return replace(profile, data_path=data_path)

    if target_epochs is None:
        # Establish target epochs on first call for this profile/template/color group.
        # Intentionally do NOT include val_frac: this keeps approx epochs comparable
        # across validation fractions by rescaling iterations with train-image count.
        key = (
            profile.name,
            getattr(profile, "template", None),
            getattr(profile, "color_preset", None),
        )
        if key not in _equalize_cache:
            _equalize_cache[key] = (profile.iterations * profile.batch_size) / max(1, T)
            if _equalize_cache[key] <= 0:
                _equalize_cache[key] = 1.0
        target_epochs = _equalize_cache[key]
    else:
        try:
            target_epochs = float(target_epochs)
        except Exception:
            target_epochs = 1.0
        if (not math.isfinite(target_epochs)) or target_epochs <= 0:
            target_epochs = 1.0

    if mode == "iterations":
        new_iter = int(math.ceil(target_epochs * T / max(1, profile.batch_size)))
        new_iter = max(100, new_iter)
        return replace(profile, data_path=data_path, iterations=new_iter)

    elif mode == "batch":
        k = max(1, profile.subdivisions)
        raw = target_epochs * T / max(1, profile.iterations)
        new_batch = int(round(raw / k) * k)
        new_batch = min(max(new_batch, k), 1024)
        return replace(profile, data_path=data_path, batch_size=new_batch)

    # Unknown mode; only inject the data_path
    return replace(profile, data_path=data_path)

# ---------------- canonical benchmark profiles ----------------

BENCHMARK_PROFILES = {
    "LegoGearsDarknetBenchmark": lego_gears_profile(
        root="/workspace/LegoGears_v2", name="LegoGearsDarknetBenchmark", backend="darknet",
        data_path="/workspace/LegoGears_v2/LegoGears.data", cfg_out="/workspace/LegoGears_v2/LegoGears.cfg",
        batch_size=64, subdivisions=1, learning_rate=0.00261,
        templates=("yolov4-tiny", "yolov7-tiny"),
        sweep_keys=("templates", "num_gpus"), sweep_values={"num_gpus": (1,)},
    ),
    "LegoGearsUltraBenchmark": lego_gears_profile(
        root="LegoGears_v2", name="LegoGearsUltraBenchmark", backend="ultralytics",
        data_path="", cfg_out="", batch_size=64, subdivisions=1, learning_rate=0.00261,
        templates=(), sweep_keys=("num_gpus", "ultra_model"),
        sweep_values={"num_gpus": (1,), "ultra_model": ("yolo11n.pt", "yolo11s.pt", "yolo26n.pt", "yolo26s.pt")},
        ultra_data="", ultra_model="yolo11n.pt",
    ),
    "LegoGearsPyTorchYOLOv4": lego_gears_profile(
        root="LegoGears_v2", name="LegoGearsPyTorchYOLOv4", backend="pytorch_yolov4",
        data_path="", cfg_out="", batch_size=16, subdivisions=1, learning_rate=0.00261,
        num_gpus=1, pytorch_cfg="cfg/yolov4-tiny.cfg",
    ),
    "LeatherDarknetBenchmark": benchmark_profile(
        definition=LEATHER_V1, root="/workspace/leather", name="LeatherDarknetBenchmark", backend="darknet",
        data_path="/workspace/leather/leather.data", cfg_out="/workspace/leather/leather.cfg",
        batch_size=64, subdivisions=1, learning_rate=0.00261, templates=("yolov4-tiny", "yolov7-tiny"),
        color_presets=(None,), sweep_keys=("templates", "num_gpus"), sweep_values={"num_gpus": (1,)},
    ),
    "LeatherUltraBenchmark": benchmark_profile(
        definition=LEATHER_V1, root="/workspace/leather", name="LeatherUltraBenchmark", backend="ultralytics",
        data_path="", cfg_out="", batch_size=64, subdivisions=1, learning_rate=0.00261, templates=(),
        color_presets=(None,), sweep_keys=("num_gpus", "ultra_model"),
        sweep_values={"num_gpus": (1,), "ultra_model": ("yolo11n.pt", "yolo11s.pt")}, ultra_data="", ultra_model="yolo11n.pt",
    ),
    "FisheyeTrafficDarknetBenchmark": benchmark_profile(
        definition=FISHEYE_TRAFFIC_LOCAL_V1, root="/blue/ranka/j.fleischer/annotation_data", name="FisheyeTrafficDarknetBenchmark", backend="darknet",
        data_path="/host_workspace/combined.data", cfg_out="/host_workspace/combined.cfg", batch_size=64, subdivisions=16,
        learning_rate=0.00261, templates=("yolov4", "yolov7"), sweep_keys=("templates", "num_gpus"), sweep_values={"num_gpus": (1,)},
    ),
    "FisheyeTrafficUltraBenchmark": benchmark_profile(
        definition=FISHEYE_TRAFFIC_LOCAL_V1, root="/blue/ranka/j.fleischer/annotation_data", name="FisheyeTrafficUltraBenchmark", backend="ultralytics",
        data_path="", cfg_out="", batch_size=64, subdivisions=16, learning_rate=0.00261, templates=(),
        sweep_keys=("num_gpus", "ultra_model"), sweep_values={"num_gpus": (1,)}, ultra_data="", ultra_model="yolo11n.pt",
    ),
    "FishEye8KDarknetBenchmark": benchmark_profile(
        definition=FISHEYE8K_OFFICIAL_V1, root="/blue/ranka/j.fleischer/Fisheye8K_all_including_trainandtest", name="FishEye8KDarknetBenchmark", backend="darknet",
        data_path="/workspace/.cache/splits/FishEye8K_official.data", cfg_out="/workspace/FishEye8K.cfg", batch_size=64, subdivisions=16,
        learning_rate=0.00261, templates=("yolov4", "yolov7"), sweep_keys=("templates", "num_gpus"), sweep_values={"num_gpus": (1,)},
    ),
    "FishEye8KUltraBenchmark": benchmark_profile(
        definition=FISHEYE8K_OFFICIAL_V1, root="/blue/ranka/j.fleischer/Fisheye8K_all_including_trainandtest", name="FishEye8KUltraBenchmark", backend="ultralytics",
        data_path="", cfg_out="", batch_size=64, subdivisions=16, learning_rate=0.00261, templates=(),
        sweep_keys=("num_gpus", "ultra_model"), sweep_values={"num_gpus": (1,)}, ultra_data="", ultra_model="yolo11n.pt",
    ),
    "CubesDarknetBenchmark": benchmark_profile(
        definition=CUBES_V1, root="/workspace/cubes", name="CubesDarknetBenchmark", backend="darknet",
        data_path="/workspace/cubes/cubes.data", cfg_out="/workspace/cubes/cubes.cfg", batch_size=64, subdivisions=1,
        learning_rate=0.00261, templates=("yolov4-tiny", "yolov7-tiny"), color_preset="preserve", color_presets=("preserve",),
        tag_color_preset=True, sweep_keys=("templates", "num_gpus"), sweep_values={"num_gpus": (1,)},
    ),
    "CubesUltraBenchmark": benchmark_profile(
        definition=CUBES_V1, root="/workspace/cubes", name="CubesUltraBenchmark", backend="ultralytics",
        data_path="", cfg_out="", batch_size=64, subdivisions=1, learning_rate=0.00261, templates=(),
        color_preset="preserve", color_presets=("preserve",), tag_color_preset=True,
        sweep_keys=("num_gpus", "ultra_model"), sweep_values={"num_gpus": (1,), "ultra_model": ("yolo11n.pt", "yolo11s.pt")},
        ultra_data="", ultra_model="yolo11n.pt",
    ),
    "CardsDarknet": benchmark_profile(
        definition=CARDS_V1, root="/workspace/ccr_playing_cards", name="CardsDarknet", backend="darknet",
        data_path="/workspace/ccr_playing_cards/ccr_playing_cards.data", cfg_out="/workspace/ccr_playing_cards/ccr_playing_cards.cfg",
        batch_size=64, subdivisions=1, learning_rate=0.00261,
        templates=("yolov4-tiny", "yolov7-tiny", "yolov4-tiny-3l"),
        sweep_keys=("templates", "num_gpus"), sweep_values={"num_gpus": (1,)},
    ),
    "CardsUltra": benchmark_profile(
        definition=CARDS_V1, root="/workspace/ccr_playing_cards", name="CardsUltra", backend="ultralytics",
        data_path="", cfg_out="", batch_size=64, subdivisions=1, learning_rate=0.00261, templates=(),
        sweep_keys=("templates", "num_gpus", "ultra_model"),
        sweep_values={"num_gpus": (1,), "ultra_model": ("yolo11n.pt", "yolo11s.pt")},
        ultra_data="", ultra_model="yolo11n.pt",
    ),
}

# ---------------- legacy and sweep variants ----------------

LEGACY_SWEEP_PROFILES = {
    "LegoGearsDarknet": legacy_variant(
        BENCHMARK_PROFILES["LegoGearsDarknetBenchmark"], "LegoGearsDarknet",
        val_fracs=LEGO_GEARS_V1.policy.validation_fractions,
        sweep_keys=("templates", "val_fracs", "num_gpus"),
    ),
    "LeatherDarknet": legacy_variant(
        BENCHMARK_PROFILES["LeatherDarknetBenchmark"], "LeatherDarknet",
        color_presets=(None, "preserve"), tag_color_preset=True,
        sweep_keys=("templates", "color_presets", "num_gpus"),
    ),

    "LegoGearsUltra": legacy_variant(
        BENCHMARK_PROFILES["LegoGearsUltraBenchmark"], "LegoGearsUltra",
        val_fracs=LEGO_GEARS_V1.policy.validation_fractions,
        sweep_keys=("val_fracs", "num_gpus", "ultra_model"),
    ),
    "LeatherUltra": legacy_variant(
        BENCHMARK_PROFILES["LeatherUltraBenchmark"], "LeatherUltra", tag_color_preset=True,
    ),


    "FisheyeTrafficDarknetLocal": legacy_variant(
        BENCHMARK_PROFILES["FisheyeTrafficDarknetBenchmark"], "FisheyeTrafficDarknetLocal",
        sweep_keys=("templates",), sweep_values={},
    ),
    "FisheyeTrafficDarknetLocalLRSweep": legacy_variant(
        BENCHMARK_PROFILES["FisheyeTrafficDarknetBenchmark"], "FisheyeTrafficDarknetLocalLRSweep",
        learning_rate=0.0013, sweep_keys=("templates", "learning_rate"),
        sweep_values={"learning_rate": (0.0010, 0.0013, 0.0020, 0.00261, 0.0040)},
    ),
    "FisheyeTrafficDarknetLocalJPG": legacy_variant(
        BENCHMARK_PROFILES["FisheyeTrafficDarknetBenchmark"], "FisheyeTrafficDarknetLocalJPG",
        sweep_keys=("templates",), sweep_values={},
        dataset=replace(
            FISHEYE_TRAFFIC_LOCAL_V1.dataset_at("/blue/ranka/ibraheem.qureshi/images"),
            root="/blue/ranka/ibraheem.qureshi/images",
            names="/blue/ranka/j.fleischer/annotation_data/obj.names",
            prefix="combined_ibraheem", flat_dir=".",
        ),
    ),
    "FisheyeTrafficUltralyticsLocal": legacy_variant(
        BENCHMARK_PROFILES["FisheyeTrafficUltraBenchmark"], "FisheyeTrafficUltralyticsLocal",
        sweep_keys=("num_gpus",), sweep_values={"num_gpus": (1,)},
    ),

    # FishEye8K ships an official camera-disjoint train/test split.  Do not
    # replace it with a random frame split: adjacent frames from one camera
    # are highly correlated and would leak into validation.
    "FishEye8KDarknet": legacy_variant(
        BENCHMARK_PROFILES["FishEye8KDarknetBenchmark"], "FishEye8KDarknet",
    ),

    "FishEye8KUltralytics": legacy_variant(
        BENCHMARK_PROFILES["FishEye8KUltraBenchmark"], "FishEye8KUltralytics",
    ),

    "CubesDarknet": legacy_variant(
        BENCHMARK_PROFILES["CubesDarknetBenchmark"], "CubesDarknet",
        val_fracs=CUBES_V1.policy.validation_fractions,
        sweep_keys=("val_fracs", "templates", "num_gpus"),
    ),
    "CubesUltra": legacy_variant(
        BENCHMARK_PROFILES["CubesUltraBenchmark"], "CubesUltra",
        val_fracs=CUBES_V1.policy.validation_fractions,
        sweep_keys=("val_fracs", "templates", "num_gpus", "ultra_model"),
    ),
}


# No remaining profiles fall outside a canonical benchmark definition.
UNCANONICALIZED_PROFILES: Dict[str, TrainProfile] = {}


# Public lookup registry.  Keep category dictionaries above so their intent is
# visible without changing any established profile names.
PROFILES = {
    **BENCHMARK_PROFILES,
    **LEGACY_SWEEP_PROFILES,
    **UNCANONICALIZED_PROFILES,
}

def get_profile(key: str) -> TrainProfile:
    if key not in PROFILES:
        raise KeyError(f"Unknown profile '{key}'. Available: {', '.join(PROFILES.keys())}")
    return PROFILES[key]
