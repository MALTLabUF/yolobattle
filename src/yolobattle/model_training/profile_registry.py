"""Canonical benchmark profiles and preserved legacy/sweep variants."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict

from yolobattle.model_training.benchmark_definitions import (
    CARDS_V1,
    CUBES_V1,
    FISHEYE8K_OFFICIAL_V1,
    FISHEYE_TRAFFIC_LOCAL_V1,
    LEATHER_V1,
    LEGO_GEARS_V1,
)
from yolobattle.model_training.profile_models import (
    TrainProfile,
    benchmark_profile,
    legacy_variant,
    lego_gears_profile,
)


# Internal base for the public legacy-style PyTorch sweep below.  Keeping it
# out of PROFILES prevents a fixed-budget PyTorch variant from being selected
# accidentally alongside the equalized sweep.
_LEGOGEARS_PYTORCH_BASE = lego_gears_profile(
    root="LegoGears_v2", name="LegoGearsPyTorchYOLOv4", backend="pytorch_yolov4",
    data_path="", cfg_out="", batch_size=64, subdivisions=1, learning_rate=0.00261,
    mosaic=0, jitter=0.3, hue=0.1, saturation=1.5, exposure=1.5, flip=0,
    num_gpus=1, pytorch_cfg="cfg/yolov4-tiny.cfg",
)


BENCHMARK_PROFILES: Dict[str, TrainProfile] = {
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
        batch_size=64, subdivisions=1, learning_rate=0.00261, templates=("yolov4-tiny", "yolov7-tiny", "yolov4-tiny-3l"),
        sweep_keys=("templates", "num_gpus"), sweep_values={"num_gpus": (1,)},
    ),
    "CardsUltra": benchmark_profile(
        definition=CARDS_V1, root="/workspace/ccr_playing_cards", name="CardsUltra", backend="ultralytics",
        data_path="", cfg_out="", batch_size=64, subdivisions=1, learning_rate=0.00261, templates=(),
        sweep_keys=("templates", "num_gpus", "ultra_model"),
        sweep_values={"num_gpus": (1,), "ultra_model": ("yolo11n.pt", "yolo11s.pt")}, ultra_data="", ultra_model="yolo11n.pt",
    ),
}


LEGACY_SWEEP_PROFILES: Dict[str, TrainProfile] = {
    "LegoGearsDarknet": legacy_variant(
        BENCHMARK_PROFILES["LegoGearsDarknetBenchmark"], "LegoGearsDarknet",
        val_fracs=LEGO_GEARS_V1.policy.validation_fractions, sweep_keys=("templates", "val_fracs", "num_gpus"),
    ),
    "LegoGearsUltra": legacy_variant(
        BENCHMARK_PROFILES["LegoGearsUltraBenchmark"], "LegoGearsUltra",
        val_fracs=LEGO_GEARS_V1.policy.validation_fractions, sweep_keys=("val_fracs", "num_gpus", "ultra_model"),
    ),
    "LegoGearsPyTorchYOLOv4": legacy_variant(
        _LEGOGEARS_PYTORCH_BASE, "LegoGearsPyTorchYOLOv4",
        # The 80% validation split leaves only 18 images for training, fewer
        # than PyTorch-YOLOv4's 64-image micro-batch.  The supported sweep
        # therefore covers the three comparable split sizes.
        val_fracs=(0.10, 0.15, 0.20), sweep_keys=("val_fracs",),
    ),
    "LeatherDarknet": legacy_variant(
        BENCHMARK_PROFILES["LeatherDarknetBenchmark"], "LeatherDarknet",
        color_presets=(None, "preserve"), tag_color_preset=True, sweep_keys=("templates", "color_presets", "num_gpus"),
    ),
    "LeatherUltra": legacy_variant(BENCHMARK_PROFILES["LeatherUltraBenchmark"], "LeatherUltra", tag_color_preset=True),
    "FisheyeTrafficDarknetLocal": legacy_variant(
        BENCHMARK_PROFILES["FisheyeTrafficDarknetBenchmark"], "FisheyeTrafficDarknetLocal", sweep_keys=("templates",), sweep_values={},
    ),
    "FisheyeTrafficDarknetLocalLRSweep": legacy_variant(
        BENCHMARK_PROFILES["FisheyeTrafficDarknetBenchmark"], "FisheyeTrafficDarknetLocalLRSweep",
        learning_rate=0.0013, sweep_keys=("templates", "learning_rate"),
        sweep_values={"learning_rate": (0.0010, 0.0013, 0.0020, 0.00261, 0.0040)},
    ),
    "FisheyeTrafficDarknetLocalJPG": legacy_variant(
        BENCHMARK_PROFILES["FisheyeTrafficDarknetBenchmark"], "FisheyeTrafficDarknetLocalJPG", sweep_keys=("templates",), sweep_values={},
        dataset=replace(
            FISHEYE_TRAFFIC_LOCAL_V1.dataset_at("/blue/ranka/ibraheem.qureshi/images"),
            names="/blue/ranka/j.fleischer/annotation_data/obj.names", prefix="combined_ibraheem", flat_dir=".",
        ),
    ),
    "FisheyeTrafficUltralyticsLocal": legacy_variant(
        BENCHMARK_PROFILES["FisheyeTrafficUltraBenchmark"], "FisheyeTrafficUltralyticsLocal",
        sweep_keys=("num_gpus",), sweep_values={"num_gpus": (1,)},
    ),
    "FishEye8KDarknet": legacy_variant(BENCHMARK_PROFILES["FishEye8KDarknetBenchmark"], "FishEye8KDarknet"),
    "FishEye8KUltralytics": legacy_variant(BENCHMARK_PROFILES["FishEye8KUltraBenchmark"], "FishEye8KUltralytics"),
    "CubesDarknet": legacy_variant(
        BENCHMARK_PROFILES["CubesDarknetBenchmark"], "CubesDarknet",
        val_fracs=CUBES_V1.policy.validation_fractions, sweep_keys=("val_fracs", "templates", "num_gpus"),
    ),
    "CubesUltra": legacy_variant(
        BENCHMARK_PROFILES["CubesUltraBenchmark"], "CubesUltra",
        val_fracs=CUBES_V1.policy.validation_fractions, sweep_keys=("val_fracs", "templates", "num_gpus", "ultra_model"),
    ),
}


UNCANONICALIZED_PROFILES: Dict[str, TrainProfile] = {}
PROFILES: Dict[str, TrainProfile] = {
    **BENCHMARK_PROFILES,
    **LEGACY_SWEEP_PROFILES,
    **UNCANONICALIZED_PROFILES,
}
