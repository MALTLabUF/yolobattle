"""LegoGears: dataset, comparison policy, and training profiles.

Edit the policy for geometry/splits/budget and the profiles for YOLO models.
"""
from ..benchmark_definitions import BenchmarkDefinition, DatasetRecipe
from ..benchmark_policy import BenchmarkPolicy
from ..profile_models import benchmark_profile, legacy_variant


LEGO_GEARS_224X160_V1 = BenchmarkPolicy(
    name="legogears_224x160_v1",
    width=224,
    height=160,
    split_seed=9001,
    validation_fraction=0.20,
    iterations=7000,
    validation_fractions=(0.10, 0.15, 0.20, 0.80),
)


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


# Internal base: only the equalized legacy PyTorch sweep is registered.
_LEGOGEARS_PYTORCH_BASE = benchmark_profile(
    definition=LEGO_GEARS_V1,
    root="LegoGears_v2",
    name="LegoGearsPyTorchYOLOv4",
    backend="pytorch_yolov4",
    data_path="",
    cfg_out="",
    batch_size=64,
    subdivisions=1,
    learning_rate=0.00261,
    mosaic=0,
    jitter=0.3,
    hue=0.1,
    saturation=1.5,
    exposure=1.5,
    flip=0,
    num_gpus=1,
    pytorch_cfg="cfg/yolov4-tiny.cfg",
)


BENCHMARK_PROFILES = {
    "LegoGearsDarknetBenchmark": benchmark_profile(
        definition=LEGO_GEARS_V1,
        root="/workspace/LegoGears_v2",
        name="LegoGearsDarknetBenchmark",
        backend="darknet",
        data_path="/workspace/LegoGears_v2/LegoGears.data",
        cfg_out="/workspace/LegoGears_v2/LegoGears.cfg",
        batch_size=64,
        subdivisions=1,
        learning_rate=0.00261,
        templates=("yolov4-tiny", "yolov7-tiny"),
        sweep_keys=("templates", "num_gpus"),
        sweep_values={"num_gpus": (1,)},
    ),
    "LegoGearsUltraBenchmark": benchmark_profile(
        definition=LEGO_GEARS_V1,
        root="LegoGears_v2",
        name="LegoGearsUltraBenchmark",
        backend="ultralytics",
        data_path="",
        cfg_out="",
        batch_size=64,
        subdivisions=1,
        learning_rate=0.00261,
        templates=(),
        sweep_keys=("num_gpus", "ultra_model"),
        sweep_values={"num_gpus": (1,), "ultra_model": ("yolo11n.pt", "yolo11s.pt", "yolo26n.pt", "yolo26s.pt")},
        ultra_data="",
        ultra_model="yolo11n.pt",
    ),
}


LEGACY_SWEEP_PROFILES = {
    "LegoGearsDarknet": legacy_variant(
        BENCHMARK_PROFILES["LegoGearsDarknetBenchmark"], "LegoGearsDarknet",
        val_fracs=LEGO_GEARS_V1.policy.validation_fractions,
        sweep_keys=("templates", "val_fracs", "num_gpus"),
    ),
    "LegoGearsUltra": legacy_variant(
        BENCHMARK_PROFILES["LegoGearsUltraBenchmark"], "LegoGearsUltra",
        val_fracs=LEGO_GEARS_V1.policy.validation_fractions,
        sweep_keys=("val_fracs", "num_gpus", "ultra_model"),
    ),
    "LegoGearsPyTorchYOLOv4": legacy_variant(
        _LEGOGEARS_PYTORCH_BASE, "LegoGearsPyTorchYOLOv4",
        # The 80% validation split leaves only 18 images for training, fewer
        # than PyTorch-YOLOv4's 64-image micro-batch.  The supported sweep
        # therefore covers the three comparable split sizes.
        val_fracs=(0.10, 0.15, 0.20),
        sweep_keys=("val_fracs",),
    ),
}
