"""Leather: dataset, comparison policy, and training profiles.

Edit the policy for geometry/splits/budget and the profiles for YOLO models.
"""
from ..benchmark_definitions import BenchmarkDefinition, DatasetRecipe
from ..benchmark_policy import BenchmarkPolicy
from ..profile_models import benchmark_profile, legacy_variant


LEATHER_256X256_V1 = BenchmarkPolicy(
    name="leather_256x256_v1",
    width=256,
    height=256,
    split_seed=9001,
    validation_fraction=0.20,
    iterations=7000,
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


BENCHMARK_PROFILES = {
    "LeatherDarknetBenchmark": benchmark_profile(
        definition=LEATHER_V1,
        root="/workspace/leather",
        name="LeatherDarknetBenchmark",
        backend="darknet",
        data_path="/workspace/leather/leather.data",
        cfg_out="/workspace/leather/leather.cfg",
        batch_size=64,
        subdivisions=1,
        learning_rate=0.00261,
        templates=("yolov4-tiny", "yolov7-tiny"),
        color_presets=(None,),
        sweep_keys=("templates", "num_gpus"),
        sweep_values={"num_gpus": (1,)},
    ),
    "LeatherUltraBenchmark": benchmark_profile(
        definition=LEATHER_V1,
        root="/workspace/leather",
        name="LeatherUltraBenchmark",
        backend="ultralytics",
        data_path="",
        cfg_out="",
        batch_size=64,
        subdivisions=1,
        learning_rate=0.00261,
        templates=(),
        color_presets=(None,),
        sweep_keys=("num_gpus", "ultra_model"),
        sweep_values={"num_gpus": (1,), "ultra_model": ("yolo11n.pt", "yolo11s.pt")},
        ultra_data="",
        ultra_model="yolo11n.pt",
    ),
}


LEGACY_SWEEP_PROFILES = {
    "LeatherDarknet": legacy_variant(
        BENCHMARK_PROFILES["LeatherDarknetBenchmark"], "LeatherDarknet",
        color_presets=(None, "preserve"),
        tag_color_preset=True,
        sweep_keys=("templates", "color_presets", "num_gpus"),
    ),
    "LeatherUltra": legacy_variant(
        BENCHMARK_PROFILES["LeatherUltraBenchmark"], "LeatherUltra",
        color_presets=(None, "preserve"),
        tag_color_preset=True,
        sweep_keys=("color_presets", "num_gpus", "ultra_model"),
        sweep_values={
            "num_gpus": (1,),
            "ultra_model": ("yolo26n.pt", "yolo26s.pt"),
        },
    ),
}
