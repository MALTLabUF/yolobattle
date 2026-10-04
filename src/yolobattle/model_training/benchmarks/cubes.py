"""Cubes: dataset, comparison policy, and training profiles.

Edit the policy for geometry/splits/budget and the profiles for YOLO models.
"""
from ..benchmark_definitions import BenchmarkDefinition, DatasetRecipe
from ..benchmark_policy import BenchmarkPolicy
from ..profile_models import benchmark_profile, legacy_variant


CUBES_224X160_V1 = BenchmarkPolicy(
    name="cubes_224x160_v1",
    width=224,
    height=160,
    split_seed=9001,
    validation_fraction=0.20,
    iterations=7000,
    validation_fractions=(0.10, 0.15, 0.20),
)


CUBES_V1 = BenchmarkDefinition(
    name="cubes_v1",
    policy=CUBES_224X160_V1,
    dataset_recipe=DatasetRecipe(
        sets=tuple(),
        classes=4,
        names="cubes.names",
        prefix="cubes",
        exts=(".jpg", ".png"),
        url="https://g-665dcc.55ba.08cc.data.globus.org/refinedcubes.zip",
        sha256="8764c5086e1cada0b66de5198df11655009315873bc9245fd44741ff6e31f4e0",
        flat_dir="darkmark_image_cache/resize",
    ),
)


BENCHMARK_PROFILES = {
    "CubesDarknetBenchmark": benchmark_profile(
        definition=CUBES_V1,
        root="/workspace/cubes",
        name="CubesDarknetBenchmark",
        backend="darknet",
        data_path="/workspace/cubes/cubes.data",
        cfg_out="/workspace/cubes/cubes.cfg",
        batch_size=64,
        subdivisions=1,
        learning_rate=0.00261,
        templates=("yolov4-tiny", "yolov7-tiny"),
        color_preset="preserve",
        color_presets=("preserve",),
        tag_color_preset=True,
        sweep_keys=("templates", "num_gpus"),
        sweep_values={"num_gpus": (1,)},
    ),
    "CubesUltraBenchmark": benchmark_profile(
        definition=CUBES_V1,
        root="/workspace/cubes",
        name="CubesUltraBenchmark",
        backend="ultralytics",
        data_path="",
        cfg_out="",
        batch_size=64,
        subdivisions=1,
        learning_rate=0.00261,
        templates=(),
        color_preset="preserve",
        color_presets=("preserve",),
        tag_color_preset=True,
        sweep_keys=("num_gpus", "ultra_model"),
        sweep_values={"num_gpus": (1,), "ultra_model": ("yolo11n.pt", "yolo11s.pt")},
        ultra_data="",
        ultra_model="yolo11n.pt",
    ),
}


LEGACY_SWEEP_PROFILES = {
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
