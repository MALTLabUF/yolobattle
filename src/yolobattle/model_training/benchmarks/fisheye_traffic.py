"""FisheyeTraffic: dataset, comparison policy, and training profiles.

Edit the policy for geometry/splits/budget and the profiles for YOLO models.
"""
from dataclasses import replace

from ..benchmark_definitions import BenchmarkDefinition, DatasetRecipe
from ..benchmark_policy import BenchmarkPolicy
from ..profile_models import benchmark_profile, legacy_variant


FISHEYE_TRAFFIC_960X736_V1 = BenchmarkPolicy(
    name="fisheye_traffic_960x736_v1",
    width=960,
    height=736,
    split_seed=9001,
    validation_fraction=0.10,
    iterations=8000,
)


FISHEYE_TRAFFIC_LOCAL_V1 = BenchmarkDefinition(
    name="fisheye_traffic_local_v1",
    policy=FISHEYE_TRAFFIC_960X736_V1,
    dataset_recipe=DatasetRecipe(
        sets=tuple(),
        classes=5,
        names="obj.names",
        prefix="combined",
        exts=(".jpg", ".png"),
        require_existing=True,
        flat_dir="darkmark_image_cache/resize",
    ),
)


BENCHMARK_PROFILES = {
    "FisheyeTrafficDarknetBenchmark": benchmark_profile(
        definition=FISHEYE_TRAFFIC_LOCAL_V1,
        root="/blue/ranka/j.fleischer/annotation_data",
        name="FisheyeTrafficDarknetBenchmark",
        backend="darknet",
        data_path="/host_workspace/combined.data",
        cfg_out="/host_workspace/combined.cfg",
        batch_size=64,
        subdivisions=16,
        learning_rate=0.00261,
        templates=("yolov4", "yolov7"),
        sweep_keys=("templates", "num_gpus"),
        sweep_values={"num_gpus": (1,)},
    ),
    "FisheyeTrafficUltraBenchmark": benchmark_profile(
        definition=FISHEYE_TRAFFIC_LOCAL_V1,
        root="/blue/ranka/j.fleischer/annotation_data",
        name="FisheyeTrafficUltraBenchmark",
        backend="ultralytics",
        data_path="",
        cfg_out="",
        batch_size=64,
        subdivisions=16,
        learning_rate=0.00261,
        templates=(),
        sweep_keys=("num_gpus", "ultra_model"),
        sweep_values={"num_gpus": (1,)},
        ultra_data="",
        ultra_model="yolo11n.pt",
    ),
}


LEGACY_SWEEP_PROFILES = {
    "FisheyeTrafficDarknetLocal": legacy_variant(
        BENCHMARK_PROFILES["FisheyeTrafficDarknetBenchmark"], "FisheyeTrafficDarknetLocal",
        sweep_keys=("templates",),
        sweep_values={},
    ),
    "FisheyeTrafficDarknetLocalLRSweep": legacy_variant(
        BENCHMARK_PROFILES["FisheyeTrafficDarknetBenchmark"], "FisheyeTrafficDarknetLocalLRSweep",
        learning_rate=0.0013,
        sweep_keys=("templates", "learning_rate"),
        sweep_values={"learning_rate": (0.0010, 0.0013, 0.0020, 0.00261, 0.0040)},
    ),
    "FisheyeTrafficDarknetLocalJPG": legacy_variant(
        BENCHMARK_PROFILES["FisheyeTrafficDarknetBenchmark"], "FisheyeTrafficDarknetLocalJPG",
        sweep_keys=("templates",),
        sweep_values={},
        dataset=replace(
            FISHEYE_TRAFFIC_LOCAL_V1.dataset_at("/blue/ranka/ibraheem.qureshi/images"),
            names="/blue/ranka/j.fleischer/annotation_data/obj.names",
            prefix="combined_ibraheem",
            flat_dir=".",
        ),
    ),
    "FisheyeTrafficUltralyticsLocal": legacy_variant(
        BENCHMARK_PROFILES["FisheyeTrafficUltraBenchmark"], "FisheyeTrafficUltralyticsLocal",
        sweep_keys=("num_gpus",),
        sweep_values={"num_gpus": (1,)},
    ),
}
