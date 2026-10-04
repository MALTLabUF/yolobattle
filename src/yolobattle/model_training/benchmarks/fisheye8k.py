"""FishEye8K: dataset, comparison policy, and training profiles.

Edit the policy for geometry/splits/budget and the profiles for YOLO models.
"""
from ..benchmark_definitions import BenchmarkDefinition, DatasetRecipe
from ..benchmark_policy import BenchmarkPolicy
from ..profile_models import benchmark_profile, legacy_variant


FISHEYE8K_OFFICIAL_1280X1280_V1 = BenchmarkPolicy(
    name="fisheye8k_official_1280x1280_v1",
    width=1280,
    height=1280,
    split_seed=9001,
    validation_fraction=0.30,
    iterations=8000,
    split_strategy="official",
)


FISHEYE8K_OFFICIAL_V1 = BenchmarkDefinition(
    name="fisheye8k_official_v1",
    policy=FISHEYE8K_OFFICIAL_1280X1280_V1,
    dataset_recipe=DatasetRecipe(
        sets=tuple(),
        classes=5,
        names="FishEye8K.names",
        prefix="FishEye8K_official",
        exts=(".jpg", ".jpeg", ".png"),
        require_existing=True,
        predefined_train_dir="train/images",
        predefined_valid_dir="test/images",
        class_names=("Bus", "Bike", "Car", "Pedestrian", "Truck"),
        annotation_format="yolo",
    ),
)


BENCHMARK_PROFILES = {
    "FishEye8KDarknetBenchmark": benchmark_profile(
        definition=FISHEYE8K_OFFICIAL_V1,
        root="/blue/ranka/j.fleischer/Fisheye8K_all_including_trainandtest",
        name="FishEye8KDarknetBenchmark",
        backend="darknet",
        data_path="/workspace/.cache/splits/FishEye8K_official.data",
        cfg_out="/workspace/FishEye8K.cfg",
        batch_size=64,
        subdivisions=16,
        learning_rate=0.00261,
        templates=("yolov4", "yolov7"),
        sweep_keys=("templates", "num_gpus"),
        sweep_values={"num_gpus": (1,)},
    ),
    "FishEye8KUltraBenchmark": benchmark_profile(
        definition=FISHEYE8K_OFFICIAL_V1,
        root="/blue/ranka/j.fleischer/Fisheye8K_all_including_trainandtest",
        name="FishEye8KUltraBenchmark",
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
    "FishEye8KDarknet": legacy_variant(BENCHMARK_PROFILES["FishEye8KDarknetBenchmark"], "FishEye8KDarknet"),
    "FishEye8KUltralytics": legacy_variant(BENCHMARK_PROFILES["FishEye8KUltraBenchmark"], "FishEye8KUltralytics"),
}
