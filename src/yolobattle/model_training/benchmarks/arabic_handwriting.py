"""ArabicHandwriting: dataset, comparison policy, and training profiles.

Edit the policy for geometry/splits/budget and the profiles for YOLO models.
"""
from ..benchmark_definitions import BenchmarkDefinition, DatasetRecipe
from ..benchmark_policy import BenchmarkPolicy
from ..profile_models import benchmark_profile


ARABIC_HANDWRITING_352X256_V1 = BenchmarkPolicy(
    name="arabic_handwriting_352x256_v1",
    width=352,
    height=256,
    split_seed=9001,
    validation_fraction=0.20,
    iterations=4000,
)


ARABIC_HANDWRITING_V1 = BenchmarkDefinition(
    name="arabic_handwriting_v1",
    policy=ARABIC_HANDWRITING_352X256_V1,
    dataset_recipe=DatasetRecipe(
        sets=tuple(),
        classes=5,
        names="handWM6.names",
        prefix="ArabicHandwriting",
        exts=(".jpg", ".png"),
        url="https://g-522bba.342e6a.8540.data.globus.org/myhandWM6.zip",
        sha256="4f07ca513d1afd12c0a319e8909a8c6095ab6849b861dcd7c92c298288736e3b",
        flat_dir="darkmark_image_cache/resize",
        archive_subdir="mnt/lustre/users/nalaas/nn/handWM6",
        class_names=("Alif", "Ta", "Ra", "Sin", "Hea"),
    ),
)


BENCHMARK_PROFILES = {
    "ArabicHandwriting": benchmark_profile(
        definition=ARABIC_HANDWRITING_V1,
        root="/workspace/ArabicHandwriting",
        name="ArabicHandwriting",
        backend="darknet",
        data_path="/workspace/ArabicHandwriting/ArabicHandwriting.data",
        cfg_out="/workspace/ArabicHandwriting/ArabicHandwriting.cfg",
        batch_size=64,
        subdivisions=16,
        learning_rate=0.00261,
        templates=("yolov4",),
        sweep_keys=("templates", "num_gpus"),
        sweep_values={"num_gpus": (1,)},
    ),
}


LEGACY_SWEEP_PROFILES = {}
