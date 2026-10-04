"""Cards: dataset, comparison policy, and training profiles.

Edit the policy for geometry/splits/budget and the profiles for YOLO models.
"""
from ..benchmark_definitions import BenchmarkDefinition, DatasetRecipe
from ..benchmark_policy import BenchmarkPolicy
from ..profile_models import benchmark_profile


CARDS_768X576_V1 = BenchmarkPolicy(
    name="cards_768x576_v1",
    width=768,
    height=576,
    split_seed=9001,
    validation_fraction=0.20,
    iterations=6000,
)


CARDS_V1 = BenchmarkDefinition(
    name="cards_v1",
    policy=CARDS_768X576_V1,
    dataset_recipe=DatasetRecipe(
        sets=tuple(),
        classes=19,
        names="ccr_playing_cards.names",
        prefix="ccr_playing_cards",
        exts=(".jpg", ".png"),
        url="https://g-665dcc.55ba.08cc.data.globus.org/playing_cards.zip",
        sha256="432d6da3a2fbec5d1dadd3278b5c4c21ccbaa2dbcd72e087daf193e9bdaf3cc4",
        flat_dir="darkmark_image_cache/resize",
    ),
)


BENCHMARK_PROFILES = {
    "CardsDarknet": benchmark_profile(
        definition=CARDS_V1,
        root="/workspace/ccr_playing_cards",
        name="CardsDarknet",
        backend="darknet",
        data_path="/workspace/ccr_playing_cards/ccr_playing_cards.data",
        cfg_out="/workspace/ccr_playing_cards/ccr_playing_cards.cfg",
        batch_size=64,
        subdivisions=1,
        learning_rate=0.00261,
        templates=("yolov4-tiny", "yolov7-tiny", "yolov4-tiny-3l"),
        sweep_keys=("templates", "num_gpus"),
        sweep_values={"num_gpus": (1,)},
    ),
    "CardsUltra": benchmark_profile(
        definition=CARDS_V1,
        root="/workspace/ccr_playing_cards",
        name="CardsUltra",
        backend="ultralytics",
        data_path="",
        cfg_out="",
        batch_size=64,
        subdivisions=1,
        learning_rate=0.00261,
        templates=(),
        sweep_keys=("templates", "num_gpus", "ultra_model"),
        sweep_values={"num_gpus": (1,), "ultra_model": ("yolo11n.pt", "yolo11s.pt")},
        ultra_data="",
        ultra_model="yolo11n.pt",
    ),
}


LEGACY_SWEEP_PROFILES = {}
