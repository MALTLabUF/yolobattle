import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from yolobattle.model_training.benchmark_policy import (
    CARDS_768X576_V1,
    CUBES_224X160_V1,
    FISHEYE8K_OFFICIAL_1280X1280_V1,
    FISHEYE_TRAFFIC_960X736_V1,
    LEATHER_256X256_V1,
    LEGO_GEARS_224X160_V1,
)
from yolobattle.model_training.benchmark_definitions import (
    CARDS_V1,
    CUBES_V1,
    FISHEYE8K_OFFICIAL_V1,
    FISHEYE_TRAFFIC_LOCAL_V1,
    LEATHER_V1,
    LEGO_GEARS_V1,
)
from yolobattle.model_training.pytorch_yolov4 import (
    apply_yolo_anchor_layout,
    build_command,
    darknet_style_schedule,
    epochs_for_iterations,
)
from yolobattle.model_training.profiles import effective_policy, get_profile


class LegoGearsPolicyTest(unittest.TestCase):
    def test_comparison_profiles_share_one_policy(self):
        profiles = [
            get_profile("LegoGearsDarknetBenchmark"),
            get_profile("LegoGearsUltraBenchmark"),
            get_profile("LegoGearsPyTorchYOLOv4"),
        ]
        expected = LEGO_GEARS_224X160_V1
        for profile in profiles:
            self.assertIs(profile.policy, expected)
            self.assertEqual((profile.width, profile.height), (224, 160))
            self.assertEqual(profile.val_fracs, (0.20,))
            self.assertEqual(profile.dataset.split_seed, 9001)
            self.assertEqual(profile.policy.coco_iou_thresholds, (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95))
            self.assertEqual(profile.policy.fingerprint(), expected.fingerprint())

    def test_training_budget_remains_framework_specific(self):
        profiles = [get_profile(name) for name in (
            "LegoGearsDarknetBenchmark", "LegoGearsUltraBenchmark", "LegoGearsPyTorchYOLOv4",
        )]
        self.assertEqual([profile.iterations for profile in profiles], [7000, 7000, 7000])
        self.assertIsNone(get_profile("LegoGearsPyTorchYOLOv4").epochs)

    def test_legacy_legogears_profiles_keep_the_validation_sweep(self):
        for name in ("LegoGearsDarknet", "LegoGearsUltra"):
            profile = get_profile(name)
            self.assertEqual(profile.val_fracs, LEGO_GEARS_224X160_V1.validation_fractions)
            self.assertIn("val_fracs", profile.sweep_keys)
            self.assertIsNone(profile.policy)

    def test_pytorch_epochs_are_derived_from_iterations(self):
        # 72 examples / batch 64 with drop_last gives 1 update per epoch.
        self.assertEqual(epochs_for_iterations(
            iterations=7000, batch_size=64, subdivisions=1, train_examples=72,
        ), 7000)

    def test_pytorch_command_passes_the_profile_learning_rate(self):
        profile = replace(
            get_profile("LegoGearsPyTorchYOLOv4"), epochs=7000, early_stopping_patience=0,
        )
        with patch.dict("os.environ", {"PYTORCH_YOLOV4_ROOT": "/opt/pytorch-yolov4"}):
            command = build_command(
                profile, Path("model.cfg"), Path("train.txt"), Path("valid.txt"), Path("output"),
            )
        self.assertEqual(command[command.index("-l") + 1], "0.00261")
        self.assertEqual(command[command.index("--burn-in") + 1], "1000")
        self.assertEqual(command[command.index("--steps") + 1:command.index("--steps") + 3], ["5600", "6300"])
        self.assertEqual(command[command.index("--scales") + 1:command.index("--scales") + 3], ["0.1", "0.1"])
        self.assertEqual(command[command.index("--batch") + 1], "64")
        self.assertEqual(command[command.index("--jitter") + 1], "0.3")
        self.assertEqual(command[command.index("--flip") + 1], "0")
        self.assertEqual(command[command.index("--eval-interval") + 1], "100")
        self.assertEqual(command[command.index("--early-stopping-patience") + 1], "0")

    def test_pytorch_schedule_matches_the_darknet_iteration_schedule(self):
        self.assertEqual(darknet_style_schedule(7000), (1000, (5600, 6300), (0.1, 0.1)))
        self.assertEqual(darknet_style_schedule(8), (5, (6, 7), (0.1, 0.1)))

    def test_pytorch_cfg_applies_the_supplied_darknet_anchor_layout(self):
        with TemporaryDirectory() as temp_dir:
            cfg = Path(temp_dir) / "model.cfg"
            cfg.write_text(
                "[net]\nwidth=224\n\n[convolutional]\nfilters=30\n\n[yolo]\n"
                "mask=3,4,5\nanchors=10,14, 23,27, 37,58, 81,82, 135,169, 344,319\nnum=6\n\n"
                "[convolutional]\nfilters=30\n\n[yolo]\n"
                "mask=1,2,3\nanchors=10,14, 23,27, 37,58, 81,82, 135,169, 344,319\nnum=6\n",
                encoding="utf-8",
            )
            apply_yolo_anchor_layout(
                cfg,
                anchors=((8, 8), (10, 10), (15, 13), (45, 44), (68, 65), (77, 74)),
                masks=((3, 4, 5), (0, 1, 2)),
            )
            patched = cfg.read_text(encoding="utf-8")
        self.assertIn("anchors=8, 8, 10, 10, 15, 13, 45, 44, 68, 65, 77, 74", patched)
        self.assertIn("mask=3,4,5", patched)
        self.assertIn("mask=0,1,2", patched)

    def test_other_framework_pairs_share_their_canonical_policy(self):
        cases = (
            (
                LEATHER_256X256_V1,
                ("LeatherDarknetBenchmark", "LeatherUltraBenchmark"),
                (256, 256), 0.20,
            ),
            (
                FISHEYE_TRAFFIC_960X736_V1,
                ("FisheyeTrafficDarknetBenchmark", "FisheyeTrafficUltraBenchmark"),
                (960, 736), 0.10,
            ),
            (
                FISHEYE8K_OFFICIAL_1280X1280_V1,
                ("FishEye8KDarknetBenchmark", "FishEye8KUltraBenchmark"),
                (1280, 1280), 0.30,
            ),
            (
                CUBES_224X160_V1,
                ("CubesDarknetBenchmark", "CubesUltraBenchmark"),
                (224, 160), 0.20,
            ),
            (
                CARDS_768X576_V1,
                ("CardsDarknet", "CardsUltra"),
                (768, 576), 0.20,
            ),
        )
        for policy, names, geometry, validation_fraction in cases:
            profiles = [get_profile(name) for name in names]
            for profile in profiles:
                self.assertIs(profile.policy, policy)
                self.assertEqual((profile.width, profile.height), geometry)
                self.assertEqual(profile.iterations, policy.iterations)
                self.assertEqual(profile.val_fracs, (validation_fraction,))
                self.assertEqual(profile.dataset.split_seed, policy.split_seed)
                self.assertEqual(profile.policy.fingerprint(), policy.fingerprint())

    def test_canonical_profiles_resolve_dataset_from_one_benchmark_definition(self):
        cases = (
            (LEGO_GEARS_V1, ("LegoGearsDarknetBenchmark", "LegoGearsUltraBenchmark", "LegoGearsPyTorchYOLOv4")),
            (LEATHER_V1, ("LeatherDarknetBenchmark", "LeatherUltraBenchmark")),
            (FISHEYE_TRAFFIC_LOCAL_V1, ("FisheyeTrafficDarknetBenchmark", "FisheyeTrafficUltraBenchmark")),
            (FISHEYE8K_OFFICIAL_V1, ("FishEye8KDarknetBenchmark", "FishEye8KUltraBenchmark")),
            (CUBES_V1, ("CubesDarknetBenchmark", "CubesUltraBenchmark")),
            (CARDS_V1, ("CardsDarknet", "CardsUltra")),
        )
        for definition, names in cases:
            for name in names:
                profile = get_profile(name)
                self.assertIs(profile.policy, definition.policy)
                self.assertEqual(profile.dataset, definition.dataset_at(profile.dataset.root))
                self.assertEqual(profile.benchmark.fingerprint(), definition.fingerprint())

    def test_definition_fingerprint_excludes_runtime_mount_path(self):
        fingerprint = LEATHER_V1.fingerprint()
        self.assertEqual(LEATHER_V1.dataset_at("/workspace/leather").root, "/workspace/leather")
        self.assertEqual(LEATHER_V1.dataset_at("/mnt/shared/leather").root, "/mnt/shared/leather")
        self.assertEqual(LEATHER_V1.fingerprint(), fingerprint)
        changed_recipe = replace(LEATHER_V1.dataset_recipe, classes=6)
        self.assertNotEqual(replace(LEATHER_V1, dataset_recipe=changed_recipe).fingerprint(), fingerprint)

    def test_effective_policy_records_an_iteration_override(self):
        overridden = replace(get_profile("LegoGearsPyTorchYOLOv4"), iterations=8)
        self.assertEqual(effective_policy(overridden).iterations, 8)

    def test_fisheye8k_policy_records_its_official_split(self):
        self.assertEqual(FISHEYE8K_OFFICIAL_1280X1280_V1.split_strategy, "official")
        for name in ("FishEye8KDarknetBenchmark", "FishEye8KUltraBenchmark"):
            dataset = get_profile(name).dataset
            self.assertEqual(dataset.predefined_train_dir, "train/images")
            self.assertEqual(dataset.predefined_valid_dir, "test/images")

    def test_legacy_cubes_profiles_use_the_policy_validation_sweep(self):
        for name in ("CubesDarknet", "CubesUltra"):
            profile = get_profile(name)
            self.assertEqual(profile.val_fracs, CUBES_224X160_V1.validation_fractions)
            self.assertIn("val_fracs", profile.sweep_keys)


if __name__ == "__main__":
    unittest.main()
