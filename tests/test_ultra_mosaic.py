"""Ultralytics mosaic overrides must reach the CLI without duplicates."""

import unittest
from types import SimpleNamespace

from yolobattle.model_training.darknet_ultralytics_translation import build_ultralytics_cmd


class UltraMosaicTest(unittest.TestCase):
    def command(self, *, mosaic=None, color_preset=None):
        profile = SimpleNamespace(
            width=416, height=416, ultra_imgsz=None, epochs=20,
            ultra_data="/tmp/dataset.yaml", ultra_model="yolo11n.yaml",
            batch_size=8, learning_rate=0.001, training_seed=9001,
            color_preset=color_preset, fliplr=None, flipud=None,
            ultra_mosaic=mosaic,
        )
        return build_ultralytics_cmd(profile=profile, device_indices=[0], run_dir="/tmp/run")

    def test_default_leaves_mosaic_to_ultralytics(self):
        self.assertNotIn("mosaic=", self.command())

    def test_explicit_zero_disables_mosaic(self):
        self.assertEqual(self.command(mosaic=0).count("mosaic=0.0"), 1)

    def test_explicit_value_overrides_color_preset(self):
        command = self.command(mosaic=0, color_preset="light")
        self.assertEqual(command.count("mosaic="), 1)
        self.assertIn("mosaic=0.0", command)
        self.assertIn("hsv_s=0.1", command)

    def test_color_preset_keeps_its_mosaic_without_override(self):
        self.assertIn("mosaic=0.5", self.command(color_preset="light"))

    def test_rejects_out_of_range_probability(self):
        with self.assertRaisesRegex(ValueError, "mosaic must be between 0 and 1"):
            self.command(mosaic=1.1)


if __name__ == "__main__":
    unittest.main()
