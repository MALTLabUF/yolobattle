"""Checkpoint selection for the PyTorch-YOLOv4 backend."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from yolobattle.model_training.backends import _tianxiaomo_checkpoint


class TianxiaomoCheckpointTest(unittest.TestCase):
    def test_final_selector_uses_the_last_update_checkpoint(self):
        with TemporaryDirectory() as temporary:
            checkpoints = Path(temporary) / "checkpoints"
            checkpoints.mkdir()
            fallback = checkpoints / "Yolov4_epoch7000.pth"
            fallback.touch()
            best = checkpoints / "Yolov4_best.pth"
            best.touch()
            self.assertEqual(
                _tianxiaomo_checkpoint(Path(temporary), selector="final"), fallback,
            )

    def test_backend_default_prefers_validated_best_checkpoint(self):
        with TemporaryDirectory() as temporary:
            checkpoints = Path(temporary) / "checkpoints"
            checkpoints.mkdir()
            final = checkpoints / "Yolov4_epoch7000.pth"
            final.touch()
            best = checkpoints / "Yolov4_best.pth"
            best.touch()
            self.assertEqual(
                _tianxiaomo_checkpoint(Path(temporary), selector="backend_default"), best,
            )

    def test_final_selector_uses_highest_update_not_file_timestamp(self):
        with TemporaryDirectory() as temporary:
            checkpoints = Path(temporary) / "checkpoints"
            checkpoints.mkdir()
            late = checkpoints / "Yolov4_epoch7000.pth"
            late.touch()
            early = checkpoints / "Yolov4_epoch1000.pth"
            early.touch()
            self.assertEqual(
                _tianxiaomo_checkpoint(Path(temporary), selector="final"), late,
            )


if __name__ == "__main__":
    unittest.main()
