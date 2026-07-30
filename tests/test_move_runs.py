"""Validation rules used by the run-artifact mover."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


MOVE_RUNS_PATH = Path(__file__).parents[1] / "tools" / "move_runs.py"
SPEC = importlib.util.spec_from_file_location("move_runs", MOVE_RUNS_PATH)
move_runs = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(move_runs)


class PyTorchRunValidationTest(unittest.TestCase):
    def test_detects_pytorch_backend_from_benchmark_csv(self):
        self.assertEqual(
            move_runs.infer_backend(Path("outputs/run"), {"Backend": "pytorch_yolov4"}),
            "pytorch_yolov4",
        )

    def test_accepts_final_checkpoint_for_any_configured_update_count(self):
        with TemporaryDirectory() as temporary:
            log = Path(temporary) / "training_output.log"
            log.write_text(
                "Training update 137/137 (100.0%)\n"
                "Checkpoint 137 saved !\n",
                encoding="utf-8",
            )
            self.assertTrue(move_runs.is_valid_run(log, "pytorch_yolov4"))

    def test_rejects_an_interval_checkpoint_before_the_final_update(self):
        with TemporaryDirectory() as temporary:
            log = Path(temporary) / "training_output.log"
            log.write_text(
                "Training update 1000/1375 (72.7%)\n"
                "Checkpoint 1000 saved !\n",
                encoding="utf-8",
            )
            self.assertFalse(move_runs.is_valid_run(log, "pytorch_yolov4"))


if __name__ == "__main__":
    unittest.main()
