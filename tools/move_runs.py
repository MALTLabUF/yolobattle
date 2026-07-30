from __future__ import annotations

from pathlib import Path
import csv
import re
import shutil
import sys

SRC = Path("artifacts/outputs")
PROJECT = Path("project")
DST = Path("../runs-yolobattle/outputs")

# Phrases that must appear in training_output.log
DARKNET_OK = 'Training iteration has reached max batch limit'
ULTRA_OK = 'epochs completed'
# Tianxiaomo's PyTorch-YOLOv4 logger emits both of these values at the end of a
# successful run.  Capturing and comparing them keeps this independent of the
# iteration budget selected by a profile.
PYTORCH_FINAL_UPDATE = re.compile(r"Training update\s+(?P<current>\d+)\s*/\s*(?P<total>\d+)")
PYTORCH_CHECKPOINT = re.compile(r"Checkpoint\s+(?P<step>\d+)\s+saved\s*!")


def _read_first_row(run_dir: Path) -> dict[str, str]:
    csv_files = sorted(run_dir.glob("benchmark__*.csv"))
    if not csv_files:
        return {}
    try:
        with csv_files[0].open(newline="") as f:
            reader = csv.DictReader(f)
            return next(reader, {}) or {}
    except Exception:
        return {}


def infer_backend(run_dir: Path, row: dict[str, str]) -> str | None:
    backend = (row.get("Backend") or "").strip().lower()
    if "darknet" in backend:
        return "darknet"
    if "ultralytics" in backend:
        return "ultralytics"
    if "pytorch_yolov4" in backend:
        return "pytorch_yolov4"

    run_path = str(run_dir)
    if "Darknet" in run_path:
        return "darknet"
    if "Ultra" in run_path:
        return "ultralytics"
    if "PyTorchYOLOv4" in run_path:
        return "pytorch_yolov4"
    return None


def infer_ok_phrase(backend: str | None) -> str | None:
    if backend == "darknet":
        return DARKNET_OK
    if backend == "ultralytics":
        return ULTRA_OK
    if backend == "pytorch_yolov4":
        return "pytorch_yolov4"
    return None


def is_valid_run(log_path: Path, ok_phrase: str) -> bool:
    try:
        txt = log_path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return False
    if ok_phrase != "pytorch_yolov4":
        return ok_phrase in txt

    # A checkpoint is also written periodically, so it alone is not proof of
    # completion.  Require a final update (current == configured total) and a
    # checkpoint for that same update; neither value is hard-coded.
    completed_updates = {
        match.group("current")
        for match in PYTORCH_FINAL_UPDATE.finditer(txt)
        if match.group("current") == match.group("total")
    }
    checkpoint_updates = {match.group("step") for match in PYTORCH_CHECKPOINT.finditer(txt)}
    return bool(completed_updates & checkpoint_updates)


def _dest_for_run(root: Path, run_dir: Path, row: dict[str, str]) -> Path | None:
    if root == SRC:
        return DST / run_dir.relative_to(SRC)

    if root == PROJECT:
        profile = (row.get("Profile") or "UnknownProfile").strip()
        yolo = (row.get("YOLO Template") or "").strip().replace(".pt", "")
        if not yolo:
            yolo = run_dir.parent.name
        return DST / profile / yolo / run_dir.name

    return None


def main() -> int:
    if not SRC.is_dir() and not PROJECT.is_dir():
        print("No artifacts/outputs or project directory found.")
        return 1

    moved = 0
    skipped = 0
    collisions = 0

    roots = [r for r in (SRC, PROJECT) if r.is_dir()]
    for root in roots:
        log_paths = list(root.rglob("training_output.log"))
        for log_path in log_paths:
            if not log_path.exists():
                skipped += 1
                continue
            run_dir = log_path.parent

            row = _read_first_row(run_dir)
            backend = infer_backend(run_dir, row)
            ok_phrase = infer_ok_phrase(backend)
            if ok_phrase is None:
                skipped += 1
                continue

            if not is_valid_run(log_path, ok_phrase):
                skipped += 1
                continue

            target = _dest_for_run(root, run_dir, row)
            if target is None:
                skipped += 1
                continue

            if target.exists():
                collisions += 1
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(run_dir, target)
            moved += 1

    print(f"Moved {moved} run(s); skipped {skipped}; collisions {collisions}.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
