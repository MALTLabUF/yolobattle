from __future__ import annotations

from pathlib import Path
import csv
import shutil
import sys

SRC = Path("artifacts/outputs")
PROJECT = Path("project")
DST = Path("../runs-yolobattle/outputs")

# Phrases that must appear in training_output.log
DARKNET_OK = 'Training iteration has reached max batch limit'
ULTRA_OK = 'epochs completed'


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

    run_path = str(run_dir)
    if "Darknet" in run_path:
        return "darknet"
    if "Ultra" in run_path:
        return "ultralytics"
    return None


def infer_ok_phrase(backend: str | None) -> str | None:
    if backend == "darknet":
        return DARKNET_OK
    if backend == "ultralytics":
        return ULTRA_OK
    return None


def is_valid_run(log_path: Path, ok_phrase: str) -> bool:
    try:
        txt = log_path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return False
    return ok_phrase in txt


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
        for log_path in root.rglob("training_output.log"):
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
