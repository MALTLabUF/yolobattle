from __future__ import annotations

from pathlib import Path
import shutil
import sys

SRC = Path('artifacts/outputs')
DST = Path('../runs-yolobattle/outputs')

# Phrases that must appear in training_output.log
DARKNET_OK = 'Training iteration has reached max batch limit'
ULTRA_OK = 'epochs completed'


def infer_ok_phrase(run_dir: Path) -> str | None:
    run_path = str(run_dir)
    if 'Darknet' in run_path:
        return DARKNET_OK
    if 'Ultra' in run_path:
        return ULTRA_OK
    return None


def is_valid_run(log_path: Path, ok_phrase: str) -> bool:
    try:
        txt = log_path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return False
    return ok_phrase in txt


def main() -> int:
    if not SRC.is_dir():
        print('No artifacts/outputs directory found.')
        return 1

    moved = 0
    skipped = 0

    for log_path in SRC.rglob('training_output.log'):
        run_dir = log_path.parent
        ok_phrase = infer_ok_phrase(run_dir)
        if ok_phrase is None:
            skipped += 1
            continue

        if not is_valid_run(log_path, ok_phrase):
            skipped += 1
            continue

        rel = run_dir.relative_to(SRC)
        target = DST / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        shutil.copytree(run_dir, target, dirs_exist_ok=True)
        shutil.rmtree(run_dir)
        moved += 1

    print(f'Moved {moved} run(s); skipped {skipped}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
