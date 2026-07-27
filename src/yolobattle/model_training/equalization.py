"""Legacy split-sweep equalization helpers."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple
import json
import math

from yolobattle.model_training.profile_models import TrainProfile


_equalize_cache: Dict[tuple, float] = {}


def _manifest_from_data_path(data_path: str) -> Path:
    return Path(data_path).with_name(Path(data_path).stem + "_split.json")


def read_split_counts_from_data(data_path: str) -> Tuple[int, int]:
    """Return train and validation counts from a generated split manifest."""
    payload = json.loads(_manifest_from_data_path(data_path).read_text(encoding="utf-8"))
    counts = payload.get("counts", {})
    return int(counts.get("train_total", 0)), int(counts.get("valid_total", 0))


def equalize_for_split(
    profile: TrainProfile,
    *,
    data_path: str,
    mode: str = "iterations",
    target_epochs: float | None = None,
) -> TrainProfile:
    """Adjust a legacy sweep so its approximate training exposure is stable."""
    try:
        train_count, _ = read_split_counts_from_data(data_path)
    except Exception:
        train_count = 0
    if train_count <= 0:
        return replace(profile, data_path=data_path)

    if target_epochs is None:
        key = (profile.name, profile.template, profile.color_preset)
        target_epochs = _equalize_cache.setdefault(
            key, max(1.0, (profile.iterations * profile.batch_size) / train_count),
        )
    else:
        try:
            target_epochs = float(target_epochs)
        except Exception:
            target_epochs = 1.0
        if not math.isfinite(target_epochs) or target_epochs <= 0:
            target_epochs = 1.0

    if mode == "iterations":
        iterations = max(100, int(math.ceil(target_epochs * train_count / profile.batch_size)))
        return replace(profile, data_path=data_path, iterations=iterations)
    if mode == "batch":
        subdivisions = max(1, profile.subdivisions)
        raw_batch = target_epochs * train_count / profile.iterations
        batch_size = min(max(int(round(raw_batch / subdivisions) * subdivisions), subdivisions), 1024)
        return replace(profile, data_path=data_path, batch_size=batch_size)
    return replace(profile, data_path=data_path)
