"""Command-line Darknet profiles using an existing train/validation split."""
from __future__ import annotations

import math
import re
import shutil
from dataclasses import replace
from pathlib import Path

from .cfg_maker import TEMPLATE_URLS
from .profile_models import TrainProfile


def read_data(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    missing = [key for key in ("classes", "train", "valid", "names") if not values.get(key)]
    if missing:
        raise ValueError(f"{path}: missing .data entries: {', '.join(missing)}")
    try:
        classes = int(values["classes"])
    except ValueError as exc:
        raise ValueError(f"{path}: classes must be a positive integer") from exc
    if classes <= 0:
        raise ValueError(f"{path}: classes must be a positive integer")
    return values


def custom_darknet_profile(
    *, name: str, data_path: str | None, cfg_path: str | None,
    template: str | None, width: int | None, height: int | None,
    batch_size: int | None, subdivisions: int | None,
    iterations: int | None, learning_rate: float | None,
    weights: str | None = None, path_maps: list[str] | None = None,
) -> TrainProfile:
    """Validate a supplied cfg or explicit template settings, without downloads."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
        raise ValueError("--custom-profile must be a name using letters, numbers, underscores, dots or hyphens")
    if not data_path:
        raise ValueError("--custom-profile requires --data-path")
    data = Path(data_path).expanduser().resolve()
    if not data.is_file():
        raise ValueError(f".data file not found inside this environment: {data}")
    settings = dict(width=width, height=height, batch_size=batch_size,
                    subdivisions=subdivisions, iterations=iterations, learning_rate=learning_rate)
    if cfg_path:
        if template or any(value is not None for value in settings.values()):
            raise ValueError("--cfg-path uses the settings in that cfg; omit --template and training-setting overrides")
        cfg = Path(cfg_path).expanduser().resolve()
        if not cfg.is_file():
            raise ValueError(f".cfg file not found inside this environment: {cfg}")
        # The CLI name is the only distinction from folder-based discovery.
        # Keep cfg validation, relocation, snapshots and evaluation identical.
        from .adhoc import load_adhoc_profile
        return replace(load_adhoc_profile(str(data.parent), cfg=str(cfg), data=str(data),
                                          weights=weights, path_maps=path_maps), name=name)
    if weights or path_maps:
        raise ValueError("--weights and --path-map require --cfg-path for a custom profile")
    if template not in TEMPLATE_URLS:
        raise ValueError("--custom-profile requires --cfg-path or --template with a supported Darknet template")
    read_data(data)
    missing = ["--" + key.replace("_", "-") for key, value in settings.items() if value is None]
    if missing:
        raise ValueError("Template-based custom profiles require " + ", ".join(missing))
    for key, value in settings.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if settings["batch_size"] % settings["subdivisions"]:
        raise ValueError("batch_size must be divisible by subdivisions")
    return TrainProfile(
        name=name, backend="darknet", data_path=str(data), cfg_out="",
        template=template, val_fracs=(), custom_data=True,
        **settings,
    )


def _source_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def stage_custom_data(profile: TrainProfile, output_dir: Path) -> TrainProfile:
    """Keep the supplied split; make run-local files with absolute input paths."""
    source = Path(profile.data_path)
    values = read_data(source)
    names = _source_path(values["names"], source.parent)
    if not names.is_file():
        raise ValueError(f"Names file not found inside the container: {names}")
    class_names = [line.strip() for line in names.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(class_names) != int(values["classes"]):
        raise ValueError(f"{names}: {len(class_names)} names, but .data declares {values['classes']} classes")

    for key, filename in (("train", "train.txt"), ("valid", "valid.txt")):
        listing = _source_path(values[key], source.parent)
        if not listing.is_file():
            raise ValueError(f"{key} list not found inside the container: {listing}")
        images = [_source_path(line.strip(), listing.parent)
                  for line in listing.read_text(encoding="utf-8-sig").splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
        if not images:
            raise ValueError(f"{key} list is empty: {listing}")
        missing = next((image for image in images if not image.is_file()), None)
        if missing is not None:
            raise ValueError(f"Image from {listing} is not visible inside the container: {missing}")
        target = output_dir / filename
        target.write_text("\n".join(map(str, images)) + "\n", encoding="utf-8")
        values[key] = str(target)

    names_target = output_dir / "classes.names"
    names_target.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    values["names"] = str(names_target)
    backup = output_dir / "weights"
    backup.mkdir(exist_ok=True)
    values["backup"] = str(backup)
    for key in ("map", "test"):
        if values.get(key):
            values[key] = str(_source_path(values[key], source.parent))
    data_target = output_dir / (source.stem + ".data")
    data_target.write_text("".join(f"{key} = {value}\n" for key, value in values.items()), encoding="utf-8")
    cfg_target = output_dir / (Path(profile.cfg_source).name if profile.cfg_source else source.stem + ".cfg")
    if profile.cfg_source:
        shutil.copy2(profile.cfg_source, cfg_target)
    return replace(profile, data_path=str(data_target), cfg_out=str(cfg_target))
