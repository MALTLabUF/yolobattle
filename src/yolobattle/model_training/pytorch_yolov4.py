"""Adapter for the repaired Tianxiaomo PyTorch-YOLOv4 fork."""
from __future__ import annotations

import os
import subprocess
import math
import re
from dataclasses import replace
from pathlib import Path

from yolobattle.model_training.dataset_setup import IMG_EXTS, make_split
from yolobattle.model_training.datasets import ensure_download_once
from yolobattle.model_training.benchmark_definitions import DatasetSpec
from yolobattle.model_training.cfg_maker import calibrate_anchor_layout_from_data
from yolobattle.model_training.profiles import TrainProfile


def _dataset_for_runtime(spec: DatasetSpec) -> DatasetSpec:
    root = Path(spec.root)
    if root.is_absolute():
        return spec
    return replace(spec, root=str((Path(os.environ.get("DATA_ROOT", "/workspace")) / root).resolve()))


def _data_paths(data_file: Path) -> tuple[Path, Path]:
    values = {}
    for line in data_file.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    try:
        return Path(values["train"]), Path(values["valid"])
    except KeyError as exc:
        raise ValueError(f"{data_file} must contain train= and valid= entries") from exc


def epochs_for_iterations(*, iterations: int, batch_size: int, subdivisions: int, train_examples: int) -> int:
    """Convert YoloBattle optimizer updates to Tianxiaomo's whole epochs.

    Its loader uses ``drop_last=True`` and performs one optimizer update per
    ``subdivisions`` micro-batches (including the final partial accumulation).
    """
    micro_batch = batch_size // subdivisions
    if iterations <= 0 or micro_batch <= 0 or train_examples < micro_batch:
        raise ValueError("iterations, batch/subdivisions, and train examples must permit an optimizer step")
    loader_batches = train_examples // micro_batch
    updates_per_epoch = math.ceil(loader_batches / subdivisions)
    return math.ceil(iterations / updates_per_epoch)


def darknet_style_schedule(iterations: int) -> tuple[int, tuple[int, int], tuple[float, float]]:
    """Return the warmup and decay schedule generated for Darknet cfg files."""
    steps = (math.floor(0.80 * iterations), math.floor(0.90 * iterations))
    if steps[0] < 2 or steps[1] <= steps[0]:
        raise ValueError("iterations must provide two ordered decay steps")
    # Darknet's normal 1000-update warmup is retained for real benchmarks.
    # Short explicit iteration overrides (used for smoke tests) get a valid
    # reduced warmup rather than an impossible schedule.
    burn_in = min(1000, steps[0] - 1)
    return burn_in, steps, (0.1, 0.1)


def apply_yolo_anchor_layout(cfg_path: Path, anchors: tuple[tuple[int, int], ...],
                             masks: tuple[tuple[int, ...], ...]) -> None:
    """Patch generated PyTorch cfg heads with the shared Darknet anchor layout."""
    sections = re.split(r"(\n\s*\n)", cfg_path.read_text(encoding="utf-8"))
    yolo_indices = [index for index, section in enumerate(sections) if section.lstrip().startswith("[yolo]")]
    if len(yolo_indices) != len(masks):
        raise ValueError(f"Expected {len(masks)} [yolo] sections in {cfg_path}, found {len(yolo_indices)}")
    anchors_csv = ", ".join(f"{width}, {height}" for width, height in anchors)
    for head_index, yolo_index in enumerate(yolo_indices):
        sections[yolo_index], anchor_count = re.subn(
            r"(?m)^anchors\s*=\s*[^\n]+$", f"anchors={anchors_csv}", sections[yolo_index],
        )
        sections[yolo_index], mask_count = re.subn(
            r"(?m)^mask\s*=\s*[^\n]+$", "mask=" + ",".join(map(str, masks[head_index])), sections[yolo_index],
        )
        sections[yolo_index], num_count = re.subn(
            r"(?m)^num\s*=\s*\d+\s*$", f"num={len(anchors)}", sections[yolo_index],
        )
        if anchor_count != 1 or mask_count != 1 or num_count != 1:
            raise ValueError(f"YOLO head {head_index} in {cfg_path} is missing anchors, mask, or num")
    cfg_path.write_text("".join(sections), encoding="utf-8")


def prepare(profile: TrainProfile, output_dir: Path) -> tuple[TrainProfile, Path, Path, Path]:
    if profile.backend != "pytorch_yolov4" or profile.dataset is None:
        raise ValueError("pytorch_yolov4 requires a profile with a DatasetSpec")
    dataset = _dataset_for_runtime(profile.dataset)
    ensure_download_once(dataset)
    val_frac = profile.policy.validation_fraction if profile.policy else float(profile.val_fracs[0])
    data_path, _ = make_split(
        root=dataset.root, sets=None if dataset.flat_dir else list(dataset.sets),
        classes=dataset.classes, names=dataset.names, prefix=f"{dataset.prefix}_v{int(round(val_frac * 100)):02d}",
        val_frac=val_frac, seed=dataset.split_seed, neg_subdirs=list(dataset.neg_subdirs) or None,
        exts=list(dataset.exts or IMG_EXTS), flat_dir=dataset.flat_dir, legos=dataset.legos,
        predefined_train_dir=dataset.predefined_train_dir, predefined_valid_dir=dataset.predefined_valid_dir,
        class_names=dataset.class_names, out_dir=output_dir,
    )
    data_path = Path(data_path)
    train_list, valid_list = _data_paths(data_path)
    train_examples = sum(1 for line in train_list.read_text(encoding="utf-8").splitlines() if line.strip())
    epochs = epochs_for_iterations(
        iterations=profile.iterations,
        batch_size=profile.batch_size,
        subdivisions=profile.subdivisions,
        train_examples=train_examples,
    )
    root = Path(os.environ["PYTORCH_YOLOV4_ROOT"])
    converter = root / "tools" / "yolo_labels_to_tianxiaomo.py"
    template = root / profile.pytorch_cfg
    model_cfg = output_dir / f"{data_path.stem}.pytorch.cfg"
    train_labels, valid_labels = output_dir / "train.tianxiaomo.txt", output_dir / "valid.tianxiaomo.txt"
    subprocess.run([
        "python", str(converter), str(train_list), str(train_labels), "--cfg-template", str(template),
        "--cfg-output", str(model_cfg), "--classes", str(dataset.classes), "--width", str(profile.width),
        "--height", str(profile.height),
    ], check=True)
    model_sections = re.split(r"(\n\s*\n)", model_cfg.read_text(encoding="utf-8"))
    num_heads = sum(section.lstrip().startswith("[yolo]") for section in model_sections)
    if num_heads == 0:
        raise ValueError(f"{model_cfg} contains no [yolo] detection heads")
    anchor_layout = calibrate_anchor_layout_from_data(
        data_path,
        template_name=Path(profile.pytorch_cfg).stem,
        num_heads=num_heads,
        width=profile.width,
        height=profile.height,
        classes=dataset.classes,
    )
    apply_yolo_anchor_layout(model_cfg, anchor_layout.anchors, anchor_layout.masks)
    subprocess.run(["python", str(converter), str(valid_list), str(valid_labels)], check=True)
    return replace(profile, dataset=dataset, data_path=str(data_path), epochs=epochs), model_cfg, train_labels, valid_labels


def build_command(profile: TrainProfile, model_cfg: Path, train_labels: Path, valid_labels: Path, output_dir: Path) -> list[str]:
    dataset_root = profile.dataset.root if profile.dataset else ""
    train_script = Path(os.environ["PYTORCH_YOLOV4_ROOT"]) / "train.py"
    burn_in, steps, scales = darknet_style_schedule(profile.iterations)
    training_seed = profile.training_seed if profile.training_seed is not None else profile.dataset.split_seed
    return [
        "python", str(train_script), "-g", "0", "--cfg", str(model_cfg),
        "--width", str(profile.width), "--height", str(profile.height), "-classes", str(profile.dataset.classes),
        "-dir", str(dataset_root), "-train_label_path", str(train_labels), "--val-label-path", str(valid_labels),
        "--epochs", str(profile.epochs), "--batch", str(profile.batch_size),
        "--subdivisions", str(profile.subdivisions), "--workers", str(min(4, os.cpu_count() or 1)),
        "-optimizer", "sgd", "-l", str(profile.learning_rate),
        "--burn-in", str(burn_in), "--steps", *(str(step) for step in steps),
        "--scales", *(str(scale) for scale in scales),
        "--seed", str(training_seed),
        "--mosaic", str(profile.mosaic), "--jitter", str(profile.jitter),
        "--hue", str(profile.hue), "--saturation", str(profile.saturation),
        "--exposure", str(profile.exposure), "--flip", str(profile.flip),
        "--eval-interval", "100", "--checkpoint-interval", "1000",
        "--early-stopping-patience", str(profile.early_stopping_patience),
        "--checkpoints", str(output_dir / "checkpoints"), "--log-dir", str(output_dir / "log"),
    ]
