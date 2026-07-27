"""Framework adapters used by the shared YoloBattle benchmark runner.

Each adapter owns only framework-specific preparation, native-log parsing, and
COCO detection export.  Dataset-independent benchmark reporting remains in
``train.py``.
"""
from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import yaml

from yolobattle.model_training.cfg_maker import generate_cfg_file
from yolobattle.model_training.darknet_ultralytics_translation import build_ultralytics_cmd
from yolobattle.model_training.evaluators_darknet import parse_darknet_summary
from yolobattle.model_training.evaluators_ultra import (
    find_ultra_results_csv,
    parse_ultra_final_val,
    parse_ultra_map,
)
from yolobattle.model_training.export_coco_dets import (
    export_darknet_detections,
    export_tianxiaomo_detections,
    export_ultra_detections,
)
from yolobattle.model_training.profile_models import TrainProfile


@dataclass(frozen=True)
class NativeMetrics:
    map_last_pct: float | None = None
    map_best_pct: float | None = None
    map_iou: float | None = None
    map_points: int | None = None
    best_iter: int | None = None
    conf_thresh: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


class Backend(Protocol):
    name: str

    def prepare(self, profile: TrainProfile, *, template: str | None, output_dir: Path,
                gpu_indices: list[int], gpus_str: str) -> tuple[TrainProfile, str]: ...
    def native_metrics(self, profile: TrainProfile, output_dir: Path) -> NativeMetrics: ...
    def counts(self, profile: TrainProfile, output_dir: Path) -> tuple[int, int, float | None]: ...
    def export_coco(self, profile: TrainProfile, *, output_dir: Path, gt_json: str,
                    det_json: str, valid_list: str, threshold: float,
                    gpu_indices: list[int]) -> None: ...
    def model_label(self, profile: TrainProfile, template: str | None) -> str: ...
    def finalize(self, profile: TrainProfile, output_dir: Path) -> None: ...


def _data_values(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path or not Path(path).is_file():
        return values
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _split_counts(data_path: str) -> tuple[int, int]:
    try:
        path = Path(data_path)
        counts = __import__("json").loads(path.with_name(path.stem + "_split.json").read_text())["counts"]
        return int(counts.get("train_total", 0)), int(counts.get("valid_total", 0))
    except Exception:
        return 0, 0


def _copy_lists(data_path: str, output_dir: Path) -> None:
    for key, destination in (("train", "train.txt"), ("valid", "valid.txt")):
        source = _data_values(data_path).get(key)
        if source and Path(source).is_file():
            shutil.copy2(source, output_dir / destination)


def _darknet_binary() -> str:
    if "APPTAINER_ENVIRONMENT" in os.environ:
        return "/host_workspace/darknet/build/src-cli/darknet"
    if os.path.exists("/.dockerenv"):
        return "/workspace/darknet/build/src-cli/darknet"
    return "darknet"


def _checkpoint_selector(profile) -> str:
    selector = profile.policy.checkpoint_selector if profile.policy else "backend_default"
    if selector not in {"final", "backend_default"}:
        raise ValueError(f"Unsupported checkpoint selector: {selector}")
    return selector


class DarknetBackend:
    name = "darknet"

    def prepare(self, profile, *, template, output_dir, gpu_indices, gpus_str):
        if not template:
            raise ValueError("Darknet requires a template")
        generate_cfg_file(template=template, data_path=profile.data_path, out_path=profile.cfg_out,
                          width=profile.width, height=profile.height, batch_size=profile.batch_size,
                          subdivisions=profile.subdivisions, iterations=profile.iterations,
                          learning_rate=profile.learning_rate, anchor_clusters=None,
                          color_preset=profile.color_preset,
                          random_multiscale=1 if template == "yolov7" else None)
        shutil.copy2(profile.cfg_out, output_dir / Path(profile.cfg_out).name)
        _copy_lists(profile.data_path, output_dir)
        extras = []
        if profile.map_thresh is not None: extras += ["-thresh", f"{profile.map_thresh:.2f}"]
        if profile.iou_thresh is not None: extras += ["-iou_thresh", f"{profile.iou_thresh:.2f}"]
        extras += ["-points", str(profile.map_points or 101)]
        darknet = _darknet_binary()
        command = (f"{darknet} detector -map {' '.join(extras)} -dont_show -nocolor "
                   + (f"-gpus {gpus_str} " if gpus_str else "")
                   + f"train {profile.data_path} {profile.cfg_out} 2>&1 | tee training_output.log")
        return replace(profile, map_points=profile.map_points or 101), command

    def native_metrics(self, profile, output_dir):
        summary = parse_darknet_summary(str(output_dir / "training_output.log"))
        return NativeMetrics(summary.get("last_map_pct"), summary.get("best_map_pct"),
                             summary.get("map_iou"), profile.map_points or 101,
                             summary.get("best_iter"), summary.get("conf_thresh_eval"),
                             summary.get("prec"), summary.get("rec"), summary.get("f1"))

    def counts(self, profile, output_dir):
        train, valid = _split_counts(profile.data_path)
        return train, valid, ((profile.iterations * profile.batch_size / train) if train else None)

    def export_coco(self, profile, *, output_dir, gt_json, det_json, valid_list, threshold, gpu_indices):
        selector = _checkpoint_selector(profile)
        cfg = Path(profile.cfg_out)
        split_dir = Path(os.environ.get("WRITABLE_BASE", "/workspace/.cache/splits"))
        final_candidates = [
            split_dir / f"{cfg.stem}_final.weights", split_dir / "final.weights",
            cfg.with_name(f"{cfg.stem}_final.weights"), cfg.with_name("final.weights"),
        ]
        fallback_candidates = [
            split_dir / f"{cfg.stem}_last.weights", split_dir / "last.weights",
            cfg.with_name(f"{cfg.stem}_last.weights"), cfg.with_name("last.weights"),
        ]
        best_candidates = [
            cfg.with_name(f"{cfg.stem}_best.weights"), cfg.with_name("best.weights"),
            split_dir / f"{cfg.stem}_best.weights", split_dir / "best.weights",
        ]
        candidates = final_candidates + fallback_candidates
        if selector == "backend_default":
            candidates = best_candidates + candidates
        weights = next((p for p in candidates if p.is_file()), None)
        if weights is None:
            raise RuntimeError("Darknet training produced no weights")
        darknet = _darknet_binary()
        export_darknet_detections(darknet_bin=darknet, data_path=profile.data_path,
            cfg_path=profile.cfg_out, weights_path=str(weights), ann_json=gt_json, out_json=det_json,
            images_txt=valid_list, thresh=threshold, letter_box=False, save_vis=True, vis_dir=str(output_dir))

    def model_label(self, profile, template): return template or "darknet"
    def finalize(self, profile, output_dir):
        cfg = Path(profile.cfg_out)
        split_dir = Path(os.environ.get("WRITABLE_BASE", "/workspace/.cache/splits"))
        for source in (
            split_dir / f"{cfg.stem}_last.weights", split_dir / "last.weights",
            cfg.with_name(f"{cfg.stem}_last.weights"), cfg.with_name("last.weights"),
        ):
            if source.is_file():
                shutil.copy2(source, output_dir / source.name)


class UltralyticsBackend:
    name = "ultralytics"

    def prepare(self, profile, *, template, output_dir, gpu_indices, gpus_str):
        ypath = Path(profile.ultra_data)
        shutil.copy2(ypath, output_dir / ypath.name)
        document = yaml.safe_load(ypath.read_text(encoding="utf-8"))
        for key, destination in (("train", "train.txt"), ("val", "valid.txt")):
            source = document.get(key)
            if isinstance(source, str) and Path(source).is_file(): shutil.copy2(source, output_dir / destination)
        return profile, build_ultralytics_cmd(profile=profile, device_indices=gpu_indices, run_dir=str(output_dir))

    def native_metrics(self, profile, output_dir):
        ap50, ap95 = parse_ultra_final_val(str(output_dir / "training_output.log"))
        if ap50 is None or ap95 is None:
            results = find_ultra_results_csv(str(output_dir))
            ap50, ap95 = parse_ultra_map(results) if results else (None, None)
        return NativeMetrics(map_last_pct=ap50, map_iou=0.50, map_points=101)

    def counts(self, profile, output_dir):
        def count(name):
            path = output_dir / name
            return sum(1 for line in path.read_text().splitlines() if line.strip()) if path.is_file() else 0
        return count("train.txt"), count("valid.txt"), profile.epochs

    def export_coco(self, profile, *, output_dir, gt_json, det_json, valid_list, threshold, gpu_indices):
        selector = _checkpoint_selector(profile)
        nms_iou = profile.policy.export_nms_iou if profile.policy else 0.45
        checkpoint = "last.pt" if selector == "final" else "best.pt"
        export_ultra_detections(weights=str(output_dir / "train" / "weights" / checkpoint),
            ann_json=gt_json, out_json=det_json, images_txt=valid_list, conf=threshold, iou=nms_iou,
            imgsz=(profile.height, profile.width), device=gpu_indices, batch=2, save_vis=True, vis_dir=str(output_dir))

    def model_label(self, profile, template): return profile.ultra_model
    def finalize(self, profile, output_dir): pass


class TianxiaomoBackend:
    name = "pytorch_yolov4"

    def prepare(self, profile, *, template, output_dir, gpu_indices, gpus_str):
        from yolobattle.model_training.pytorch_yolov4 import build_command, prepare
        profile, cfg, train_labels, valid_labels = prepare(profile, output_dir)
        _copy_lists(profile.data_path, output_dir)
        command = shlex.join(build_command(profile, cfg, train_labels, valid_labels, output_dir))
        return profile, command + " 2>&1 | tee " + shlex.quote(str(output_dir / "training_output.log"))

    def native_metrics(self, profile, output_dir): return NativeMetrics(map_iou=0.50, map_points=101)
    def counts(self, profile, output_dir):
        train, valid = _split_counts(profile.data_path)
        return train, valid, profile.epochs

    def export_coco(self, profile, *, output_dir, gt_json, det_json, valid_list, threshold, gpu_indices):
        _checkpoint_selector(profile)
        checkpoints = sorted((output_dir / "checkpoints").glob("Yolov4_epoch*.pth"), key=lambda p: p.stat().st_mtime)
        if not checkpoints: raise RuntimeError("Tianxiaomo training produced no checkpoint")
        export_tianxiaomo_detections(repo_path=os.environ["PYTORCH_YOLOV4_ROOT"], checkpoint=str(checkpoints[-1]),
            cfg_path=str(output_dir / f"{Path(profile.data_path).stem}.pytorch.cfg"), ann_json=gt_json,
            out_json=det_json, images_txt=valid_list, width=profile.width, height=profile.height,
            conf=threshold, iou=profile.policy.export_nms_iou if profile.policy else 0.45, device="cuda")

    def model_label(self, profile, template): return profile.pytorch_cfg
    def finalize(self, profile, output_dir): pass


BACKENDS: dict[str, Backend] = {backend.name: backend for backend in (DarknetBackend(), UltralyticsBackend(), TianxiaomoBackend())}

def get_backend(name: str) -> Backend:
    try: return BACKENDS[name]
    except KeyError as exc: raise ValueError(f"Unsupported backend: {name}") from exc
