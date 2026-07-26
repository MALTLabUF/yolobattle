#!/usr/bin/env python3
from __future__ import annotations
import os, re, csv, getpass, subprocess, zipfile, unicodedata, argparse, math
from pathlib import Path
from datetime import datetime
from threading import Thread, Event
from typing import Optional, Tuple, Any, Dict, List
import itertools
from dataclasses import replace
import json
import yaml

from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.gpu.gpu import Gpu

from yolobattle.model_training.hw_info import (
    summarize_env, resolve_gpu_selection, fio_seq_rw, get_disk_info, cpu_threads_used
)
from yolobattle.model_training.profiles import TrainProfile, effective_policy, get_profile, equalize_for_split
from yolobattle.model_training.dataset_setup import make_split, IMG_EXTS

from yolobattle.model_training.datasets import ensure_download_once
from yolobattle.model_training.backends import get_backend

WRITABLE_BASE = Path(os.environ.get("WRITABLE_BASE", "/workspace/.cache/splits"))

# near the top of the file
_FIO_CACHE: tuple[float, float] | None = None

def get_fio_cached() -> tuple[float, float]:
    global _FIO_CACHE
    if _FIO_CACHE is None:
        print("[disk] fio (first time)…")
        _FIO_CACHE = fio_seq_rw()
    else:
        print("[disk] using cached fio results")
    return _FIO_CACHE

# ---------- small utils ----------
def slugify(text: str, allowed: str = "-_.") -> str:
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    s = s.replace(" ", "_")
    s = re.sub(fr"[^A-Za-z0-9{re.escape(allowed)}]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s[:180]

def is_wsl() -> bool:
    try:
        with open("/proc/version", "r") as f:
            v = f.read()
        return ("Microsoft" in v) or ("WSL" in v)
    except Exception:
        return False

def effective_username() -> str:
    for key in ("TRUE_USER", "SUDO_USER", "USER"):
        if os.environ.get(key):
            return os.environ[key]
    return getpass.getuser()

def darknet_path() -> str:
    if "APPTAINER_ENVIRONMENT" in os.environ:
        return "/host_workspace/darknet/build/src-cli/darknet"
    elif os.path.exists("/.dockerenv"):
        return "/workspace/darknet/build/src-cli/darknet"
    return "darknet"

def _color_token(p) -> str:
    # Only emit if the profile opted in
    if not getattr(p, "tag_color_preset", False):
        return ""
    v = getattr(p, "color_preset", None)
    # encode "off" vs specific preset
    if v in (None, "", "off"):
        return "__color_off"
    return f"__color_{slugify(str(v))}"


def parse_darknet_data_file(data_path: str) -> dict[str, str]:
    """Read generated Darknet ``.data`` paths needed by shared COCO export."""
    values: dict[str, str] = {}
    path = Path(data_path)
    if not data_path or not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def build_split_for(vf: float, ds, out_dir: str | Path | None = None) -> tuple[str, str]:
    ratio_tag = f"v{int(round(vf*100)):02d}"
    prefix = f"{ds.prefix}_{ratio_tag}"

    sets = None if getattr(ds, "flat_dir", None) else list(ds.sets)

    # Default to the existing adjacent-label split behavior if not provided.
    out_dir = Path(out_dir) if out_dir is not None else Path(ds.root)

    data_path, yaml_path = make_split(
        root=ds.root,
        sets=sets,
        classes=ds.classes,
        names=ds.names,
        prefix=prefix,
        val_frac=vf,
        seed=ds.split_seed,
        neg_subdirs=list(getattr(ds, "neg_subdirs", ())) or None,
        exts=list(getattr(ds, "exts", IMG_EXTS)),
        flat_dir=getattr(ds, "flat_dir", None),
        legos=bool(getattr(ds, "legos", False)),
        predefined_train_dir=getattr(ds, "predefined_train_dir", None),
        predefined_valid_dir=getattr(ds, "predefined_valid_dir", None),
        class_names=tuple(getattr(ds, "class_names", ())),

        # NEW: write outputs under out_dir (i.e., /host_workspace)
        out_dir=out_dir,
    )
    return data_path, ratio_tag


# helper to pull value-list for a given key
def _values_for_key(p: TrainProfile, key: str) -> Tuple[Any, ...]:
    # explicit values win
    if getattr(p, "sweep_values", None) and key in p.sweep_values and p.sweep_values[key]:
        return tuple(p.sweep_values[key])

    # conventional mappings from existing profile fields
    if key == "templates":
        # prefer templated list; else single template; allow None (ultra)
        if p.templates:
            return tuple(p.templates)
        return (p.template,) if p.template is not None else (None,)

    if key == "val_fracs":
        return tuple(p.val_fracs)

    if key == "color_presets":
        if getattr(p, "color_presets", None):
            return tuple(p.color_presets)
        return (p.color_preset,)

    # generic: if the profile has a tuple/list field with this name, use it
    if hasattr(p, key):
        v = getattr(p, key)
        if isinstance(v, (tuple, list)):
            return tuple(v)
        # allow sweeping scalar via explicit sweep_values only
    raise KeyError(f"sweep key '{key}' has no values (add to profile.sweep_values or provide a plural field)")

def _apply_one(p: TrainProfile, key: str, val: Any) -> TrainProfile:
    if key == "templates":
        return replace(p, template=val)
    if key == "color_presets":
        return replace(p, color_preset=val)
    if key == "val_fracs":
        # Never store a scalar; keep the invariant that val_fracs is a tuple
        if isinstance(val, (int, float)):
            return replace(p, val_fracs=(float(val),))
        if isinstance(val, (list, tuple)):
            return replace(p, val_fracs=tuple(val))
        # Unknown type → don’t change it
        return p
    if hasattr(p, key):
        return replace(p, **{key: val})
    return p


def split_manifest_from_data_path(data_path: str) -> Path:
    """'/.../LegoGears_v15.data' -> '/.../LegoGears_v15_split.json'"""
    p = Path(data_path)
    return p.with_name(p.stem + "_split.json")

def read_split_counts_from_data(data_path: str) -> Tuple[int, int]:
    """Return (train_count, valid_count); (0,0) if manifest missing."""
    try:
        manifest = split_manifest_from_data_path(data_path)
        js = json.loads(manifest.read_text(encoding="utf-8"))
        c = js.get("counts", {})
        return int(c.get("train_total", 0)), int(c.get("valid_total", 0))
    except Exception:
        return 0, 0

def _first_val_frac(p: TrainProfile) -> float | None:
    vf_field = getattr(p, "val_fracs", None)
    if isinstance(vf_field, (tuple, list)) and vf_field:
        try:
            return float(vf_field[0])
        except Exception:
            return None
    if isinstance(vf_field, (int, float)):
        return float(vf_field)
    return None

def _target_epochs_from_reference(profile: TrainProfile, reference_data_path: str | None) -> float | None:
    if not reference_data_path:
        return None
    ref_train, _ = read_split_counts_from_data(reference_data_path)
    if ref_train <= 0:
        return None
    return (profile.iterations * profile.batch_size) / float(ref_train)

def _with_ultra_reference_epochs(profile: TrainProfile, reference_data_path: str | None) -> TrainProfile:
    if profile.epochs is not None:
        return profile
    target_epochs = _target_epochs_from_reference(profile, reference_data_path)
    if target_epochs is None:
        return profile
    return replace(profile, epochs=max(1, int(math.ceil(target_epochs))))


# ---------- one run ----------
def run_once(*, p: TrainProfile, template: Optional[str], out_root: str,
             flat_output: bool = False) -> None:
    p = replace(p, policy=effective_policy(p))
    backend = get_backend(p.backend)
    policy = p.policy
    user = effective_username()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    gpu = Gpu()
    sel = resolve_gpu_selection(gpu)

    indices_all = sel["indices_abs"]
    requested = getattr(p, "num_gpus", None)

    if isinstance(requested, int) and requested > 0:
        effective = min(requested, len(indices_all))
        if requested > len(indices_all):
            print(f"[gpu] requested {requested} GPUs but only {len(indices_all)} visible; using {effective}.")
        indices = indices_all[:effective]
    else:
        indices = indices_all

    gpus_str = ",".join(str(i) for i in indices)

    names_all = sel.get("selected_names", []) or []
    vram_all  = sel.get("selected_vram", []) or []

    gpu_name = ", ".join(names_all[:len(indices)]) if names_all else "Unknown GPU"
    vram     = ", ".join(vram_all[:len(indices)])  if vram_all  else "N/A"
    gpu_name_safe = slugify(gpu_name.replace(",", "-"))


    # Keep size in folder names
    size_token = ""
    if getattr(p, "width", None) and getattr(p, "height", None):
        try:
            size_token = f"__{int(p.width)}x{int(p.height)}"
        except Exception:
            size_token = f"__{p.width}x{p.height}"

    # One subdir per YOLO variant (template for Darknet, model name for Ultralytics)
    yolo_variant_raw = backend.model_label(p, template) or "unknown-model"
    yolo_variant_safe = slugify(Path(yolo_variant_raw).stem)
    
    base_tag = p.backend  # "darknet" or "ultralytics" (no template here)

        # --- derive ratio ("Val Fraction") for naming and CSV ---
    ratio_pct = None
    ratio_float = None

    # 1) Primary: profile.val_fracs
    vf_field = getattr(p, "val_fracs", None)
    if isinstance(vf_field, (list, tuple)) and vf_field:
        try:
            ratio_float = float(vf_field[0])
        except (TypeError, ValueError):
            pass
    elif isinstance(vf_field, (int, float)):
        ratio_float = float(vf_field)

    # 2) Secondary: infer from data_path or ultra_data filename if still unknown
    for path_attr, pattern in (
        ("data_path",  r"_v(\d{2})(?:\.data)?$"),
        ("ultra_data", r"_v(\d{2})(?:\.ya?ml)?$"),
    ):
        if ratio_float is None:
            pth = getattr(p, path_attr, None)
            if pth:
                m = re.search(pattern, os.path.basename(pth))
                if m:
                    ratio_pct = m.group(1)
                    ratio_float = int(ratio_pct) / 100.0

    # 3) If we only know the float, format it as XX for naming
    if ratio_float is not None and ratio_pct is None:
        ratio_pct = f"{int(round(ratio_float * 100)):02d}"

    ratio_suffix = f"__val{ratio_pct}" if ratio_pct is not None else ""

    # NEW: attach repeat index if provided by batch driver
    repeat_suffix = ""
    ee_repeat = os.environ.get("EE_REPEAT")
    if ee_repeat:
        repeat_suffix = f"__repeat_{ee_repeat}"

    tag = base_tag + ratio_suffix + size_token + _color_token(p) + repeat_suffix

    if flat_output:
        # Do not create nested variant/benchmark dirs; just use out_root as-is
        variant_dir = out_root
        Path(variant_dir).mkdir(parents=True, exist_ok=True)
        output_dir = variant_dir
    else:
        variant_dir = os.path.join(out_root, yolo_variant_safe)
        os.makedirs(variant_dir, exist_ok=True)
        output_dir = os.path.join(
            variant_dir,
            f"benchmark__{user}__{gpu_name_safe}__{tag}__{now}",
        )
        os.makedirs(output_dir, exist_ok=True)

    os.chdir(output_dir)
    print(f"[out] {output_dir}")

    # GPU watcher (host-indexed for nvidia-smi)
    watch_log = os.path.join(output_dir, "mylogfile.log")
    stop_evt = Event()
    t = None

    # Map selected CUDA logical indices -> host nvidia-smi indices for gpu.watch
    watch_indices = None
    rt_map = sel.get("runtime_smi_map", [])  # [{'logical','bus_id','name','smi_index'}]
    if rt_map:
        l2s = {row["logical"]: row.get("smi_index") for row in rt_map if row.get("smi_index") is not None}
        mapped = [l2s.get(li) for li in indices]
        if all(m is not None for m in mapped) and len(mapped) > 0:
            watch_indices = mapped  # host indices that align with your selected logical indices

    try:
        if gpu.count > 0:
            t = Thread(target=gpu.watch, kwargs={
                "logfile": watch_log, "delay": 1.0, "dense": True,
                "gpu": watch_indices,  # None => watch all; else host indices
                "install_signal_handler": False, "stop_event": stop_evt,
            })
            t.daemon = True
            t.start()
            if watch_indices:
                print(f"[gpuwatch] -> {watch_log} (host idx: {','.join(map(str, watch_indices))} for logical {','.join(map(str, indices))})")
            else:
                print(f"[gpuwatch] -> {watch_log} (watching all)")
        else:
            print("[gpuwatch] no GPUs visible")
    except Exception as e:
        print(f"[gpuwatch] skipped: {e}")

    # Train
    try:
        StopWatch.start("benchmark")
        p, cmd = backend.prepare(p, template=template, output_dir=Path(output_dir),
                                 gpu_indices=indices, gpus_str=gpus_str)
        print(f"[train] {cmd}")
        # ``pipefail`` ensures a trainer failure is not hidden by ``tee``.
        returncode = subprocess.call(["bash", "-o", "pipefail", "-c", cmd])
        StopWatch.stop("benchmark")
        if returncode:
            raise RuntimeError(f"Training command failed with exit code {returncode}")
    finally:
        if t is not None:
            try:
                stop_evt.set()
                gpu.running = False
                t.join(timeout=3)
                print(f"[gpuwatch] stopped")
            except Exception as e:
                print(f"[gpuwatch] stop err: {e}")

    # Post: metrics, env, bundle
    bench = StopWatch.get_benchmark()
    sysinfo = bench["sysinfo"]; b = bench["benchmark"]["benchmark"]
    cpu_name_safe = slugify(sysinfo["cpu"])

    print("[disk] probing…")
    disk_info = get_disk_info()
    dd_w, dd_r = get_fio_cached()
    print(f"[disk] write={dd_w} read={dd_r}")

    env = summarize_env(indices=indices, training_log_path=os.path.join(output_dir, "training_output.log"))

    # --- derive evaluation metrics ---
    map_last_pct = None
    map_best_pct = None
    map_iou = None
    map_points = None
    prf = dict(prec=None, rec=None, f1=None)
    conf_thresh_eval = None

    native = backend.native_metrics(p, Path(output_dir))
    map_last_pct, map_best_pct = native.map_last_pct, native.map_best_pct
    map_iou, map_points, best_iter = native.map_iou, native.map_points, native.best_iter
    conf_thresh_eval = native.conf_thresh
    prf = dict(prec=native.precision, rec=native.recall, f1=native.f1)


    # --- dataset sizing & effective epochs (for CSV) ---
    train_count = valid_count = 0
    approx_epochs = None

    train_count, valid_count, approx_epochs = backend.counts(p, Path(output_dir))


    color_preset_for_csv = p.color_preset if p.color_preset is not None else "off"

    # defaults so names exist even on failure
    coco_ap5095 = coco_ap50 = coco_ap75 = None
    per_iou_cols = {}
    gt_json = det_json = None
    cm_csv_cols = {}   # what we'll merge into row later


    # === External COCO evaluation (framework-agnostic, no env vars) ===
    try:
        from yolobattle.model_training.coco_eval import (
            coco_eval_bbox
        )

        # 1) make sure we have a val list
        val_list = os.path.join(output_dir, "valid.txt")
        if not os.path.isfile(val_list):
            print("[coco] No valid.txt found; skipping external COCO eval")
            raise RuntimeError("no_valid_list")


        # 2) build COCO GT deterministically from DatasetSpec
        from yolobattle.model_training.coco_gt_dispatch import build_coco_gt_for_dataset

        gt_json = os.path.join(output_dir, "val.coco.gt.json")

        if not os.path.isfile(gt_json):
            if not getattr(p, "dataset", None):
                raise RuntimeError("COCO GT requires p.dataset")

            data_fields = parse_darknet_data_file(p.data_path) if p.data_path else {}
            generated_names = data_fields.get("names")

            build_coco_gt_for_dataset(
                dataset=p.dataset,
                valid_list=Path(output_dir) / "valid.txt",
                out_json=Path(gt_json),
                names_path=Path(generated_names) if generated_names else None,
            )

        export_thresh = policy.export_confidence if policy else 0.01

        # 3) framework adapter exports detections using the common policy.
        det_json = os.path.join(output_dir, f"dets_{p.backend}.coco.json")
        backend.export_coco(p, output_dir=Path(output_dir), gt_json=gt_json, det_json=det_json,
                            valid_list=val_list, threshold=export_thresh, gpu_indices=indices)

        coco_metrics = coco_eval_bbox(
            gt_json,
            det_json,
            iou_thresholds=policy.coco_iou_thresholds if policy else None,
        )
        coco_ap5095 = coco_metrics["AP"]
        coco_ap50   = coco_metrics["AP50"]
        coco_ap75   = coco_metrics["AP75"]
        ap_per_iou_pairs = coco_metrics["AP_per_IoU"]
        per_iou_cols = {f"COCO AP@{int(iou*100)} (%)": ap for (iou, ap) in ap_per_iou_pairs}
    except Exception as e:
        coco_ap5095 = None
        coco_ap50 = coco_ap75 = None
        per_iou_cols = {}
        print(f"[coco] external COCO eval skipped: {e}")


    # Prepare defaults
    cm = None
    cm_csv_cols = {}

    # --- Confusion matrix at deployment operating point (dataset-agnostic CSV) ---
    try:
        if gt_json and det_json:
            from yolobattle.model_training.confusion_eval import compute_confusion_from_coco
            cm = compute_confusion_from_coco(
                gt_json,
                det_json,
                iou_thresh=policy.confusion_iou if policy else 0.50,
                conf_thresh=policy.confusion_confidence if policy else 0.50,
                csv_style="generic",  # "per-class" or "none"
                write_json_path=os.path.join(output_dir, "confusion_matrix.json"),
                json_indent=2,
            )
            print(f"[confusion] IoU>={cm['params']['iou_thresh']}, conf>={cm['params']['conf_thresh']}")
            cm_csv_cols = cm["csv_cols"]
        else:
            print("[confusion] skipped: missing COCO JSONs (gt/det)")
    except Exception as e:
        print(f"[confusion] skipped: {e}")
        cm = None
        cm_csv_cols = {}

    # --- optional pretty print (console only) ---
    if cm is not None:
        try:
            per_cls = cm.get("per_class", [])
            names   = [c["class"] for c in per_cls]
            M       = cm.get("matrix", [])

            print("[confusion] per-class")
            for c in per_cls:
                print(f"  {c['class']:<16} TP={c['TP']:>4} FP={c['FP']:>4} FN={c['FN']:>4} "
                    f"P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f}")

            if M and names:
                col_header = " ".join(f"{n[:8]:>9}" for n in names)
                print("[confusion] matrix (pred rows × gt cols)")
                print(f"           {col_header}")
                for i, row in enumerate(M):
                    row_str = " ".join(f"{v:>9d}" for v in row)
                    print(f"{names[i][:10]:>10} {row_str}")
        except Exception:
            # don't fail the run if printing breaks
            pass

    row = {
        "Backend": p.backend,
        "Profile": p.name,
        "YOLO Template": backend.model_label(p, template),
        "Benchmark Policy": policy.name if policy else "legacy",
        "Benchmark Policy Fingerprint": policy.fingerprint() if policy else "legacy",
        "Benchmark Definition": p.benchmark.name if p.benchmark else "legacy",
        "Benchmark Definition Fingerprint": p.benchmark.fingerprint() if p.benchmark else "legacy",
        "Benchmark Time (s)": b["time"],
        "CPU Name": sysinfo["cpu"],
        "CPU Threads": sysinfo["cpu_threads"],
        "CPU Threads Used": cpu_threads_used(),
        "GPU Name": gpu_name,
        "GPU VRAM": vram,
        "Total Memory": sysinfo["mem.total"],
        "OS": "WSL" if is_wsl() else sysinfo["uname.system"],
        "Architecture": sysinfo["uname.machine"],
        "Python Version": sysinfo["python.version"],
        "Disk Capacity": disk_info["Disk Capacity"],
        "Disk Model": disk_info["Disk Model"],
        "Write Speed": dd_w,
        "Read Speed": dd_r,
        "Working Dir": os.getenv("ACTUAL_PWD", "N/A"),
        "CUDA Version": env["cuda_version"],
        "cuDNN Version": env["cudnn_version"],
        "GPUs Used": env["num_gpus_used"],
        "Compute Capability": env["compute_caps_str"],

        # Training knobs + dataset sizing
        "Input Width":  p.width,
        "Input Height": p.height,
        "Input Size":   f"{p.width}x{p.height}",
        "Iterations":   p.iterations,
        "Batch Size":   p.batch_size,
        "Subdivisions": p.subdivisions,
        "Train Images": train_count,
        "Valid Images": valid_count,
        "Approx Epochs": approx_epochs,
        "Color Preset": color_preset_for_csv,

        # Seeds (explicit provenance)
        "Split Seed": getattr(p.dataset, "split_seed", None),
        "Training Seed": getattr(p, "training_seed", None),
        "Repeat": ee_repeat,  # <- NEW: repeat index from EE_REPEAT env (will be empty in CSV if None)

        # Evaluation knobs (fixed schema)
        "Val Fraction": ratio_float,
        "IoU (mAP)": map_iou,
        "mAP Points": map_points,
        "mAP (last %)": map_last_pct,
        "mAP (best %)": map_best_pct,
        "Best Iteration": best_iter,

        # PRF at printed conf
        "Conf Thresh (PRF)": conf_thresh_eval,
        "Precision@Conf": prf["prec"],
        "Recall@Conf": prf["rec"],
        "F1@Conf": prf["f1"],

    }

    row.update(cm_csv_cols)


    # If we have per-IoU APs (COCO), add them as extra columns
    if per_iou_cols:
        row.update({
            "COCO AP50-95 (%)": coco_ap5095,
            "COCO AP50 (%)": coco_ap50,
            "COCO AP75 (%)": coco_ap75,
        })
        row.update(per_iou_cols)

    # Only record pycocotools value; if COCO eval failed, omit the column
    if coco_ap5095 is not None:
        row["mAP50-95 (%)"] = coco_ap5095

    csv_name = f"benchmark__{user}__{gpu_name_safe}__{cpu_name_safe}__{tag}__{now}.csv"
    csv_path = os.path.join(output_dir, csv_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        w.writeheader()
        w.writerow(row)
    print(f"[csv] {csv_name}")

    # Also emit a YAML with the same contents
    yaml_name = Path(csv_name).with_suffix(".yaml").name
    yaml_path = os.path.join(output_dir, yaml_name)
    with open(yaml_path, "w", encoding="utf-8") as fy:
        yaml.safe_dump(row, fy, sort_keys=False)
    print(f"[yaml] {yaml_name}")

    backend.finalize(p, Path(output_dir))


    # Zip (exclude .weights)
    bundle = f"benchmark_bundle__{user}__{gpu_name_safe}__{cpu_name_safe}__{tag}__{now}.zip"
    bundle_path = Path(output_dir) / bundle
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for q in Path(output_dir).rglob("*"):
            if not q.is_file(): continue
            if q.name == bundle: continue
            if q.suffix.lower() == ".weights": continue
            z.write(q, arcname=q.relative_to(output_dir))
    print(f"[zip] {bundle_path}")


# ---------- main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train with a named profile (profiles may contain multiple templates).")
    ap.add_argument("--profile", default="LegoGearsDarknetBenchmark", help="Profile name in profiles.PROFILES")

    #cloudmesh ee
    ap.add_argument("--template", default=None, help="Darknet template override (e.g., yolov7-tiny)")
    ap.add_argument("--val-frac", type=float, default=None, help="Validation fraction override")
    ap.add_argument("--num-gpus", type=int, default=None, help="Request N GPUs")
    ap.add_argument("--iterations", type=int, default=None, help="Optimizer-update budget override")
    ap.add_argument("--learning-rate", type=float, default=None, help="Learning rate override")
    ap.add_argument("--color-preset", default=None, help="Color preset override")
    ap.add_argument("--ultra-model", default=None, help="Ultralytics model override")

    ap.add_argument(
        "--dataset-root",
        default=None,
        help="Override p.dataset.root. If relative, it will be resolved under $DATA_ROOT inside container."
    )

    ap.add_argument("--no-sweep", action="store_true",
                    help="Force single-run even if profile has sweep_keys")

    args = ap.parse_args()

    base_profile = get_profile(args.profile)
    p = base_profile

    #cloudmesh ee
    overrides_used = False

    # Where we were when the script was invoked (ExperimentExecutor run dir, etc.)
    original_cwd = os.getcwd()


    if args.template is not None:
        p = replace(p, template=args.template, templates=())
        overrides_used = True

    if args.val_frac is not None:
        p = replace(p, val_fracs=(float(args.val_frac),))
        overrides_used = True

    if args.num_gpus is not None:
        p = replace(p, num_gpus=int(args.num_gpus))
        overrides_used = True

    if args.iterations is not None:
        if args.iterations <= 0:
            ap.error("--iterations must be positive")
        p = replace(p, iterations=int(args.iterations))
        overrides_used = True

    if args.learning_rate is not None:
        p = replace(p, learning_rate=float(args.learning_rate))
        overrides_used = True

    if args.color_preset is not None:
        p = replace(p, color_preset=args.color_preset, color_presets=(args.color_preset,))
        overrides_used = True

    if args.ultra_model is not None:
        p = replace(p, ultra_model=args.ultra_model)
        overrides_used = True

    if args.dataset_root is not None:
        if getattr(p, "dataset", None) is None:
            print("[dataset-root] profile has no dataset spec; ignoring override.")
        else:
            ds = p.dataset
            ds = replace(ds, root=args.dataset_root)
            p = replace(p, dataset=ds)
            overrides_used = True
            print(f"[dataset-root] using root override: {args.dataset_root}")

    if getattr(p, "dataset", None) is not None:
        ds = p.dataset
        if ds.require_existing:
            r = Path(ds.root)
            # If they passed relative, let later normalization happen;
            # but if absolute and missing, fail early.
            if r.is_absolute() and not r.exists():
                raise FileNotFoundError(
                    f"--dataset-root points to missing path: {ds.root} "
                    f"(profile {p.name} has require_existing=True)"
                )


    # If ee is driving params, don't double-sweep:
    if args.no_sweep or overrides_used:
        p = replace(p, sweep_keys=(), sweep_values={})

    ###

        
    # new (no helpers, single inline check)
    inside_container = os.path.exists("/.dockerenv") or ("APPTAINER_ENVIRONMENT" in os.environ)

    if overrides_used:
        # If any CLI override is used, keep artifacts in the directory
        # where this script was invoked (no nested /outputs/.../benchmark__...).
        out_root = original_cwd
    else:
        out_root_base = "/outputs" if inside_container else "artifacts/outputs"
        out_root = os.path.join(out_root_base, p.name)
        os.makedirs(out_root, exist_ok=True)

    # --- make sure dataset exists at the expected path before split generation ---
    if getattr(p, "dataset", None):
        ds = p.dataset
        base = Path(os.environ.get("DATA_ROOT", "/workspace"))
        root = Path(ds.root)
        # Normalize relative roots to DATA_ROOT (inside container)
        if not root.is_absolute():
            ds = replace(ds, root=str((base / root).resolve()))
        # Download/extract/promote once so set_* dirs are present
        ensure_download_once(ds)
        # Keep normalized spec on the profile for subsequent uses
        p = replace(p, dataset=ds)


    reference_data_path: str | None = None
    if p.backend in ("darknet", "ultralytics") and getattr(p, "dataset", None):
        ref_vf = _first_val_frac(base_profile)
        if ref_vf is not None:
            reference_data_path, _ = build_split_for(ref_vf, p.dataset, out_dir=WRITABLE_BASE)
            print(f"[equalize] reference split val_frac={ref_vf:.4f}")

    sweep_keys = tuple(getattr(p, "sweep_keys", ()) or ())
    if p.backend == "darknet" and sweep_keys:
        # build cartesian product of declared sweep variables
        grid_lists = [ _values_for_key(p, k) for k in sweep_keys ]
        for combo in itertools.product(*grid_lists):
            p_variant = p
            combo_map = dict(zip(sweep_keys, combo))

            # apply *all* sweep keys, including val_fracs
            for k, v in combo_map.items():
                p_variant = _apply_one(p_variant, k, v)

            # decide dataset split for this run
            if "val_fracs" in combo_map:
                vf = float(combo_map["val_fracs"])
            else:
                vf = (p_variant.val_fracs[0] if isinstance(p_variant.val_fracs, (tuple, list)) else float(p_variant.val_fracs))


            if getattr(p_variant, "dataset", None):
                data_path, _ = build_split_for(vf, p_variant.dataset, out_dir=WRITABLE_BASE)
            else:
                data_path = p_variant.data_path  # must already exist



            # equalize per-template to keep epochs ~constant
            target_epochs = _target_epochs_from_reference(p_variant, reference_data_path)
            p_variant = equalize_for_split(
                p_variant,
                data_path=data_path,
                mode="iterations",
                target_epochs=target_epochs,
            )

            # run it
            run_once(p=p_variant, template=p_variant.template, out_root=out_root, 
                # flat_output=overrides_used,
            )

    elif p.backend == "darknet":
        # single Darknet run: still build/refresh split if using DatasetSpec
        data_path = p.data_path
        if getattr(p, "dataset", None):
            vf = (p.val_fracs[0] if isinstance(p.val_fracs, (tuple, list)) else float(p.val_fracs))
            data_path, _ = build_split_for(vf, p.dataset, out_dir=WRITABLE_BASE)
        target_epochs = _target_epochs_from_reference(p, reference_data_path)
        p = equalize_for_split(
            p,
            data_path=data_path,
            mode="iterations",
            target_epochs=target_epochs,
        )
        run_once(p=p, template=p.template or (p.templates[0] if p.templates else None), out_root=out_root,
            # flat_output=overrides_used,
        )



    elif p.backend == "ultralytics":
        # ultralytics: you can still declare sweep_keys for things like epochs/batch_size if you want
        if sweep_keys:
            grid_lists = [ _values_for_key(p, k) for k in sweep_keys ]
            for combo in itertools.product(*grid_lists):
                p_variant = p
                for k, v in dict(zip(sweep_keys, combo)).items():
                    p_variant = _apply_one(p_variant, k, v)
                if getattr(p_variant, "dataset", None):
                    vf = (
                        p_variant.val_fracs[0]
                        if isinstance(p_variant.val_fracs, (tuple, list))
                        else float(p_variant.val_fracs)
                    )
                    data_path, _ = build_split_for(vf, p_variant.dataset, out_dir=WRITABLE_BASE)
                    yaml_path = str(Path(data_path).with_suffix(".yaml"))
                    p_variant = replace(p_variant, data_path=data_path, ultra_data=yaml_path)
                p_variant = _with_ultra_reference_epochs(p_variant, reference_data_path)
                run_once(p=p_variant, template=None, out_root=out_root, 
                    # flat_output=overrides_used
                )
        else:
            if getattr(p, "dataset", None):
                vf = (p.val_fracs[0] if isinstance(p.val_fracs, (tuple, list)) else float(p.val_fracs))
                data_path, _ = build_split_for(vf, p.dataset, out_dir=WRITABLE_BASE)
                yaml_path = str(Path(data_path).with_suffix(".yaml"))
                p = replace(p, data_path=data_path, ultra_data=yaml_path)
            p = _with_ultra_reference_epochs(p, reference_data_path)
            run_once(p=p, template=None, out_root=out_root,
                # flat_output=overrides_used
            )
    elif p.backend == "pytorch_yolov4":
        # The adapter creates its own framework-specific split/labels in the
        # output directory; the shared runner owns everything after training.
        run_once(p=p, template=None, out_root=out_root)
    else:
        raise ValueError(f"Unsupported backend: {p.backend}")
