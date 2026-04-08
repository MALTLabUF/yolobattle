#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

from plot_common import git_repo_root, iter_benchmark_csvs, normalize_dataset_name


RUNS_DIR_NAME = "runs-yolobattle"
DEFAULT_DATASET = "LegoGears"

FRAMEWORK_TO_DETS = {
    "ultralytics": "dets_ultralytics.coco.json",
    "darknet": "dets_darknet.coco.json",
}

CLASS_COLORS = [
    "#0b84f3",
    "#f39c12",
    "#27ae60",
    "#d1495b",
    "#8e44ad",
    "#16a085",
    "#e67e22",
    "#2c3e50",
]


@dataclass
class ImageAggregate:
    file_name: str
    width: int
    height: int
    image_path: Path | None = None
    gt_boxes: Dict[tuple, Tuple[float, float, float, float]] | None = None
    matched_boxes: Dict[tuple, List[Tuple[Tuple[float, float, float, float], float, float]]] | None = None
    class_names: Dict[int, str] | None = None
    gt_categories: Dict[tuple, int] | None = None
    matched_count: int = 0
    run_count: int = 0

    def __post_init__(self) -> None:
        if self.gt_boxes is None:
            self.gt_boxes = {}
        if self.matched_boxes is None:
            self.matched_boxes = {}
        if self.class_names is None:
            self.class_names = {}
        if self.gt_categories is None:
            self.gt_categories = {}


def find_runs_dir() -> Path:
    repo_root = git_repo_root()
    candidate = repo_root.parent / RUNS_DIR_NAME
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"Could not find '{RUNS_DIR_NAME}' next to repo root '{repo_root}'. "
        f"Expected: {candidate}"
    )


def benchmark_base() -> Path:
    return find_runs_dir() / "outputs"


def load_csv(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.read_csv(csv_path, engine="python")


def infer_framework(df: pd.DataFrame) -> str | None:
    if "Backend" not in df.columns or df.empty:
        return None
    value = str(df.iloc[0]["Backend"]).strip().lower()
    return value if value in FRAMEWORK_TO_DETS else None


def infer_yolo_variant(df: pd.DataFrame, csv_path: Path) -> str:
    if "YOLO Template" in df.columns and not df.empty:
        value = str(df.iloc[0]["YOLO Template"]).strip()
        if value:
            return Path(value).stem.lower()
    return csv_path.parent.parent.name.strip().lower() or "unknown_yolo"


def infer_dataset(df: pd.DataFrame, csv_path: Path) -> str:
    for col in ("Profile", "Dataset", "Data Profile"):
        if col in df.columns and not df.empty:
            value = str(df.iloc[0][col]).strip()
            if value:
                return normalize_dataset_name(value, str(csv_path))
    return normalize_dataset_name(csv_path.as_posix(), str(csv_path))


def canonical_sort_key(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0, w, h)


def normalize_box(bbox: List[float], width: float, height: float) -> Tuple[float, float, float, float] | None:
    if len(bbox) != 4 or width <= 0 or height <= 0:
        return None
    x, y, w, h = map(float, bbox)
    if w <= 0 or h <= 0:
        return None

    x = max(0.0, min(width, x))
    y = max(0.0, min(height, y))
    w = max(0.0, min(width - x, w))
    h = max(0.0, min(height - y, h))
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def color_for_class(category_id: int) -> str:
    idx = max(0, category_id)
    return CLASS_COLORS[idx % len(CLASS_COLORS)]


def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def stable_gt_key(category_id: int, bbox: Tuple[float, float, float, float]) -> tuple:
    x, y, w, h = bbox
    return (int(category_id), round(x, 1), round(y, 1), round(w, 1), round(h, 1))


def valid_txt_image_paths(valid_path: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not valid_path.is_file():
        return mapping
    for line in valid_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = Path(s)
        mapping[p.name] = p
    return mapping


def image_search_roots() -> List[Path]:
    repo_root = git_repo_root()
    return [
        repo_root / ".yolobattle" / "workspace" / ".cache" / "datasets",
        repo_root / ".yolobattle" / "workspace",
        repo_root.parent / "Downloads",
        Path.home() / "Downloads",
    ]


def resolve_existing_image_path(original: Path, basename: str, cache: Dict[str, Path | None]) -> Path | None:
    cached = cache.get(basename)
    if basename in cache:
        return cached

    candidates = [original]
    try:
        parts = original.parts
        if parts[:3] == ("/", "workspace", ".cache"):
            rel = Path(*parts[3:])
            candidates.append(git_repo_root() / ".yolobattle" / "workspace" / ".cache" / rel)
        if parts[:2] == ("/", "workspace"):
            rel = Path(*parts[2:])
            candidates.append(git_repo_root() / ".yolobattle" / "workspace" / rel)
    except Exception:
        pass

    for candidate in candidates:
        if candidate.is_file():
            cache[basename] = candidate
            return candidate

    for root in image_search_roots():
        if not root.is_dir():
            continue
        match = next(root.rglob(basename), None)
        if match is not None and match.is_file():
            cache[basename] = match
            return match

    cache[basename] = None
    return None


def iter_target_runs(base_dir: Path, dataset_name: str) -> Iterable[Tuple[Path, pd.DataFrame, str, str]]:
    for csv_str in iter_benchmark_csvs([str(base_dir)]):
        csv_path = Path(csv_str)
        df = load_csv(csv_path)
        dataset = infer_dataset(df, csv_path)
        if dataset != dataset_name:
            continue
        framework = infer_framework(df)
        if framework is None:
            continue
        yolo_variant = infer_yolo_variant(df, csv_path)
        yield csv_path, df, framework, yolo_variant


def aggregate_framework(
    runs: Iterable[Tuple[Path, pd.DataFrame, str, str]],
    *,
    framework: str,
    yolo_variant: str,
    score_threshold: float,
    match_iou_threshold: float,
) -> Tuple[Dict[str, ImageAggregate], List[str]]:
    aggregates: Dict[str, ImageAggregate] = {}
    warnings: List[str] = []
    image_cache: Dict[str, Path | None] = {}

    for csv_path, _df, run_framework, run_yolo_variant in runs:
        if run_framework != framework or run_yolo_variant != yolo_variant:
            continue

        run_dir = csv_path.parent
        dets_path = run_dir / FRAMEWORK_TO_DETS[framework]
        gt_path = run_dir / "val.coco.gt.json"
        valid_path = run_dir / "valid.txt"

        if not dets_path.is_file():
            warnings.append(f"missing dets: {dets_path}")
            continue
        if not gt_path.is_file():
            warnings.append(f"missing gt: {gt_path}")
            continue

        gt = load_json(gt_path)
        if not isinstance(gt, dict) or "images" not in gt:
            warnings.append(f"invalid gt json: {gt_path}")
            continue

        valid_mapping = valid_txt_image_paths(valid_path)
        category_names = {
            int(rec["id"]): str(rec["name"])
            for rec in gt.get("categories", [])
            if "id" in rec and "name" in rec
        }
        image_meta = {
            int(rec["id"]): {
                "file_name": Path(rec["file_name"]).name,
                "width": int(rec["width"]),
                "height": int(rec["height"]),
                "image_path": resolve_existing_image_path(
                    valid_mapping.get(Path(rec["file_name"]).name, Path(rec["file_name"])),
                    Path(rec["file_name"]).name,
                    image_cache,
                ),
            }
            for rec in gt["images"]
        }
        gt_annotations_by_image: Dict[int, List[dict]] = {}
        for ann in gt.get("annotations", []):
            image_id = ann.get("image_id")
            ann_id = ann.get("id")
            bbox = ann.get("bbox")
            category_id = ann.get("category_id")
            if image_id is None or ann_id is None or not isinstance(bbox, list) or category_id is None:
                continue
            meta = image_meta.get(int(image_id))
            if meta is None:
                continue
            bbox_px = normalize_box(bbox, meta["width"], meta["height"])
            if bbox_px is None:
                continue
            gt_annotations_by_image.setdefault(int(image_id), []).append(
                {
                    "category_id": int(category_id),
                    "bbox": bbox_px,
                    "key": stable_gt_key(int(category_id), bbox_px),
                }
            )

        dets = load_json(dets_path)
        if not isinstance(dets, list):
            warnings.append(f"invalid det json: {dets_path}")
            continue

        per_image_dets: Dict[int, List[dict]] = {}
        for det in dets:
            if not isinstance(det, dict):
                continue
            score = float(det.get("score", 0.0) or 0.0)
            if score < score_threshold:
                continue
            image_id = det.get("image_id")
            meta = image_meta.get(int(image_id)) if image_id is not None else None
            if meta is None:
                continue
            bbox = det.get("bbox")
            if not isinstance(bbox, list):
                continue
            bbox_px = normalize_box(bbox, meta["width"], meta["height"])
            if bbox_px is None:
                continue
            category_id = int(det.get("category_id", -1))
            per_image_dets.setdefault(int(image_id), []).append(
                {"category_id": category_id, "bbox": bbox_px, "score": score}
            )

        matched_image_names: set[str] = set()
        for image_id, anns in gt_annotations_by_image.items():
            meta = image_meta.get(image_id)
            if meta is None:
                continue
            key = meta["file_name"]
            agg = aggregates.get(key)
            if agg is None:
                agg = ImageAggregate(
                    file_name=key,
                    width=meta["width"],
                    height=meta["height"],
                    image_path=meta["image_path"],
                    class_names=dict(category_names),
                )
                aggregates[key] = agg
            elif agg.image_path is None and meta["image_path"] is not None:
                agg.image_path = meta["image_path"]
            if not agg.class_names:
                agg.class_names = dict(category_names)
            for ann in anns:
                agg.gt_boxes.setdefault(ann["key"], ann["bbox"])
                agg.gt_categories.setdefault(ann["key"], ann["category_id"])

            detections = per_image_dets.get(image_id, [])
            used_det_idxs: set[int] = set()
            matched_here = False
            for ann in anns:
                best_idx = None
                best_iou = -1.0
                for idx, det in enumerate(detections):
                    if idx in used_det_idxs:
                        continue
                    if det["category_id"] != ann["category_id"]:
                        continue
                    iou = bbox_iou(ann["bbox"], det["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                if best_idx is None or best_iou < match_iou_threshold:
                    continue
                det = detections[best_idx]
                used_det_idxs.add(best_idx)
                agg.matched_boxes.setdefault(ann["key"], []).append((det["bbox"], float(det["score"]), best_iou))
                agg.matched_count += 1
                matched_here = True

            if matched_here:
                matched_image_names.add(key)

        for key in matched_image_names:
            aggregates[key].run_count += 1

    return aggregates, warnings


def weighted_mean_box(
    items: List[Tuple[Tuple[float, float, float, float], float, float]]
) -> Tuple[float, float, float, float]:
    weights = [max(1e-6, score * iou) for _bbox, score, iou in items]
    denom = float(sum(weights))
    if denom <= 0:
        denom = float(len(items))
    return (
        sum(bbox[0] * w for (bbox, _score, _iou), w in zip(items, weights)) / denom,
        sum(bbox[1] * w for (bbox, _score, _iou), w in zip(items, weights)) / denom,
        sum(bbox[2] * w for (bbox, _score, _iou), w in zip(items, weights)) / denom,
        sum(bbox[3] * w for (bbox, _score, _iou), w in zip(items, weights)) / denom,
    )


def mean_support(items: List[Tuple[Tuple[float, float, float, float], float, float]]) -> float:
    if not items:
        return 0.0
    return sum(score for _bbox, score, _iou in items) / float(len(items))


def plot_framework_boxes(
    aggregates: Dict[str, ImageAggregate],
    *,
    framework: str,
    yolo_variant: str,
    out_dir: Path,
) -> List[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[dict] = []

    for image_name in sorted(aggregates):
        agg = aggregates[image_name]
        if not agg.matched_boxes:
            continue
        fig_w = max(5.0, 6.0 * (agg.width / max(1, agg.height)))
        fig_h = 5.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        if agg.image_path is not None and agg.image_path.is_file():
            img = plt.imread(agg.image_path)
            ax.imshow(img, extent=[0, agg.width, agg.height, 0], aspect="auto")
        else:
            ax.set_facecolor("#f3f3f3")

        legend_handles: Dict[int, patches.Patch] = {}
        for ann_id, gt_bbox in sorted(agg.gt_boxes.items()):
            gt_category = agg.gt_categories.get(ann_id, -1)
            gt_rect = patches.Rectangle(
                (gt_bbox[0], gt_bbox[1]),
                gt_bbox[2],
                gt_bbox[3],
                linewidth=1.4,
                edgecolor="#7a7a7a",
                facecolor="none",
                linestyle="--",
                alpha=0.8,
            )
            ax.add_patch(gt_rect)
            if -1 not in legend_handles:
                legend_handles[-1] = patches.Patch(
                    facecolor="none",
                    edgecolor="#7a7a7a",
                    linewidth=1.4,
                    linestyle="--",
                    label="ground truth",
                )

            items = agg.matched_boxes.get(ann_id, [])
            if not items:
                continue
            avg_x, avg_y, avg_w, avg_h = weighted_mean_box(items)
            color = color_for_class(gt_category)
            rect = patches.Rectangle(
                (avg_x, avg_y),
                avg_w,
                avg_h,
                linewidth=2.2,
                edgecolor=color,
                facecolor="none",
                alpha=0.95,
            )
            ax.add_patch(rect)
            support_n = len(items)
            support_mean = mean_support(items)
            label_x = avg_x
            label_y = max(10.0, avg_y - 6.0)
            ax.text(
                label_x,
                label_y,
                f"n={support_n}, c={support_mean:.2f}",
                fontsize=8,
                color=color,
                bbox={
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.75,
                    "boxstyle": "round,pad=0.15",
                },
            )
            if gt_category not in legend_handles:
                class_name = agg.class_names.get(gt_category, f"class {gt_category}")
                legend_handles[gt_category] = patches.Patch(
                    facecolor="none",
                    edgecolor=color,
                    linewidth=2.2,
                    label=class_name,
                )

        ax.set_xlim(0, agg.width)
        ax.set_ylim(agg.height, 0)
        ax.set_title(f"{framework} {yolo_variant} average boxes: {agg.file_name}")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        if legend_handles:
            ax.legend(
                handles=[legend_handles[k] for k in sorted(legend_handles, key=lambda x: (x == -1, x))],
                loc="lower left",
                framealpha=0.85,
                facecolor="white",
                title="Classes",
            )
        fig.tight_layout()

        out_path = out_dir / f"{Path(image_name).stem}__{framework}__{yolo_variant}_avg_boxes.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

        manifest_rows.append(
            {
                "framework": framework,
                "yolo_variant": yolo_variant,
                "image_name": agg.file_name,
                "image_width": agg.width,
                "image_height": agg.height,
                "matched_boxes_used": agg.matched_count,
                "runs_used": agg.run_count,
                "avg_boxes_drawn": len(agg.matched_boxes),
                "image_path": str(agg.image_path) if agg.image_path is not None else "",
                "output_png": str(out_path),
            }
        )

    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-image plots with the original image and averaged "
            "bounding-box outlines from dets_*.coco.json "
            "files discovered under runs-yolobattle."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Canonical dataset name, e.g. LegoGears.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Ignore detections below this score.")
    parser.add_argument("--match-iou-threshold", type=float, default=0.10, help="Minimum IoU to match a detection to a GT object.")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Defaults to analysis/darknet/avg_boxes_<dataset>.",
    )
    args = parser.parse_args()

    base_dir = benchmark_base()
    out_dir = (
        Path(args.outdir).resolve()
        if args.outdir
        else Path(__file__).resolve().parent / f"avg_boxes_{args.dataset}"
    )

    runs = list(iter_target_runs(base_dir, args.dataset))
    if not runs:
        raise SystemExit(f"No benchmark CSV runs found for dataset '{args.dataset}' under {base_dir}")

    all_rows: List[dict] = []
    warning_rows: List[dict] = []

    present_pairs = sorted({(framework, yolo_variant) for _csv, _df, framework, yolo_variant in runs})

    for framework, yolo_variant in present_pairs:
        framework_aggregates, warnings = aggregate_framework(
            runs,
            framework=framework,
            yolo_variant=yolo_variant,
            score_threshold=args.score_threshold,
            match_iou_threshold=args.match_iou_threshold,
        )
        for warning in warnings:
            warning_rows.append({"framework": framework, "yolo_variant": yolo_variant, "warning": warning})

        if not framework_aggregates:
            print(
                f"[WARN] No usable {framework} detections found for dataset "
                f"'{args.dataset}' and yolo '{yolo_variant}'."
            )
            continue

        framework_dir = out_dir / framework / yolo_variant
        rows = plot_framework_boxes(
            framework_aggregates,
            framework=framework,
            yolo_variant=yolo_variant,
            out_dir=framework_dir,
        )
        all_rows.extend(rows)
        print(
            f"[OK] Wrote {len(rows)} average-box plot(s) for {framework}/{yolo_variant} "
            f"to {framework_dir}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    pd.DataFrame(all_rows).to_csv(manifest_path, index=False)
    print(f"[INFO] Manifest written to {manifest_path}")

    if warning_rows:
        warnings_path = out_dir / "warnings.csv"
        pd.DataFrame(warning_rows).to_csv(warnings_path, index=False)
        print(f"[INFO] Warning log written to {warnings_path}")


if __name__ == "__main__":
    main()
