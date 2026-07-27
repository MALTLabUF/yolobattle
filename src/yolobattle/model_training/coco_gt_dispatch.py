from pathlib import Path
from yolobattle.model_training.coco_build_gt import (
    build_coco_gt,
    build_coco_gt_from_yolo_lists,
)
from yolobattle.model_training.benchmark_definitions import DatasetSpec

def build_coco_gt_for_dataset(
    *,
    dataset: DatasetSpec,
    valid_list: Path,
    out_json: Path,
    names_path: Path | None = None,
) -> None:
    if not valid_list.is_file():
        raise FileNotFoundError(f"valid.txt not found: {valid_list}")

    # Standard YOLO datasets may not ship a Darknet .names file.  In that
    # case dataset_setup creates one beside the writable split artifacts; use
    # the path recorded in the generated .data file when supplied.
    names_path = names_path or (Path(dataset.root) / dataset.names)
    if not names_path.is_file():
        raise FileNotFoundError(f"names file not found: {names_path}")

    annotation_format = dataset.annotation_format.lower().strip()
    if annotation_format not in {"auto", "darkmark_json", "yolo"}:
        raise ValueError(
            f"unsupported annotation_format={dataset.annotation_format!r} for "
            f"dataset {dataset.prefix}; expected 'auto', 'darkmark_json', or 'yolo'"
        )

    # A dataset can contain unrelated JSON metadata in its root.  Do not let
    # that override an explicitly declared YOLO-label evaluation format.
    if annotation_format == "yolo":
        return build_coco_gt_from_yolo_lists(
            list_file=str(valid_list),
            out_json=str(out_json),
            names_path=str(names_path),
        )

    ann_root = Path(dataset.root) / (dataset.flat_dir or "")
    build_coco_gt(
        ann_root=str(ann_root),
        out_json=str(out_json),
        list_file=str(valid_list),
        names_path=str(names_path),
    )
