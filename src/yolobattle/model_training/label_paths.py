"""Resolve YOLO label files without forcing a single dataset directory layout."""

from __future__ import annotations

from pathlib import Path


def label_path_for_image(image_path: Path) -> Path:
    """Return the preferred YOLO ``.txt`` label path for an image.

    DarkMark-style datasets keep the label next to the image; that layout is
    checked first.  Standard YOLO datasets instead map the nearest ``images``
    directory to its sibling ``labels`` directory while preserving nested
    relative paths.
    """
    image_path = Path(image_path)
    adjacent = image_path.with_suffix(".txt")
    if adjacent.is_file():
        return adjacent

    for images_dir in (image_path.parent, *image_path.parents):
        if images_dir.name.lower() != "images":
            continue
        relative_image = image_path.relative_to(images_dir)
        standard_yolo = (
            images_dir.parent / "labels" / relative_image.with_suffix(".txt")
        )
        if standard_yolo.is_file():
            return standard_yolo

    # Preserve the adjacent-label path when neither layout has a label.  The
    # caller's existing ``is_file``/``exists`` check decides how missing labels
    # are handled.
    return adjacent
