"""Load an existing Darknet project without a registered training profile."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import TYPE_CHECKING

from .benchmark_definitions import DatasetSpec

if TYPE_CHECKING:
    from .profile_models import TrainProfile


@dataclass(frozen=True)
class DarknetProject:
    cfg: Path
    cfg_bytes: bytes
    data: Path
    data_values: dict[str, str]
    names: tuple[str, ...]
    train: tuple[str, ...]
    valid: tuple[str, ...]
    weights: Path | None
    letter_box: bool
    path_relocations: tuple[tuple[str, str, str], ...]


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith(("#", ";"))]


def _settings(lines) -> dict[str, str]:
    values = {}
    for line in lines:
        line = re.split(r"\s*[#;]", line, maxsplit=1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip().lower()] = value.strip()
    return values


def _select(root: Path, value: str | None, suffix: str) -> Path:
    if value:
        path = Path(value).expanduser()
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise ValueError(f"File not found: {path}")
        return path.resolve()
    candidates = sorted(root.glob(f"*{suffix}"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one {suffix} file in {root}; found {len(candidates)}. "
                         f"Select one with --{suffix[1:]}-path.")
    return candidates[0].resolve()


def load_adhoc_profile(folder: str, *, cfg: str | None = None, data: str | None = None,
                       weights: str | None = None, path_maps: list[str] | None = None) -> TrainProfile:
    from .profile_models import TrainProfile

    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Ad hoc dataset folder does not exist: {root}")
    cfg_path, data_path = _select(root, cfg, ".cfg"), _select(root, data, ".data")
    mappings = []
    for mapping in path_maps or []:
        old, sep, new = mapping.partition("=")
        if not sep or not old.strip() or not new.strip():
            raise ValueError("--path-map must be OLD_PREFIX=NEW_PREFIX")
        mappings.append((old.replace("\\", "/").rstrip("/"),
                         str(Path(new).expanduser().resolve()).replace("\\", "/")))
    mappings.sort(key=lambda pair: len(pair[0]), reverse=True)
    inferred: set[str] = set()
    inferred_prefixes: tuple[str, ...] = ()
    anchors = {root.name}
    resolved_images: dict[tuple[str, Path], Path] = {}
    relocations: set[tuple[str, str, str]] = set()

    def resolve_file(value: str, parent: Path, *, metadata: bool = False) -> Path:
        normalized = value.strip().strip('"').replace("\\", "/")
        cache_key = (normalized, parent)
        if not metadata and cache_key in resolved_images:
            return resolved_images[cache_key]

        def remember(candidate: Path) -> Path:
            resolved = candidate.resolve()
            if not metadata:
                resolved_images[cache_key] = resolved
            return resolved

        mapped = False
        for old, new in mappings:
            if normalized == old or normalized.startswith(old + "/"):
                normalized = new + normalized[len(old):]
                mapped = True
                relocations.add((old, new, "explicit"))
                break
        path = Path(normalized).expanduser()
        absolute = PurePosixPath(normalized).is_absolute() or PureWindowsPath(normalized).is_absolute()
        parts = PurePosixPath(normalized).parts

        def choose(matches: dict[Path, str]) -> Path | None:
            if len(matches) > 1:
                raise ValueError(f"Ambiguous relocated path {value!r}: "
                                 f"{', '.join(map(str, matches))}. Use --path-map OLD_PREFIX=NEW_PREFIX.")
            if matches:
                candidate, old_prefix = next(iter(matches.items()))
                if metadata:
                    inferred.add(old_prefix)
                if old_prefix != root.as_posix():
                    relocations.add((old_prefix, root.as_posix(), "automatic"))
                return remember(candidate)
            return None

        # Explicit mappings always win, including when their destination is
        # missing. Do not silently rescue a typo using automatic inference.
        if absolute and not mapped and ".." not in parts:
            if metadata:
                # Metadata can identify the old root even when the project or
                # its container mount was renamed. Search only root-relative
                # suffixes, never recursively by filename across the dataset.
                matches = {}
                for index in range(1, len(parts)):
                    candidate = root.joinpath(*parts[index:])
                    if candidate.is_file():
                        matches[candidate.resolve()] = str(PurePosixPath(*parts[:index]))
                selected = choose(matches)
                if selected is not None:
                    return selected
            else:
                for old in inferred_prefixes:
                    if normalized.startswith(old + "/"):
                        path = root / normalized[len(old) + 1:]
                        mapped = True
                        break
                if not mapped:
                    # Lists can mix prefixes from multiple machines. Preserve
                    # everything below an exact project-directory component.
                    matches = {}
                    for index, part in enumerate(parts[:-1]):
                        if part in anchors:
                            candidate = root.joinpath(*parts[index + 1:])
                            if candidate.is_file():
                                matches[candidate.resolve()] = str(PurePosixPath(*parts[:index + 1]))
                    selected = choose(matches)
                    if selected is not None:
                        return selected
        candidates = [path] if absolute or path.is_absolute() else [parent / path, root / path]
        for candidate in candidates:
            if candidate.is_file():
                return remember(candidate)
        raise ValueError(f"Cannot find {value!r} referenced from {parent}. "
                         "Bind its directory into the container or use --path-map OLD_PREFIX=NEW_PREFIX.")

    cfg_bytes = cfg_path.read_bytes()
    sections: list[tuple[str, dict[str, str]]] = []
    for line in cfg_bytes.decode("utf-8-sig").splitlines():
        clean = re.split(r"\s*[#;]", line, maxsplit=1)[0].strip()
        if clean.startswith("[") and clean.endswith("]"):
            sections.append((clean[1:-1].strip().lower(), {}))
        elif sections:
            sections[-1][1].update(_settings([line]))
    nets = [values for name, values in sections if name in {"net", "network"}]
    if len(nets) != 1:
        raise ValueError(f"{cfg_path} must contain exactly one [net] section")
    net = nets[0]

    def positive_int(key: str) -> int:
        try:
            value = int(net[key])
            if value > 0:
                return value
        except (KeyError, ValueError):
            pass
        raise ValueError(f"{cfg_path}: [net] {key} must be a positive integer")

    width, height = positive_int("width"), positive_int("height")
    batch, subdivisions = positive_int("batch"), positive_int("subdivisions")
    iterations = positive_int("max_batches")
    if batch % subdivisions:
        raise ValueError("cfg batch must be divisible by subdivisions")
    try:
        learning_rate = float(net["learning_rate"])
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError
    except (KeyError, ValueError):
        raise ValueError(f"{cfg_path}: learning_rate must be a positive finite number") from None

    values = _settings(_lines(data_path))
    for key in ("classes", "names", "train", "valid"):
        if not values.get(key):
            raise ValueError(f"{data_path} is missing {key}=")
    names_path = resolve_file(values["names"], data_path.parent, metadata=True)
    names = tuple(_lines(names_path))
    try:
        classes = int(values["classes"])
        heads = [int(section["classes"]) for name, section in sections if name == "yolo"]
    except (KeyError, ValueError):
        raise ValueError(".data and every [yolo] head must specify integer classes") from None
    if not names or classes != len(names) or not heads or any(n != classes for n in heads):
        raise ValueError("Class counts must agree in .names, .data, and every [yolo] head")

    # Resolve both metadata paths first so either can establish the old root
    # before processing images (the other metadata paths may be relative).
    lists = {key: resolve_file(values[key], data_path.parent, metadata=True) for key in ("train", "valid")}
    inferred_prefixes = tuple(sorted(inferred, key=len, reverse=True))
    anchors.update(PurePosixPath(old).name for old in inferred)

    def read_list(key: str) -> tuple[str, ...]:
        path = lists[key]
        images = tuple(str(resolve_file(line, path.parent)) for line in _lines(path))
        if not images:
            raise ValueError(f"{key} image list is empty: {path}")
        return images

    train, valid = read_list("train"), read_list("valid")
    for key in ("map", "test"):
        if values.get(key):
            values[key] = str(resolve_file(values[key], data_path.parent))
    project = DarknetProject(
        cfg=cfg_path, cfg_bytes=cfg_bytes, data=data_path, data_values=values,
        names=names, train=train, valid=valid,
        weights=resolve_file(weights, root) if weights else None,
        letter_box=net.get("letter_box", "0") == "1",
        path_relocations=tuple(sorted(relocations)),
    )
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", root.name).strip("._-") or "dataset"
    return TrainProfile(
        name=f"adhoc_{safe_name}", backend="darknet", data_path=str(data_path),
        cfg_out=str(cfg_path), width=width, height=height, batch_size=batch,
        subdivisions=subdivisions, iterations=iterations, learning_rate=learning_rate,
        template=cfg_path.stem, val_fracs=(len(valid) / (len(train) + len(valid)),),
        dataset=DatasetSpec(root=str(root), sets=(), classes=classes, names=str(names_path),
                            prefix=safe_name, require_existing=True, class_names=names,
                            annotation_format="yolo"),
        darknet_project=project, custom_data=True, cfg_source=str(cfg_path),
    )


def prepare_adhoc(profile: TrainProfile, output_dir: Path) -> TrainProfile:
    """Snapshot the cfg and normalize data paths in a writable run directory."""
    from .dataset_setup import project_paths_for_darknet

    project = profile.darknet_project
    if project is None:
        raise ValueError("Missing ad hoc project")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = output_dir / project.cfg.name
    data = output_dir / "dataset.data"
    names = output_dir / "classes.names"
    cfg.write_bytes(project.cfg_bytes)
    names.write_text("\n".join(project.names) + "\n", encoding="utf-8")
    native_lists = {}
    for key, images in (("train", project.train), ("valid", project.valid)):
        (output_dir / f"{key}.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
        # Shared evaluation keeps original paths. Only Darknet sees the
        # adjacent-label view, just as it does for dataset-backed profiles.
        native_images, projected = project_paths_for_darknet(
            list(map(Path, images)), out_dir=output_dir / "darknet_inputs",
            prefix="dataset", split_name=key,
        )
        native_list = output_dir / f"{key}.txt"
        if projected:
            native_list = output_dir / f"darknet_{key}.txt"
            native_list.write_text("\n".join(map(str, native_images)) + "\n", encoding="utf-8")
        native_lists[key] = str(native_list)
    values = dict(project.data_values)
    values.update(native_lists, names=str(names), backup=str(output_dir))
    data.write_text("".join(f"{key} = {value}\n" for key, value in values.items()), encoding="utf-8")
    (output_dir / "dataset_split.json").write_text(json.dumps({
        "source": "existing_darknet_lists",
        "counts": {"train_total": len(project.train), "valid_total": len(project.valid)},
    }, indent=2), encoding="utf-8")
    (output_dir / "adhoc.json").write_text(json.dumps({
        "source_cfg": str(project.cfg), "source_data": str(project.data),
        "cfg_sha256": hashlib.sha256(project.cfg_bytes).hexdigest(),
        "initial_weights": str(project.weights) if project.weights else None,
        "path_relocations": [dict(source=old, destination=new, method=method)
                             for old, new, method in project.path_relocations],
    }, indent=2), encoding="utf-8")
    return replace(profile, cfg_out=str(cfg), data_path=str(data))
