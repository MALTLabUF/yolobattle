from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

try:
    from spython.main import Client
except Exception:
    Client = None

SUPPORTED_BACKENDS = {"darknet", "ultralytics"}
DEF_NAME = "apptainer.def"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def _state_dir(root: Path) -> Path:
    state_dir = root / ".yolobattle" / "apptainer"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _template_path(root: Path, backend: str) -> Path:
    if backend == "darknet":
        return root / "apptainer" / "darknet" / DEF_NAME
    if backend == "ultralytics":
        return root / "apptainer" / "ultralytics" / DEF_NAME
    raise SystemExit(f"Unsupported backend: {backend}")


def _default_image(root: Path, backend: str) -> Path:
    return _state_dir(root) / f"yolobattle-{backend}.sif"


def _slurm_script_path(root: Path, backend: str) -> Path:
    return root / "slurm" / backend / "script.slurm"


def _slurm_batch_template_path(root: Path, backend: str) -> Path:
    return root / "slurm" / backend / "script.in.slurm"


def _slurm_batch_config_path(root: Path, backend: str) -> Path:
    return root / "slurm" / backend / "config.batch.yaml"


@contextmanager
def _chdir(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _client():
    if Client is None:
        raise SystemExit("Python package 'spython' is not installed. Install with: pip install spython")
    if shutil.which("singularity") is None and shutil.which("apptainer") is None:
        raise SystemExit(
            "Singularity/Apptainer CLI not found in PATH. "
            "Install Apptainer or ensure 'singularity' is available."
        )
    return Client


def _resolve_backend(profile: str | None, backend: str | None) -> str:
    if backend:
        if backend not in SUPPORTED_BACKENDS:
            raise SystemExit(f"Unsupported backend: {backend}")
        return backend
    if profile:
        try:
            from yolobattle.model_training.profiles import get_profile
            p = get_profile(profile)
            if getattr(p, "backend", None) in SUPPORTED_BACKENDS:
                return p.backend
        except Exception:
            pass
    raise SystemExit("Could not resolve backend. Pass --backend darknet|ultralytics.")


def _load_experiment_executor():
    last_exc: Exception | None = None
    try:
        from cloudmesh.ee.experimentexecutor import ExperimentExecutor
        return ExperimentExecutor
    except Exception as first_exc:
        last_exc = first_exc
        root = _repo_root()
        for base in (root.parent, root):
            for repo in ("cloudmesh-rivanna", "cloudmesh-ee"):
                candidate = base / repo / "src"
                if candidate.is_dir() and str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
        try:
            from cloudmesh.ee.experimentexecutor import ExperimentExecutor
            return ExperimentExecutor
        except Exception as import_exc:
            last_exc = import_exc
        raise SystemExit(
            "cloudmesh-ee is required for --batch mode. Install it or place its source at "
            "'../cloudmesh-ee/src' (plus cloudmesh-rivanna) or './cloudmesh-ee/src'. "
            f"Import error: {type(last_exc).__name__}: {last_exc}"
        ) from first_exc


def _stream_lines(stream) -> None:
    for line in stream:
        if isinstance(line, bytes):
            sys.stdout.write(line.decode("utf-8", errors="replace"))
        else:
            sys.stdout.write(str(line))
        if not str(line).endswith("\n"):
            sys.stdout.write("\n")


def _clean_darknet_workspace(workspace_dir: Path, enabled: bool) -> None:
    if not enabled:
        return
    target = workspace_dir / "darknet"
    if target.is_dir():
        shutil.rmtree(target)


def _normalize_train_args(train_args: list[str] | None) -> list[str]:
    normalized = list(train_args or [])
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    return normalized


def _run_sbatch(script: Path, extra: list[str] | None = None, cwd: Path | None = None) -> None:
    sbatch = shutil.which("sbatch")
    if sbatch is None:
        raise SystemExit("sbatch not found in PATH.")

    cmd = [sbatch, str(script)]
    if extra:
        cmd.extend(extra)
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())


def _build_image(args: argparse.Namespace) -> Path:
    client = _client()
    backend = _resolve_backend(profile=None, backend=args.backend)
    root = _repo_root()
    def_path = _template_path(root, backend)
    if not def_path.is_file():
        raise SystemExit(f"Missing Apptainer definition at {def_path}")

    image_path = Path(args.image).resolve() if args.image else _default_image(root, backend)
    options: list[str] = []
    if args.fakeroot:
        options.append("--fakeroot")

    with _chdir(root):
        result = client.build(
            recipe=str(def_path),
            image=str(image_path),
            sudo=args.sudo,
            force=args.force,
            options=options,
            stream=args.stream,
        )

    if args.stream:
        image_out, builder = result
        _stream_lines(builder)
        return Path(image_out)

    return Path(result)


def _run_image(args: argparse.Namespace) -> None:
    client = _client()
    backend = _resolve_backend(profile=args.profile, backend=args.backend)
    root = _repo_root()

    image_path = Path(args.image).resolve() if args.image else _default_image(root, backend)
    if args.build or not image_path.is_file():
        if not image_path.is_file():
            print(f"Image '{image_path}' not found; building it first.")
        build_args = argparse.Namespace(
            backend=backend,
            image=str(image_path),
            sudo=args.sudo,
            fakeroot=args.fakeroot,
            force=args.force,
            stream=True,
        )
        _build_image(build_args)

    workspace_dir = _state_dir(root) / "workspace"
    outputs_dir = Path(args.outputs).resolve() if args.outputs else (root / "artifacts" / "outputs").resolve()
    src_dir = (root / "src").resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _clean_darknet_workspace(workspace_dir, backend == "darknet" and not args.no_clean)

    src_target = "/opt/app/src"
    binds = [
        f"{workspace_dir}:/workspace",
        f"{workspace_dir}:/host_workspace",
        f"{outputs_dir}:/outputs",
        f"{src_dir}:{src_target}",
    ]

    if backend == "ultralytics":
        client.setenv("DATA_ROOT", "/workspace/.cache/datasets")

    client.setenv("TRUE_USER", os.environ.get("USERNAME") or os.environ.get("USER") or "unknown")
    client.setenv("ACTUAL_PWD", str(Path.cwd()))
    client.setenv("WRITABLE_BASE", "/workspace/.cache/splits")
    client.setenv("DARKNET_PARENT", "/host_workspace")

    args_list = ["--profile", args.profile] + _normalize_train_args(args.train_args)

    run_kwargs = dict(
        args=args_list,
        bind=binds,
        nv=not args.no_gpu,
        stream=args.stream,
    )

    result = client.run(str(image_path), **run_kwargs)
    if args.stream:
        _stream_lines(result)


def _generate_slurm_batch(args: argparse.Namespace, backend: str, root: Path) -> Path:
    ExperimentExecutor = _load_experiment_executor()

    source = Path(args.batch_source).resolve() if args.batch_source else _slurm_batch_template_path(root, backend)
    config = Path(args.batch_config).resolve() if args.batch_config else _slurm_batch_config_path(root, backend)
    output_dir = Path(args.batch_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source.is_file():
        raise SystemExit(f"Batch Slurm template not found: {source}")
    if not config.is_file():
        raise SystemExit(f"Batch config not found: {config}")

    name = args.batch_name
    if not name.endswith(".json"):
        name = f"{name}.json"
    name_path = output_dir / name

    ee = ExperimentExecutor(verbose=args.batch_verbose)
    ee.verbose = bool(args.batch_verbose)
    ee.execution_mode = "h"
    ee.name = str(name_path)
    ee.input_dir = str(source.parent.resolve())
    ee.output_dir = str(output_dir)
    ee.source = ExperimentExecutor.update_with_directory(ee.input_dir, source.name)
    script_out_name = source.name.replace(".in.", ".")
    ee.script_out = ExperimentExecutor.update_with_directory(ee.output_dir, script_out_name)
    ee.load_source_template(ee.source)
    ee.update_from_file(str(config))

    if args.batch_os:
        ee.os_variables = [value.strip() for value in args.batch_os.split(",") if value.strip()]
        ee.update_from_os(ee.os_variables)

    if not args.batch_noos:
        ee.update_from_os_environ()

    if not args.batch_nocm:
        ee.update_from_cm_variables()

    if args.batch_experiment:
        ee.experiments = args.batch_experiment
        ee.generate_experiment_permutations(ee.experiments)

    if not ee.permutations:
        raise SystemExit(
            f"No experiment permutations found in {config}. Add an 'experiment' section with list values "
            "or pass --batch-experiment."
        )

    # Keep batch templates resilient when optional fields are omitted in custom configs.
    optional_defaults = {
        "val_frac": None,
        "num_gpus": None,
        "color_preset": None,
        "repeat": None,
    }
    if backend == "darknet":
        optional_defaults["template"] = None
    if backend == "ultralytics":
        optional_defaults["ultra_model"] = None
    for permutation in ee.permutations:
        for key, value in optional_defaults.items():
            permutation.setdefault(key, value)

    required_keys = {"directive", "profile"}
    for permutation in ee.permutations:
        missing = [key for key in required_keys if key not in permutation]
        if missing:
            raise SystemExit(
                f"Batch config must define experiment.{', experiment.'.join(missing)} "
                "(as list values) so cloudmesh-ee can generate Slurm jobs."
            )

    ee.data.setdefault("system", {}).setdefault("dataset_root", None)

    ee.generate_experiment_batch_scripts()
    ee.save_experiment_configuration(name=str(name_path))

    count = len(ee.configuration_parameters or {})
    print(f"Generated {count} Slurm job(s) in: {output_dir}")
    print(f"Experiment manifest: {name_path}")
    return name_path


def _submit_generated_slurm_batch(manifest: Path, extra: list[str] | None = None) -> None:
    if not manifest.is_file():
        raise SystemExit(f"Batch manifest not found: {manifest}")

    content = manifest.read_text(encoding="utf-8")
    experiments = json.loads(content)
    if not experiments:
        raise SystemExit(f"No experiments in manifest: {manifest}")

    submitted = 0
    for exp_id, experiment in experiments.items():
        directory = Path(experiment["directory"])
        script_name = Path(experiment["script"]).name
        if not directory.is_dir():
            raise SystemExit(f"Experiment directory missing for '{exp_id}': {directory}")
        _run_sbatch(Path(script_name), extra=extra, cwd=directory)
        submitted += 1
    print(f"Submitted {submitted} job(s) from: {manifest}")


def _submit_slurm(args: argparse.Namespace) -> None:
    backend = _resolve_backend(profile=args.profile, backend=args.backend)
    root = _repo_root()

    if args.batch:
        manifest = _generate_slurm_batch(args=args, backend=backend, root=root)
        if args.batch_no_submit:
            print("Batch generation complete; submission skipped (--batch-no-submit).")
            return
        _submit_generated_slurm_batch(manifest=manifest, extra=args.extra)
        return

    script = Path(args.script).resolve() if args.script else _slurm_script_path(root, backend)
    if not script.is_file():
        raise SystemExit(f"Slurm script not found: {script}")
    _run_sbatch(script=script, extra=args.extra)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build and run Apptainer images via spython.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build an Apptainer image.")
    build.add_argument("--backend", default="darknet", help="Backend to build (darknet|ultralytics).")
    build.add_argument("--image", default=None, help="Path to output .sif image.")
    build.add_argument("--sudo", action="store_true", help="Use sudo for build.")
    build.add_argument("--fakeroot", action="store_true", help="Use --fakeroot during build.")
    build.add_argument("--force", action="store_true", help="Overwrite existing image.")
    build.add_argument("--stream", action="store_true", help="Stream build output.")
    build.set_defaults(func=_build_image)

    run = subparsers.add_parser("run", help="Run an Apptainer image.")
    run.add_argument("--backend", default=None, help="Override backend (darknet|ultralytics).")
    run.add_argument("--profile", default="LegoGearsDarknet", help="Training profile name.")
    run.add_argument("--image", default=None, help="Path to .sif image.")
    run.add_argument("--outputs", default=None, help="Host outputs directory.")
    run.add_argument("--build", action="store_true", help="Build image before running.")
    run.add_argument("--sudo", action="store_true", help="Use sudo for build when auto-building.")
    run.add_argument("--fakeroot", action="store_true", help="Use --fakeroot when auto-building.")
    run.add_argument("--force", action="store_true", help="Overwrite existing image when auto-building.")
    run.add_argument("--no-gpu", action="store_true", help="Disable --nv.")
    run.add_argument("--no-clean", action="store_true", help="Do not delete /workspace/darknet before run.")
    run.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming output.")
    run.add_argument("train_args", nargs=argparse.REMAINDER, help="Args passed to train module.")
    run.set_defaults(func=_run_image, stream=True)

    slurm = subparsers.add_parser("slurm", help="Submit Slurm jobs for Apptainer.")
    slurm.add_argument("--backend", default=None, help="Override backend (darknet|ultralytics).")
    slurm.add_argument("--profile", default=None, help="Profile name (used to infer backend).")
    slurm.add_argument("--script", default=None, help="Path to slurm script (overrides default).")
    slurm.add_argument("--batch", action="store_true", help="Use cloudmesh-ee API to generate and submit a job batch.")
    slurm.add_argument("--batch-source", default=None, help="Path to EE Slurm template (default: slurm/<backend>/script.in.slurm).")
    slurm.add_argument("--batch-config", default=None, help="Path to EE config file (default: slurm/<backend>/config.batch.yaml).")
    slurm.add_argument("--batch-name", default="yolobattle_runs", help="Experiment manifest name (.json is added automatically).")
    slurm.add_argument("--batch-output-dir", default="project", help="Output directory for generated batch jobs.")
    slurm.add_argument("--batch-experiment", default=None, help="Override EE experiment expression, e.g. \"val_frac=[0.1,0.2] num_gpus=[1]\".")
    slurm.add_argument("--batch-os", default=None, help="Comma-separated OS variables to inject under EE 'os'.")
    slurm.add_argument("--batch-noos", action="store_true", help="Do not merge full OS environment into EE variables.")
    slurm.add_argument("--batch-nocm", action="store_true", help="Do not merge cloudmesh variables into EE variables.")
    slurm.add_argument("--batch-no-submit", action="store_true", help="Generate batch scripts only; skip sbatch submission.")
    slurm.add_argument("--batch-verbose", action="store_true", help="Verbose cloudmesh-ee generation output.")
    slurm.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to sbatch.")
    slurm.set_defaults(func=_submit_slurm)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
