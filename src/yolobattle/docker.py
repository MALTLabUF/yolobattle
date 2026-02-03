from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import docker
    from docker.errors import DockerException, ImageNotFound, NotFound
    from docker.types import DeviceRequest
except Exception:
    docker = None
    DockerException = Exception
    ImageNotFound = Exception
    NotFound = Exception
    DeviceRequest = None

DOCKERFILE_NAME = "Dockerfile"
BUILD_SCRIPT_NAME = "build_darknet.sh"
SUPPORTED_BACKENDS = {"darknet", "ultralytics"}
DEFAULT_IMAGE = "yolobattle-container"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def _state_dirs(root: Path) -> tuple[Path, Path]:
    state_dir = root / ".yolobattle"
    workspace_dir = state_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return state_dir, workspace_dir


def _template_paths(root: Path, backend: str) -> tuple[Path, Path | None]:
    if backend == "darknet":
        template_dir = root / "docker" / "darknet"
        dockerfile = template_dir / DOCKERFILE_NAME
        build_sh = template_dir / BUILD_SCRIPT_NAME
        return dockerfile, build_sh
    if backend == "ultralytics":
        template_dir = root / "docker" / "ultralytics"
        dockerfile = template_dir / DOCKERFILE_NAME
        return dockerfile, None
    raise SystemExit(f"Unsupported backend: {backend}")


def _ensure_assets(root: Path, backend: str) -> dict[str, Path]:
    _, workspace_dir = _state_dirs(root)
    dockerfile, build_sh = _template_paths(root, backend)
    if not dockerfile.is_file():
        raise SystemExit(f"Missing Dockerfile template at {dockerfile}")
    if build_sh is not None:
        if not build_sh.is_file():
            raise SystemExit(f"Missing build script template at {build_sh}")
        workspace_build = workspace_dir / BUILD_SCRIPT_NAME
        workspace_build.write_bytes(build_sh.read_bytes())
        if os.name != "nt":
            workspace_build.chmod(0o755)
    return {
        "dockerfile": dockerfile,
        "workspace": workspace_dir,
    }


def _docker_client():
    if docker is None:
        raise SystemExit("Python package 'docker' is not installed. Install with: pip install docker")
    try:
        client = docker.from_env()
        client.ping()
    except DockerException as exc:
        raise SystemExit(f"Docker API not available: {exc}")
    return client


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


def _default_image_tag(backend: str) -> str:
    return f"yolobattle-{backend}"


def _image_exists(client, image: str) -> bool:
    try:
        client.images.get(image)
        return True
    except ImageNotFound:
        return False
    except DockerException:
        return False


def _default_uid_gid() -> tuple[int, int]:
    if os.name == "nt":
        return 1000, 1000
    return os.getuid(), os.getgid()


def _windows_disk_env() -> dict[str, str]:
    if os.name != "nt":
        return {}
    env: dict[str, str] = {}
    for key in ("WINDOWS_HARD_DRIVE", "WINDOWS_HARD_DRIVE_CAPACITY"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    if env.get("WINDOWS_HARD_DRIVE") and env.get("WINDOWS_HARD_DRIVE_CAPACITY"):
        return env

    def _wmic_value(field: str) -> str | None:
        try:
            output = subprocess.check_output(
                ["wmic", "diskdrive", "where", "DeviceID like '%PHYSICALDRIVE0%'", "get", field],
                text=True,
            )
        except Exception:
            return None
        lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
        for ln in lines:
            if ln.lower() == field.lower():
                continue
            return ln
        return None

    model = _wmic_value("Model")
    size = _wmic_value("Size")
    if model:
        env["WINDOWS_HARD_DRIVE"] = model
    if size:
        env["WINDOWS_HARD_DRIVE_CAPACITY"] = size
    return env


def _shell_quote(value: str) -> str:
    if value == "":
        return "''"
    if any(c in value for c in " \t\n\"'\\$`"):
        return "'" + value.replace("'", "'\"'\"'") + "'"
    return value


def _shell_join(parts: list[str]) -> str:
    return " ".join(_shell_quote(p) for p in parts)


def _parse_shm_size(value: str) -> int:
    v = value.strip().lower()
    if v.endswith("b"):
        v = v[:-1]
    unit = v[-1] if v and v[-1].isalpha() else ""
    num_str = v[:-1] if unit else v
    try:
        num = float(num_str)
    except ValueError:
        raise SystemExit(f"Invalid --shm-size value: {value}")
    if num <= 0:
        raise SystemExit(f"Invalid --shm-size value: {value}")
    factors = {
        "": 1,
        "k": 1024,
        "m": 1024 ** 2,
        "g": 1024 ** 3,
        "t": 1024 ** 4,
    }
    if unit not in factors:
        raise SystemExit(f"Invalid --shm-size unit: {value}")
    return int(num * factors[unit])


def _handle_build_event(event: dict) -> None:
    if "stream" in event and event["stream"]:
        sys.stdout.write(event["stream"])
        sys.stdout.flush()
        return
    if "status" in event and event["status"]:
        status = event["status"]
        if "progress" in event:
            status = f"{status} {event['progress']}"
        print(status, flush=True)
        return
    if "error" in event and event["error"]:
        raise SystemExit(event["error"])


def _print_build_logs(logs) -> None:
    buffer = ""
    for chunk in logs:
        if isinstance(chunk, dict):
            _handle_build_event(chunk)
            continue

        if isinstance(chunk, bytes):
            text = chunk.decode("utf-8", errors="replace")
        elif isinstance(chunk, str):
            text = chunk
        else:
            continue

        buffer += text
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip("\r")
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                print(line, flush=True)
                continue
            if isinstance(event, dict):
                _handle_build_event(event)


def _device_requests(gpus: str) -> list | None:
    if DeviceRequest is None:
        return None
    if gpus == "all":
        return [DeviceRequest(count=-1, capabilities=[["gpu"]])]
    if gpus.startswith("device="):
        gpus = gpus.split("=", 1)[1]
    ids = [s.strip() for s in gpus.split(",") if s.strip()]
    if ids and all(i.isdigit() for i in ids):
        return [DeviceRequest(device_ids=ids, capabilities=[["gpu"]])]
    if gpus.isdigit():
        return [DeviceRequest(count=int(gpus), capabilities=[["gpu"]])]
    return [DeviceRequest(count=-1, capabilities=[["gpu"]])]


def _build_image(
    *,
    client,
    image: str,
    backend: str,
    uid: int | None = None,
    gid: int | None = None,
    no_cache: bool = False,
    pull: bool = False,
    dry_run: bool = False,
) -> None:
    root = _repo_root()
    assets = _ensure_assets(root, backend)
    dockerfile = assets["dockerfile"]

    default_uid, default_gid = _default_uid_gid()
    uid = default_uid if uid is None else uid
    gid = default_gid if gid is None else gid

    try:
        dockerfile_arg = dockerfile.relative_to(root).as_posix()
    except ValueError:
        dockerfile_arg = dockerfile.as_posix()

    print(f"[docker] build image={image} dockerfile={dockerfile_arg} context={root}")
    if dry_run:
        return

    _, logs = client.images.build(
        path=str(root),
        dockerfile=dockerfile_arg,
        tag=image,
        buildargs={"UID": str(uid), "GID": str(gid)},
        nocache=no_cache,
        pull=pull,
        rm=True,
    )
    _print_build_logs(logs)


def _build(args: argparse.Namespace) -> None:
    client = _docker_client()
    backend = _resolve_backend(profile=None, backend=args.backend)
    image = args.image
    if image == DEFAULT_IMAGE:
        image = _default_image_tag(backend)
    _build_image(
        client=client,
        image=image,
        backend=backend,
        uid=args.uid,
        gid=args.gid,
        no_cache=args.no_cache,
        pull=args.pull,
        dry_run=args.dry_run,
    )


def _remove_container(client, name: str) -> None:
    try:
        container = client.containers.get(name)
        container.remove(force=True)
    except NotFound:
        return
    except DockerException as exc:
        raise SystemExit(f"Failed to remove container '{name}': {exc}")


def _run_container(args: argparse.Namespace) -> None:
    client = _docker_client()
    backend = _resolve_backend(profile=args.profile, backend=args.backend)
    image = args.image
    if image == DEFAULT_IMAGE:
        image = _default_image_tag(backend)

    root = _repo_root()
    assets = _ensure_assets(root, backend)
    workspace_dir = assets["workspace"]
    outputs_dir = Path(args.outputs).resolve() if args.outputs else (root / "artifacts" / "outputs").resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    src_dir = (root / "src").resolve()
    if not src_dir.is_dir():
        raise SystemExit(f"Missing src directory at {src_dir}")

    if args.build or not _image_exists(client, image):
        if not _image_exists(client, image):
            print(f"Image '{image}' not found; building it first.")
        _build_image(client=client, image=image, backend=backend, dry_run=args.dry_run)

    if args.dry_run:
        print(f"[docker] would run container name={args.name} image={image}")
        return

    _remove_container(client, args.name)

    envs = {
        "TRUE_USER": os.environ.get("USERNAME") or os.environ.get("USER") or "unknown",
        "ACTUAL_PWD": str(Path.cwd()),
        "WRITABLE_BASE": "/workspace/.cache/splits",
        "DATA_ROOT": "/workspace/.cache/datasets" if backend == "ultralytics" else "/workspace",
    }
    envs.update(_windows_disk_env())
    if args.darknet_ref:
        envs["TARGET_REF"] = args.darknet_ref

    if args.env:
        for item in args.env:
            if "=" not in item:
                raise SystemExit(f"Invalid --env value (expected KEY=VALUE): {item}")
            k, v = item.split("=", 1)
            envs[k] = v

    train_cmd = [
        "python",
        "-u",
        "-m",
        "yolobattle.model_training.train",
        "--profile",
        args.profile,
    ]
    if args.train_args:
        train_cmd.extend(args.train_args)
    train_cmd_str = _shell_join(train_cmd)

    if backend == "darknet":
        command = f"bash {BUILD_SCRIPT_NAME}; {train_cmd_str}"
    else:
        command = train_cmd_str
    shm_size = args.shm_size
    if shm_size is None and backend == "ultralytics":
        shm_size = "8g"
    shm_bytes = _parse_shm_size(shm_size) if shm_size else None

    src_target = "/ultralytics/src" if backend == "ultralytics" else "/opt/app/src"
    volumes = {
        str(workspace_dir): {"bind": "/workspace", "mode": "rw"},
        str(outputs_dir): {"bind": "/outputs", "mode": "rw"},
        str(src_dir): {"bind": src_target, "mode": "rw"},
    }

    device_requests = None if args.no_gpu else _device_requests(args.gpus)

    run_kwargs = dict(
        image=image,
        name=args.name,
        detach=True,
        remove=True,
        volumes=volumes,
        environment=envs,
        command=["bash", "-lc", command],
        device_requests=device_requests,
    )
    if shm_bytes is not None:
        run_kwargs["shm_size"] = shm_bytes

    container = client.containers.run(**run_kwargs)

    if args.follow_logs:
        for line in container.logs(stream=True, follow=True):
            try:
                sys.stdout.write(line.decode("utf-8", errors="replace"))
            except Exception:
                sys.stdout.write(str(line))


def _logs(args: argparse.Namespace) -> None:
    if getattr(args, "dry_run", False):
        print(f"[docker] would show logs for container name={args.name}")
        return
    client = _docker_client()
    try:
        container = client.containers.get(args.name)
    except NotFound:
        raise SystemExit(f"Container '{args.name}' not found")

    if args.follow:
        for line in container.logs(stream=True, follow=True):
            try:
                sys.stdout.write(line.decode("utf-8", errors="replace"))
            except Exception:
                sys.stdout.write(str(line))
    else:
        data = container.logs()
        try:
            sys.stdout.write(data.decode("utf-8", errors="replace"))
        except Exception:
            sys.stdout.write(str(data))


def _shell(args: argparse.Namespace) -> None:
    raise SystemExit(
        "Interactive shell is not supported without the Docker CLI. "
        "Use: docker exec -it <name> /bin/bash"
    )


def _stop(args: argparse.Namespace) -> None:
    client = _docker_client()
    _remove_container(client, args.name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build and run the yolobattle Docker workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build the yolobattle Docker image.")
    build.add_argument("--image", default="yolobattle-container", help="Docker image tag.")
    build.add_argument("--backend", default="darknet", help="Backend to build (darknet|ultralytics).")
    build.add_argument("--uid", type=int, default=None, help="UID to bake into image.")
    build.add_argument("--gid", type=int, default=None, help="GID to bake into image.")
    build.add_argument("--no-cache", action="store_true", help="Disable build cache.")
    build.add_argument("--pull", action="store_true", help="Always attempt to pull newer base image.")
    build.add_argument("--dry-run", action="store_true", help="Print actions without running.")
    build.set_defaults(func=_build)

    run = subparsers.add_parser("run", help="Run the yolobattle Docker container.")
    run.add_argument("--image", default="yolobattle-container", help="Docker image tag.")
    run.add_argument("--name", default="yolobattle", help="Container name.")
    run.add_argument("--profile", default="LegoGearsDarknet", help="Training profile name.")
    run.add_argument("--gpus", default="all", help="GPU selection (all|0,1|device=0,1).")
    run.add_argument("--backend", default=None, help="Override backend (darknet|ultralytics).")
    run.add_argument(
        "--shm-size",
        default=None,
        help="Shared memory size (e.g. 8g, 2g). Default: 8g for ultralytics, none for darknet.",
    )
    run.add_argument("--no-gpu", action="store_true", help="Disable GPU request.")
    run.add_argument("--build", action="store_true", help="Build image before running.")
    run.add_argument("--outputs", default=None, help="Host outputs directory.")
    run.add_argument("--darknet-ref", default=None, help="Darknet ref (branch/tag/commit).")
    run.add_argument("--env", action="append", help="Extra env var KEY=VALUE (repeatable).")
    run.add_argument("--no-follow", dest="follow_logs", action="store_false", help="Do not follow logs.")
    run.add_argument("--dry-run", action="store_true", help="Print actions without running.")
    run.add_argument("train_args", nargs=argparse.REMAINDER, help="Args passed to train module.")
    run.set_defaults(func=_run_container, follow_logs=True)

    logs = subparsers.add_parser("logs", help="Show container logs.")
    logs.add_argument("--name", default="yolobattle", help="Container name.")
    logs.add_argument("--follow", action="store_true", help="Follow logs.")
    logs.add_argument("--dry-run", action="store_true", help="Print actions without running.")
    logs.set_defaults(func=_logs)

    shell = subparsers.add_parser("shell", help="Open a shell inside the container.")
    shell.add_argument("--name", default="yolobattle", help="Container name.")
    shell.set_defaults(func=_shell)

    stop = subparsers.add_parser("stop", help="Stop and remove the container.")
    stop.add_argument("--name", default="yolobattle", help="Container name.")
    stop.set_defaults(func=_stop)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
