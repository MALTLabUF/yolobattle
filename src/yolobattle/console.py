import pkgutil
import runpy
import sys


TOP_LEVEL_MODULES = {"docker", "apptainer"}


def _available_modules() -> tuple[set[str], set[str]]:
    model_modules: set[str] = set()
    try:
        from yolobattle import model_training as mt
        model_modules = {m.name for m in pkgutil.iter_modules(mt.__path__)}
    except Exception:
        model_modules = set()
    top_modules = set(TOP_LEVEL_MODULES)
    return top_modules, model_modules


def _usage(mods: set[str]) -> str:
    mods_list = "|".join(sorted(mods)) if mods else "train"
    return (
        "yolobattle (YOLO training/benchmark tools)\n\n"
        "Usage:\n"
        "  yolobattle -m <module> [<args>...]\n"
        "  yolobattle <module> [<args>...]\n\n"
        f"Available modules: {mods_list}\n"
    )


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    top_modules, model_modules = _available_modules()
    all_modules = top_modules | model_modules

    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(_usage(all_modules))
        return

    if argv[0] in {"-m", "--module"}:
        if len(argv) < 2:
            print("Missing module name.\n")
            print(_usage(all_modules))
            sys.exit(2)
        module = argv[1]
        rest = argv[2:]
    else:
        module = argv[0]
        rest = argv[1:]

    if module not in all_modules:
        print(f"Unknown module: {module}\n")
        print(_usage(all_modules))
        sys.exit(2)

    sys.argv = [f"yolobattle -m {module}"] + rest
    if module in top_modules:
        runpy.run_module(f"yolobattle.{module}", run_name="__main__")
    else:
        runpy.run_module(f"yolobattle.model_training.{module}", run_name="__main__")


if __name__ == "__main__":
    main()
