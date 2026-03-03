#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, List
import hashlib
import re

import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "artifacts" / "plot"))
from username_machine_pairings import get_machine

# Preferred YOLO ordering for all plots
PREFERRED_YOLO_ORDER = [
    "yolov3-tiny",
    "yolov3-tiny-3l",
    "yolov3",
    "yolov4-tiny",
    "yolov4-tiny-3l",
    "yolov4",
    "yolov7-tiny",
    "yolov7",

    # YOLO11 family (size ordering)
    "yolo11n",
    "yolo11s",
    "yolo11m",
]

# Default fairness keys (shared across scripts)
DEFAULT_FAIR_KEYS: tuple[str, ...] = (
    "CPU Name",
    "CPU Threads Used",
    "GPU Name",
    "Input Width",
    "Input Height",
    "Batch Size",
    "Subdivisions",
)


def get_row_value_str(row: pd.Series, col: str) -> Optional[str]:
    v = row.get(col)
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return s if s else None


def make_fair_key(
    row: pd.Series,
    *,
    fair_keys: Sequence[str] = DEFAULT_FAIR_KEYS,
    drop_keys: Sequence[str] = (),
) -> Optional[Tuple[str, ...]]:
    drop = set(drop_keys)
    parts = []
    for k in fair_keys:
        if k in drop:
            continue
        v = get_row_value_str(row, k)
        if v is None:
            return None
        parts.append(v)
    return tuple(parts)


def keep_largest_fair_subset(
    df: pd.DataFrame,
    *,
    fair_keys: Sequence[str] = DEFAULT_FAIR_KEYS,
    drop_keys: Sequence[str] = (),
    group_cols: Sequence[str] = (),
    key_col: str = "_fair_key",
) -> pd.DataFrame:
    d = df.copy()

    d[key_col] = d.apply(lambda r: make_fair_key(r, fair_keys=fair_keys, drop_keys=drop_keys), axis=1)
    d = d[d[key_col].notna()].copy()
    if d.empty:
        return d

    if not group_cols:
        best_key = d[key_col].value_counts().idxmax()
        return d[d[key_col] == best_key].copy()

    def _pick_block(g: pd.DataFrame) -> pd.DataFrame:
        vc = g[key_col].value_counts()
        if vc.empty:
            return g.iloc[0:0]
        best = vc.idxmax()
        return g[g[key_col] == best]

    out = d.groupby(list(group_cols), sort=False, group_keys=False).apply(_pick_block)
    return out.copy()


def filter_equal_cpu_threads_used(df: pd.DataFrame) -> pd.DataFrame:
    if "CPU Threads Used" not in df.columns:
        return df.iloc[0:0].copy()
    d = df.copy()
    d["CPU Threads Used"] = d["CPU Threads Used"].astype(str).str.strip()
    return d[d["CPU Threads Used"] != ""].copy()


def get_ordered_yolos(present_yolos) -> list[str]:
    present_set = set(present_yolos)
    ordered = [y for y in PREFERRED_YOLO_ORDER if y in present_set]
    extras = sorted(y for y in present_set if y not in ordered)
    return ordered + extras


def git_repo_root() -> Path:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return Path(out)
    except Exception:
        return Path.cwd()


def find_valid_file(run_dir: str, max_up: int = 5) -> str | None:
    run_path = Path(run_dir).resolve()
    parents = run_path.parents
    for up in range(max_up + 1):
        if up == 0:
            p = run_path
        else:
            if up - 1 >= len(parents):
                break
            p = parents[up - 1]
        candidate = p / "valid.txt"
        if candidate.is_file():
            return str(candidate)
    return None


def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def valid_basename_signature(path: str) -> tuple[str, ...]:
    names: list[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(Path(line).name)
    return tuple(names)


def normalize_dataset_name(profile: str, csv_path: str = "") -> str:
    dataset = "unknown"
    profile = str(profile) if profile is not None else ""
    lower_profile = profile.lower()
    lower_path = csv_path.lower() if csv_path else ""

    if "legogears" in lower_profile or "legogears" in lower_path:
        return "LegoGears"
    if "fisheyetraffic" in lower_profile or "fisheyetraffic" in lower_path:
        if "jpg" in lower_profile or "jpg" in lower_path:
            return "FisheyeTrafficJPG"
        return "FisheyeTraffic"
    if "leather" in lower_profile or "leather" in lower_path:
        return "Leather"
    if "cubes" in lower_profile or "cubes" in lower_path:
        return "Cubes"

    if profile:
        m = re.match(r"^([A-Za-z]+)", profile)
        if m:
            return m.group(1)

    return dataset


def iter_benchmark_csvs(base_dirs: List[str]) -> Iterable[str]:
    for base_dir in base_dirs:
        for root, _dirs, files in os.walk(base_dir):
            for f in files:
                if not f.endswith(".csv"):
                    continue
                if "benchmark__" not in f:
                    continue
                if 'val80' in f:
                    continue
                yield os.path.join(root, f)


def infer_dataset_name_from_csv(csv_path: str) -> str:
    try:
        df0 = pd.read_csv(csv_path, nrows=1)
    except Exception:
        df0 = pd.read_csv(csv_path, nrows=1, engine="python")

    for col in ["Profile", "Dataset", "Data Profile"]:
        if col in df0.columns:
            val = str(df0.iloc[0][col])
            if val and val.strip():
                return normalize_dataset_name(val, csv_path)

    if "Backend" in df0.columns:
        val = str(df0.iloc[0]["Backend"])
        if val and val.strip():
            return val.strip()

    d1 = os.path.dirname(csv_path)
    d2 = os.path.dirname(d1)
    d3 = os.path.dirname(d2)
    guess = os.path.basename(d3)
    return guess or "UnknownDataset"


def infer_input_resolution_from_csv(csv_path: str) -> str:
    try:
        df0 = pd.read_csv(csv_path, nrows=1)
    except Exception:
        df0 = pd.read_csv(csv_path, nrows=1, engine="python")

    if "Input Size" in df0.columns:
        val = str(df0.iloc[0]["Input Size"])
        if val and val.strip():
            return val.strip()

    w = None
    h = None
    if "Input Width" in df0.columns:
        w = pd.to_numeric(df0.iloc[0]["Input Width"], errors="coerce")
    if "Input Height" in df0.columns:
        h = pd.to_numeric(df0.iloc[0]["Input Height"], errors="coerce")

    if w is not None and not pd.isna(w) and h is not None and not pd.isna(h):
        return f"{int(w)}x{int(h)}"

    return "unknown_res"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

VAL_FRACTION_TARGET = 0.15
VAL_FRACTION_COL = "Val Fraction"
RUNS_DIR_NAME = "runs-yolobattle"
OUTPUTS_SUBPATH = Path("outputs") / "LegoGearsDarknet" / "yolov7-tiny"
BENCHMARK_TIME_COL = "Benchmark Time (s)"


def find_runs_dir() -> Path:
    repo_root = git_repo_root()
    candidate = repo_root.parent / RUNS_DIR_NAME
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"Could not find '{RUNS_DIR_NAME}' next to repo root '{repo_root}'. "
        f"Expected: {candidate}"
    )


def find_benchmark_base() -> Path:
    runs_dir = find_runs_dir()
    base = runs_dir / OUTPUTS_SUBPATH
    if not base.is_dir():
        raise FileNotFoundError(
            f"Expected benchmark base directory does not exist: {base}"
        )
    return base


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_compile_csvs(base: Path) -> pd.DataFrame:
    base_dirs = [str(base)]
    frames: list[pd.DataFrame] = []

    for csv_path in iter_benchmark_csvs(base_dirs):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            try:
                df = pd.read_csv(csv_path, engine="python")
            except Exception as exc:
                print(f"  [WARN] Could not read {csv_path}: {exc}")
                continue

        if VAL_FRACTION_COL in df.columns:
            df[VAL_FRACTION_COL] = pd.to_numeric(df[VAL_FRACTION_COL], errors="coerce")
            df = df[df[VAL_FRACTION_COL] == VAL_FRACTION_TARGET].copy()
        else:
            print(f"  [WARN] '{VAL_FRACTION_COL}' column missing in {csv_path}, skipping.")
            continue

        if df.empty:
            continue

        frames.append(df)

    if not frames:
        raise ValueError(
            f"No rows with {VAL_FRACTION_COL}={VAL_FRACTION_TARGET} found under {base}"
        )

    compiled = pd.concat(frames, ignore_index=True)
    return compiled


# ---------------------------------------------------------------------------
# Averaging
# ---------------------------------------------------------------------------

AVERAGE_GROUP_KEYS = ["CPU Name", "GPU Name"]


def average_by_cpu_gpu(df: pd.DataFrame) -> pd.DataFrame:
    for k in AVERAGE_GROUP_KEYS:
        if k not in df.columns:
            raise KeyError(f"Required column '{k}' not found in compiled data.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [
        c for c in df.columns
        if c not in numeric_cols and c not in AVERAGE_GROUP_KEYS
    ]

    agg_numeric = df.groupby(AVERAGE_GROUP_KEYS, sort=False)[numeric_cols].mean()
    agg_other = df.groupby(AVERAGE_GROUP_KEYS, sort=False)[non_numeric_cols].first()

    averaged = pd.concat([agg_other, agg_numeric], axis=1).reset_index()
    return averaged


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

MACHINE_COLORS = {
    "HiPerGator (UF)": "blue",
    "Afton (UVA)":     "orange",
    "MALTLab":         "green",
    "Personal Machine":"gray",
}


def plot_benchmark_time(averaged: pd.DataFrame, out_dir: Path) -> None:
    if BENCHMARK_TIME_COL not in averaged.columns:
        print(f"  [WARN] '{BENCHMARK_TIME_COL}' column not found, skipping plot.")
        return

    averaged = averaged.copy()

    # Extract username from Working Dir (e.g. /home/username/... or /sfs/.../username/...)
    def infer_username(row: pd.Series) -> str:
        wd = str(row.get("Working Dir", ""))
        # Try /home/username or /sfs/.../username pattern
        m = re.search(r"/(?:home|users?)/([^/]+)", wd)
        if m:
            return m.group(1)
        # Fallback: last path component before project folder
        parts = [p for p in wd.split("/") if p]
        if parts:
            return parts[-2] if len(parts) >= 2 else parts[-1]
        return ""

    averaged["_username"] = averaged.apply(infer_username, axis=1)
    averaged["_machine"] = averaged["_username"].apply(get_machine)
    averaged["_color"]   = averaged["_machine"].map(MACHINE_COLORS).fillna("gray")

    # Build label: GPU on first line, CPU on second
    averaged["_label"] = averaged.apply(
        lambda r: f"{r['GPU Name']}\n{r['CPU Name']}", axis=1
    )

    # Sort greatest to least
    plot_df = averaged[["_label", BENCHMARK_TIME_COL, "_color", "_machine"]].dropna(
        subset=[BENCHMARK_TIME_COL]
    )
    plot_df = plot_df.sort_values(BENCHMARK_TIME_COL, ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(max(10, len(plot_df) * 1.6), 7))

    bars = ax.bar(
        plot_df["_label"],
        plot_df[BENCHMARK_TIME_COL],
        color=plot_df["_color"],
        edgecolor="white",
    )

    # Value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Legend
    seen = {}
    for machine, color in MACHINE_COLORS.items():
        if machine in plot_df["_machine"].values:
            seen[machine] = color
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=m)
        for m, c in seen.items()
    ]
    ax.legend(handles=legend_handles, title="Machine", loc="upper right")

    ax.set_title("yolov7-tiny — Average Benchmark Time by CPU / GPU", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Benchmark Time (s)", fontsize=11)

    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["_label"], rotation=45, ha="right", fontsize=8)

    plt.tight_layout()

    out_path = out_dir / "plot_common_benchmark_time.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir = Path(__file__).parent

    print(f"Looking for '{RUNS_DIR_NAME}' ...")
    base = find_benchmark_base()
    print(f"Benchmark base: {base}\n")

    print("Loading and compiling CSVs ...")
    compiled = load_and_compile_csvs(base)
    print(f"  Compiled {len(compiled)} rows from benchmarks with "
          f"{VAL_FRACTION_COL}={VAL_FRACTION_TARGET}\n")

    print("Averaging by CPU Name and GPU Name ...")
    averaged = average_by_cpu_gpu(compiled)
    print(averaged.to_string(index=False))

    csv_out = script_dir / "plot_common_averaged.csv"
    averaged.to_csv(csv_out, index=False)
    print(f"\nSaved averaged results to: {csv_out}")

    print("\nGenerating plot ...")
    plot_benchmark_time(averaged, script_dir)


if __name__ == "__main__":
    main()