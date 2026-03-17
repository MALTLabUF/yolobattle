#!/usr/bin/env python3
from __future__ import annotations
  
import os
import re
from pathlib import Path
  
import pandas as pd
import matplotlib.pyplot as plt
  
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "artifacts" / "plot"))
from username_machine_pairings import get_machine
  
from plot_common import (
    git_repo_root,
    iter_benchmark_csvs,
    DEFAULT_FAIR_KEYS,
    keep_largest_fair_subset,
)
  
# ---------------------------------------------------------------------------
# Path resolution and filters
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
# Name normalization
# ---------------------------------------------------------------------------

def strip_cpu_frequency(cpu: str) -> str:
    if not cpu:
        return cpu
    s = str(cpu)
    # Remove trailing frequency like "@ 3.20GHz"
    s = re.sub(r"\s*@\s*\d+(?:\.\d+)?\s*GHz\b", "", s, flags=re.IGNORECASE)
    # Remove common suffixes
    s = re.sub(r"\s+Processor\s*$", "", s, flags=re.IGNORECASE)
    # Remove standalone "CPU" anywhere
    s = re.sub(r"\bCPU\b", "", s, flags=re.IGNORECASE)
    # Normalize spaces
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def normalize_cpu_name(cpu: str) -> str:
    if cpu is None:
        return "N/A"
    s = str(cpu).strip()
    if not s:
        return "N/A"
    s = strip_cpu_frequency(s)
    # If it already contains lowercase letters, assume it's human-formatted enough
    if re.search(r"[a-z]", s):
        return s
    tokens = s.split()
    preserve_upper = {"AMD", "EPYC", "APU", "GPU", "INTEL", "XEON", "GOLD", "SILVER", "PLATINUM"}
    out: list[str] = []
    for t in tokens:
        if re.fullmatch(r"v\d+", t, flags=re.IGNORECASE):
            out.append(t.lower())
            continue
        if re.fullmatch(r"\d+([A-Za-z]+)?", t):
            out.append(t)
            continue
        bare = re.sub(r"[^\w()/-]", "", t)
        if bare.upper() in preserve_upper:
            out.append(bare.upper())
            continue
        m = re.match(r"^([A-Z]+)(\([^)]+\))?$", t)
        if m:
            base = m.group(1).capitalize()
            suffix = m.group(2) or ""
            out.append(base + suffix)
        else:
            out.append(t.capitalize())
    return " ".join(out)


def normalize_gpu_name(gpu: str) -> str:
    if gpu is None:
        return "N/A"
    s = str(gpu).strip()
    if not s:
        return "N/A"
    # If it already contains lowercase letters, assume it's human-formatted
    if re.search(r"[a-z]", s):
        return s
    tokens = s.split()
    preserve_upper = {"NVIDIA", "AMD", "GPU", "GTX", "RTX", "GEFORCE", "RADEON", "TITAN", "QUADRO", "TESLA", "A100", "V100", "P100"}
    out: list[str] = []
    for t in tokens:
        if re.fullmatch(r"\d+([A-Za-z]+)?", t):
            out.append(t)
            continue
        bare = re.sub(r"[^\w()/-]", "", t)
        if bare.upper() in preserve_upper:
            out.append(bare.upper())
            continue
        m = re.match(r"^([A-Z]+)(\([^)]+\))?$", t)
        if m:
            base = m.group(1).capitalize()
            suffix = m.group(2) or ""
            out.append(base + suffix)
        else:
            out.append(t.capitalize())
    return " ".join(out)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
  
def _is_completed_run(run_dir: str) -> bool:
    log_path = os.path.join(run_dir, "training_output.log")
    if not os.path.isfile(log_path):
        print(f"  [INVALID] Missing training_output.log: {run_dir}")
        return False
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_content = f.read()
        if "Training iteration has reached max batch limit" not in log_content:
            print(f"  [INVALID] Training did not complete: {run_dir}")
            return False
    except Exception as e:
        print(f"  [WARN] Could not read {log_path}: {e}")
        return False
    return True
  
  
def load_and_compile_csvs(base: Path) -> pd.DataFrame:
    base_dirs = [str(base)]
    frames: list[pd.DataFrame] = []
  
    for csv_path in iter_benchmark_csvs(base_dirs):
        run_dir = os.path.dirname(csv_path)
        if not _is_completed_run(run_dir):
            continue
  
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
  
    agg_numeric_mean = df.groupby(AVERAGE_GROUP_KEYS, sort=False)[numeric_cols].mean()
    agg_numeric_min = df.groupby(AVERAGE_GROUP_KEYS, sort=False)[numeric_cols].min().add_suffix("_min")
    agg_numeric_max = df.groupby(AVERAGE_GROUP_KEYS, sort=False)[numeric_cols].max().add_suffix("_max")
    agg_other = df.groupby(AVERAGE_GROUP_KEYS, sort=False)[non_numeric_cols].first()
    agg_count = df.groupby(AVERAGE_GROUP_KEYS, sort=False).size().rename("Number of Runs")
  
    averaged = pd.concat([agg_other, agg_numeric_mean, agg_numeric_min, agg_numeric_max, agg_count], axis=1).reset_index()
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
  
  
def plot_benchmark_time_bar(averaged: pd.DataFrame, out_dir: Path) -> None:
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
  
    # Normalize and build label: GPU on first line, CPU on second
    averaged["_gpu_normalized"] = averaged["GPU Name"].apply(normalize_gpu_name)
    averaged["_cpu_normalized"] = averaged["CPU Name"].apply(normalize_cpu_name)
    averaged["_label"] = averaged.apply(
        lambda r: f"{r['_gpu_normalized']}\n{r['_cpu_normalized']}", axis=1
    )
  
    # Keep only usable rows
    plot_df = averaged[["_label", BENCHMARK_TIME_COL, "_color", "_machine"]].dropna(
        subset=[BENCHMARK_TIME_COL]
    )
    if plot_df.empty:
        print("  [WARN] No rows to plot after filtering.")
        return
  
    # Sort greatest to least
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
            fontsize=12,
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
    ax.legend(handles=legend_handles, title="Machine", loc="upper right", fontsize=12, title_fontsize=13)
  
    ax.set_xlabel("")
    ax.set_ylabel("Benchmark Time (s)", fontsize=14)
  
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["_label"], rotation=45, ha="right", fontsize=11)
    ax.tick_params(axis='y', labelsize=12)
  
    plt.tight_layout()
  
    out_path = out_dir / "plot_common_benchmark_time.pdf"
    fig.savefig(out_path)
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
    print()
    print(f"Compiled {len(compiled)} rows from benchmarks with "
          f"{VAL_FRACTION_COL}={VAL_FRACTION_TARGET}\n")
  
    # Fairness filter: within each CPU/GPU group, keep only the most common fair-key
    compiled = keep_largest_fair_subset(
        compiled,
        fair_keys=DEFAULT_FAIR_KEYS,
        group_cols=("CPU Name", "GPU Name"),
        key_col="_fair_key",
    )
  
    if compiled.empty:
        raise SystemExit("No rows left after fairness filtering.")
  
    averaged = average_by_cpu_gpu(compiled)
  
    benchmark_time_min = f"{BENCHMARK_TIME_COL}_min"
    benchmark_time_max = f"{BENCHMARK_TIME_COL}_max"
  
    if all(col in averaged.columns for col in [BENCHMARK_TIME_COL, benchmark_time_min, benchmark_time_max, "Number of Runs"]):
        print("\nResults:")
        for _, row in averaged.iterrows():
            print(f"\nCPU: {row['CPU Name']}")
            print(f"GPU: {row['GPU Name']}")
            print(f"Benchmark Time (s): {row[benchmark_time_min]:.2f} / {row[BENCHMARK_TIME_COL]:.2f} / {row[benchmark_time_max]:.2f} (min / avg / max)")
            print(f"Number of Runs: {int(row['Number of Runs'])}")
  
    csv_out = script_dir / "plot_common_averaged.csv"
    averaged.to_csv(csv_out, index=False)
    print(f"\nSaved averaged results to: {csv_out}")
  
    print("\nGenerating plot ...")
    plot_benchmark_time_bar(averaged, script_dir)
  
  
if __name__ == "__main__":
    main()