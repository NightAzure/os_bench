#!/usr/bin/env python3
"""
Analysis and figure generation for the CSAI benchmark package.

Reads raw_data.csv produced by scripts/run_targeted_matrix.sh or scripts/run_all.sh
and generates the manuscript figures plus summary_stats.txt.

CSV columns:
  host_label, library, config, thread_mode, thread_setting, concurrency, trial,
  p50_ms, p90_ms, p99_ms, rps, invol_csw_per_s, cpu_migrations_per_s, notes

Statistical design:
  - median central tendency for right-skewed latency distributions
  - bootstrap percentile confidence intervals
  - Mann-Whitney U tests with Bonferroni correction versus C0
  - Cliff's delta effect sizes
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIGS = ["C0", "C1", "C2", "C3", "C4"]
CONFIG_LABELS = {
    "C0": "C0: Linux default\n(soft affinity)",
    "C1": "C1: Coarse mask\n(first 2 logical CPUs)",
    "C2": "C2: nice -10\n(priority)",
    "C3": "C3: 1 worker per\nlogical CPU",
    "C4": "C4: Paired\nconstrained mask",
}
CONFIG_COLORS = {"C0": "#555555", "C1": "#1B9E77", "C2": "#D95F02", "C3": "#7570B3", "C4": "#E7298A"}
CONFIG_MARKERS = {"C0": "o", "C1": "s", "C2": "^", "C3": "D", "C4": "P"}

THREAD_COLORS  = {"serial": "#1565C0", "parallel": "#B71C1C"}
THREAD_HATCHES = {"serial": "", "parallel": "///"}
LIBRARY_COLORS = {"pytorch": "#EF6C00", "numpy": "#1565C0", "sklearn": "#2E7D32"}
LIBRARY_LABELS = {
    "pytorch": "Var. A\n(PyTorch)",
    "numpy":   "Var. B\n(numpy/\nOpenBLAS)",
    "sklearn":  "Var. C\n(scikit-learn)",
}

N_BOOTSTRAP    = 10_000
ALPHA_NOMINAL  = 0.05
N_COMPARISONS  = len(CONFIGS) - 1
ALPHA_BONF     = ALPHA_NOMINAL / N_COMPARISONS

# Cliff's delta magnitude thresholds (Romano et al., 2006)
CLIFFS_THRESHOLDS = [(0.474, "large"), (0.330, "medium"), (0.147, "small"), (0.0, "negligible")]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "host_label", "library", "config", "thread_mode", "thread_setting",
        "concurrency", "trial", "p50_ms", "p90_ms", "p99_ms", "rps",
        "invol_csw_per_s", "cpu_migrations_per_s", "notes",
    }
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[error] Missing columns in CSV: {missing}\n"
                 f"Found columns: {list(df.columns)}")
    if df.empty:
        sys.exit("[error] CSV has no data rows")

    numeric_cols = [
        "concurrency", "trial", "p50_ms", "p90_ms", "p99_ms", "rps",
        "invol_csw_per_s", "cpu_migrations_per_s",
    ]
    for col in numeric_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = converted.isna()
        if bad.any():
            rows = ", ".join(str(i + 2) for i in df.index[bad][:10])
            sys.exit(f"[error] Malformed numeric values in column {col}; CSV rows: {rows}")
        df[col] = converted

    df["concurrency"] = df["concurrency"].astype(int)
    df["trial"] = df["trial"].astype(int)
    positive_cols = ["p50_ms", "p90_ms", "p99_ms", "rps"]
    for col in positive_cols:
        bad = df[col] <= 0
        if bad.any():
            rows = ", ".join(str(i + 2) for i in df.index[bad][:10])
            sys.exit(f"[error] Nonpositive values in column {col}; CSV rows: {rows}")
    nonnegative_cols = ["invol_csw_per_s", "cpu_migrations_per_s"]
    for col in nonnegative_cols:
        bad = df[col] < 0
        if bad.any():
            rows = ", ".join(str(i + 2) for i in df.index[bad][:10])
            sys.exit(f"[error] Negative values in column {col}; CSV rows: {rows}")
    print(f"[info] Loaded {len(df)} rows.")
    print(f"[info] Libraries: {sorted(df['library'].unique())}")
    print(f"[info] Configs:   {sorted(df['config'].unique())}")
    print(f"[info] Thread modes: {sorted(df['thread_mode'].unique())}")
    return df


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def bootstrap_median_ci(data: np.ndarray, n_boot: int = N_BOOTSTRAP,
                         ci: float = 0.95) -> tuple[float, float, float]:
    """Return (median, lower_ci, upper_ci) using bootstrap percentile method."""
    rng = np.random.default_rng(42)
    medians = np.array([
        np.median(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(medians, (1 - ci) / 2 * 100)
    hi = np.percentile(medians, (1 - (1 - ci) / 2) * 100)
    return float(np.median(data)), float(lo), float(hi)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: positive = x tends to be larger than y."""
    n_gt = sum(1 for xi, yi in itertools.product(x, y) if xi > yi)
    n_lt = sum(1 for xi, yi in itertools.product(x, y) if xi < yi)
    return (n_gt - n_lt) / (len(x) * len(y))


def cliffs_magnitude(d: float) -> str:
    for threshold, label in CLIFFS_THRESHOLDS:
        if abs(d) >= threshold:
            return label
    return "negligible"


def cell(df, library=None, config=None, thread_mode=None,
          concurrency=None, metric="p99_ms") -> np.ndarray:
    """Filter dataframe and return metric values as array."""
    mask = pd.Series([True] * len(df), index=df.index)
    if library     is not None: mask &= df["library"]     == library
    if config      is not None: mask &= df["config"]      == config
    if thread_mode is not None: mask &= df["thread_mode"] == thread_mode
    if concurrency is not None: mask &= df["concurrency"] == concurrency
    return df.loc[mask, metric].values


# ---------------------------------------------------------------------------
# Fig. 2 - p99 vs concurrency (PyTorch, serialized)
# ---------------------------------------------------------------------------

def plot_fig1(df: pd.DataFrame, out: Path) -> None:
    lib = "pytorch"
    if lib not in df["library"].unique():
        print("[warn] fig1: no PyTorch data - skipping")
        return
    conc = 20 if 20 in df["concurrency"].unique() else max(df["concurrency"].unique())
    configs_present = [c for c in CONFIGS if len(cell(df, lib, c, concurrency=conc)) > 0]
    if not configs_present:
        print("[warn] fig1: no affinity-spectrum data - skipping")
        return

    thread_modes = [tm for tm in ["parallel", "serial"] if tm in df["thread_mode"].unique()]
    fig, axes = plt.subplots(len(thread_modes), 1, figsize=(7.0, 3.3 * len(thread_modes)), sharex=True)
    if len(thread_modes) == 1:
        axes = [axes]

    for ax, tm in zip(axes, thread_modes):
        xs = np.arange(len(configs_present))
        medians, lo_errs, hi_errs = [], [], []
        for cfg in configs_present:
            vals = cell(df, lib, cfg, tm, conc)
            if len(vals) == 0:
                medians.append(0); lo_errs.append(0); hi_errs.append(0)
                continue
            med, lo, hi = bootstrap_median_ci(vals)
            medians.append(med); lo_errs.append(med - lo); hi_errs.append(hi - med)

        bars = ax.bar(
            xs, medians, color=[CONFIG_COLORS[c] for c in configs_present],
            hatch=["//" if c == "C4" else "" for c in configs_present],
            yerr=[lo_errs, hi_errs], ecolor="black",
            error_kw={"elinewidth": 1.1, "capsize": 4}, alpha=0.88
        )
        for bar, cfg in zip(bars, configs_present):
            ax.scatter(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       marker=CONFIG_MARKERS[cfg], color="black", s=36, zorder=4)
        mode_label = "Runtime-default native threads" if tm == "parallel" else "Serialized native threads"
        ax.set_ylabel("Median p99 latency (ms)")
        ax.set_title(f"{mode_label}, c={conc}", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[-1].set_xticks(np.arange(len(configs_present)))
    axes[-1].set_xticklabels([CONFIG_LABELS[c] for c in configs_present], fontsize=8)
    fig.suptitle("Fig. 1  Affinity-Mask Spectrum (PyTorch)", fontsize=11)
    fig.tight_layout()
    _save(fig, out, "fig1_affinity_spectrum_p99")


def plot_fig2(df: pd.DataFrame, out: Path) -> None:
    lib = "pytorch"; tm = "serial"
    conc_levels = sorted(df["concurrency"].unique())
    configs_present = [c for c in CONFIGS if len(cell(df, lib, c, tm)) > 0]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for cfg in configs_present:
        xs, ys, lo_err, hi_err = [], [], [], []
        for c in conc_levels:
            vals = cell(df, lib, cfg, tm, c)
            if len(vals) == 0:
                continue
            med, lo, hi = bootstrap_median_ci(vals)
            xs.append(c); ys.append(med)
            lo_err.append(med - lo); hi_err.append(hi - med)
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=[lo_err, hi_err],
                    label=CONFIG_LABELS[cfg],
                    color=CONFIG_COLORS[cfg],
                    marker=CONFIG_MARKERS[cfg],
                    linewidth=1.8, markersize=6,
                    capsize=4, elinewidth=1)

    ax.set_xlabel("Concurrent Clients", fontsize=11)
    ax.set_ylabel("p99 Latency (ms)", fontsize=11)
    ax.set_title("Fig. 2  p99 Latency vs. Concurrency (RQ2)\n"
                 "PyTorch/Serialized, error bars = bootstrap 95% CI", fontsize=9)
    ax.set_xticks(conc_levels)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, out, "fig2_p99_vs_concurrency")


# ---------------------------------------------------------------------------
# Fig. 3 - Thread-interaction plot: C0/C3 x native-thread modes
# ---------------------------------------------------------------------------

def plot_fig3(df: pd.DataFrame, out: Path) -> None:
    """
    Primary result figure (RQ1 / H1).
    Grouped bar chart: x = config (C0, C3), groups = thread mode (serial, parallel).
    Shown at concurrency = 10 and 20 (side-by-side subplots).
    """
    target_concs = [c for c in [10, 20] if c in df["concurrency"].unique()]
    if not target_concs:
        target_concs = [max(df["concurrency"].unique())]

    lib = "pytorch"
    fig, axes = plt.subplots(1, len(target_concs), figsize=(5.5 * len(target_concs), 4.5),
                             sharey=True)
    if len(target_concs) == 1:
        axes = [axes]

    for ax, conc in zip(axes, target_concs):
        configs_used = ["C0", "C3"]
        thread_modes = [tm for tm in ["parallel", "limited", "serial"] if tm in df["thread_mode"].unique()]
        x = np.arange(len(configs_used))
        width = min(0.28, 0.8 / max(len(thread_modes), 1))

        for i, tm in enumerate(thread_modes):
            medians, lo_errs, hi_errs = [], [], []
            for cfg in configs_used:
                vals = cell(df, lib, cfg, tm, conc)
                if len(vals) == 0:
                    medians.append(0); lo_errs.append(0); hi_errs.append(0)
                    continue
                med, lo, hi = bootstrap_median_ci(vals)
                medians.append(med)
                lo_errs.append(med - lo)
                hi_errs.append(hi - med)

            offset = (i - (len(thread_modes) - 1) / 2) * width
            label = {
                "parallel": "runtime default",
                "limited": "2 native threads\n(exploratory)",
                "serial": "1 native thread",
            }.get(tm, tm)
            color = {
                "parallel": "#B71C1C",
                "limited": "#6A3D9A",
                "serial": "#1565C0",
            }.get(tm, "#555555")
            hatch = {
                "parallel": "///",
                "limited": "..",
                "serial": "",
            }.get(tm, "")
            bars = ax.bar(
                x + offset, medians, width,
                label=label,
                color=color,
                hatch=hatch,
                alpha=0.85,
                error_kw={"elinewidth": 1.2, "capsize": 4},
                yerr=[lo_errs, hi_errs],
                ecolor="black"
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{cfg}\n({'Linux default' if cfg == 'C0' else 'per-logical-CPU'})"
                             for cfg in configs_used], fontsize=9)
        ax.set_xlabel("OS Configuration", fontsize=10)
        ax.set_ylabel("Median p99 Latency (ms)", fontsize=10)
        ax.set_title(f"c = {conc} concurrent clients", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Fig. 3  Thread-Affinity Interaction (PyTorch)\n"
                 "Does PYTORCH_NUM_THREADS=1 change the direction of the C3 affinity effect?",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    _save(fig, out, "fig3_thread_interaction")


# ---------------------------------------------------------------------------
# Fig. 4 - Bootstrap CI strip at c=20 (PyTorch, serialized)
# ---------------------------------------------------------------------------

def plot_fig4(df: pd.DataFrame, out: Path) -> None:
    lib = "pytorch"; tm = "serial"; conc = 20
    configs_present = [c for c in CONFIGS
                       if len(cell(df, lib, c, tm, conc)) > 0]
    if not configs_present:
        print("[warn] fig4: no data for pytorch/serial/c=20 - skipping")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4))
    rng = np.random.default_rng(0)
    for i, cfg in enumerate(configs_present):
        vals = cell(df, lib, cfg, tm, conc)
        med, lo, hi = bootstrap_median_ci(vals)
        jit = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jit, vals,
                   color=CONFIG_COLORS[cfg], alpha=0.55, s=28, zorder=3)
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]],
                    fmt=CONFIG_MARKERS[cfg], color=CONFIG_COLORS[cfg],
                    markersize=10, capsize=8, linewidth=2, zorder=4,
                    label=f"{cfg}: {med:.1f} ms  [{lo:.1f}, {hi:.1f}]")

    ax.set_xticks(range(len(configs_present)))
    ax.set_xticklabels(configs_present)
    ax.set_xlabel("Configuration"); ax.set_ylabel("p99 Latency (ms)")
    ax.set_title("Fig. 4  Per-Trial p99 + Bootstrap 95% CI  (c=20, PyTorch, serialized)\n"
                 "Points: individual trials.  Marker: median.  Bars: 95% CI.", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, out, "fig4_bootstrap_ci_c20")


# ---------------------------------------------------------------------------
# Fig. 5 - Throughput vs concurrency (PyTorch, serialized)
# ---------------------------------------------------------------------------

def plot_fig5(df: pd.DataFrame, out: Path) -> None:
    lib = "pytorch"; tm = "serial"
    conc_levels = sorted(df["concurrency"].unique())
    configs_present = [c for c in CONFIGS if len(cell(df, lib, c, tm)) > 0]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for cfg in configs_present:
        xs, ys = [], []
        for c in conc_levels:
            vals = cell(df, lib, cfg, tm, c, metric="rps")
            if len(vals) == 0:
                continue
            xs.append(c); ys.append(float(np.median(vals)))
        if xs:
            ax.plot(xs, ys, label=CONFIG_LABELS[cfg],
                    color=CONFIG_COLORS[cfg], marker=CONFIG_MARKERS[cfg],
                    linewidth=1.8, markersize=6)

    ax.set_xlabel("Concurrent Clients", fontsize=11)
    ax.set_ylabel("Throughput (req/s)", fontsize=11)
    ax.set_title("Fig. 5  Throughput vs. Concurrency (PyTorch, serialized)", fontsize=9)
    ax.set_xticks(conc_levels)
    ax.legend(fontsize=8); ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, out, "fig5_throughput")


# ---------------------------------------------------------------------------
# Fig. 6 - Involuntary context switches at c=20 (PyTorch)
# ---------------------------------------------------------------------------

def plot_fig6(df: pd.DataFrame, out: Path) -> None:
    lib = "pytorch"; conc = 20
    configs_present = [c for c in CONFIGS
                       if len(cell(df, lib, c, "serial", conc)) > 0]
    if not configs_present:
        print("[warn] fig6: no data - skipping")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(configs_present))
    width = 0.35

    for i, tm in enumerate(["serial", "parallel"]):
        vals = [float(np.median(cell(df, lib, cfg, tm, conc, "invol_csw_per_s")))
                for cfg in configs_present]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width,
               label=f"{'Serialized' if tm=='serial' else 'Parallel'}",
               color=THREAD_COLORS[tm], alpha=0.85, hatch=THREAD_HATCHES[tm])

    ax.set_xticks(x); ax.set_xticklabels(configs_present)
    ax.set_xlabel("Configuration"); ax.set_ylabel("Involuntary ctx switches / process / s")
    ax.set_title("Fig. 6  Involuntary Context Switches at c=20 (PyTorch)\n"
                 "Higher = more preemption pressure under the active affinity mask", fontsize=9)
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, out, "fig6_invol_ctx")


# ---------------------------------------------------------------------------
# Fig. 7 - Cross-library comparison
# ---------------------------------------------------------------------------

def plot_fig8(df: pd.DataFrame, out: Path) -> None:
    lib = "pytorch"
    if lib not in df["library"].unique():
        print("[warn] fig8: no PyTorch data - skipping")
        return
    conc = 20 if 20 in df["concurrency"].unique() else max(df["concurrency"].unique())
    configs_present = [
        c for c in CONFIGS
        if len(cell(df, lib, c, "serial", conc)) > 0 or len(cell(df, lib, c, "parallel", conc)) > 0
    ]
    if not configs_present:
        print("[warn] fig8: no migration data - skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(configs_present))
    width = 0.35

    for i, tm in enumerate(["serial", "parallel"]):
        vals = [float(np.median(cell(df, lib, cfg, tm, conc, "cpu_migrations_per_s")))
                for cfg in configs_present]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width,
               label=f"{'Serialized' if tm == 'serial' else 'Runtime default'}",
               color=THREAD_COLORS[tm], alpha=0.85, hatch=THREAD_HATCHES[tm])

    ax.set_xticks(x)
    ax.set_xticklabels(configs_present)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("CPU migrations / worker / s")
    ax.set_title(f"Fig. 8  CPU Migrations per Worker at c={conc} (PyTorch)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, out, "fig8_cpu_migrations")


def plot_fig7(df: pd.DataFrame, out: Path) -> None:
    """
    RQ3 figure: grouped bars for C3 x (serial, parallel) across 3 library variants.
    Tests H6: does the thread-affinity pathology replicate across libraries?
    """
    conc = max(df["concurrency"].unique())
    libraries_present = [l for l in ["pytorch", "numpy", "sklearn"]
                         if l in df["library"].unique()]
    if len(libraries_present) < 2:
        print("[warn] fig7: fewer than 2 libraries in data - skipping cross-library plot")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(libraries_present))
    width = 0.35

    for i, tm in enumerate(["serial", "parallel"]):
        medians, lo_errs, hi_errs = [], [], []
        for lib in libraries_present:
            vals = cell(df, lib, "C3", tm, conc)
            if len(vals) == 0:
                medians.append(0); lo_errs.append(0); hi_errs.append(0)
                continue
            med, lo, hi = bootstrap_median_ci(vals)
            medians.append(med); lo_errs.append(med - lo); hi_errs.append(hi - med)

        offset = (i - 0.5) * width
        ax.bar(x + offset, medians, width,
               label=f"{'Serialized (BLAS_NUM_THREADS=1)' if tm == 'serial' else 'Parallel (default)'}",
               color=THREAD_COLORS[tm], hatch=THREAD_HATCHES[tm], alpha=0.85,
               yerr=[lo_errs, hi_errs], ecolor="black",
               error_kw={"elinewidth": 1.2, "capsize": 4})

    # Baseline C0/serial for each library (dotted reference line per library)
    for j, lib in enumerate(libraries_present):
        vals_base = cell(df, lib, "C0", "serial", conc)
        if len(vals_base) > 0:
            med_base = float(np.median(vals_base))
            ax.hlines(med_base, j - width, j + width,
                      colors=LIBRARY_COLORS[lib], linestyles="dotted",
                      linewidth=1.5, label=f"C0/serial baseline ({lib})" if j == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels([LIBRARY_LABELS[l] for l in libraries_present], fontsize=9)
    ax.set_xlabel("Library Variant (c=C3, concurrency={})".format(conc), fontsize=10)
    ax.set_ylabel("Median p99 Latency (ms)", fontsize=10)
    ax.set_title("Fig. 7  Cross-Library Thread-Affinity Interaction (RQ3 / H6)\n"
                 "Does C3 serial vs parallel effect replicate across BLAS-backed libraries?",
                 fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, out, "fig7_cross_library")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_tests(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append(f"STATISTICAL RESULTS  (Mann-Whitney U, Bonferroni alpha={ALPHA_BONF:.4f}, Cliff's delta)")
    lines.append("=" * 72)

    lib = "pytorch"
    for conc in sorted(df["concurrency"].unique()):
        lines.append(f"\n--- PyTorch, serialized thread mode, c={conc} ---")
        baseline = cell(df, lib, "C0", "serial", conc)
        if len(baseline) == 0:
            lines.append("  [no baseline data]")
            continue
        med_base = float(np.median(baseline))
        _, lo, hi = bootstrap_median_ci(baseline)
        lines.append(f"  C0 (baseline): median p99 = {med_base:.2f} ms  "
                     f"95%CI [{lo:.2f}, {hi:.2f}]  n={len(baseline)}")

        for cfg in ["C1", "C2", "C3", "C4"]:
            trt = cell(df, lib, cfg, "serial", conc)
            if len(trt) == 0:
                continue
            u, p = stats.mannwhitneyu(trt, baseline, alternative="two-sided")
            d = cliffs_delta(trt, baseline)
            p_corr = min(p * N_COMPARISONS, 1.0)
            med_trt = float(np.median(trt))
            reduction = med_base - med_trt
            sig = "*" if p_corr < ALPHA_NOMINAL else " "
            lines.append(
                f"  {cfg}: median={med_trt:.2f} ms  delta_ms={reduction:+.2f} "
                f"({reduction/max(med_base,1e-9)*100:+.1f}%)  "
                f"p_corr={p_corr:.4f}{sig}  cliffs_delta={d:.3f} ({cliffs_magnitude(d)})"
            )

    lines.append("\n" + "=" * 72)
    lines.append("RQ1 THREAD-INTERACTION TEST  (C3: parallel vs baseline C0/parallel)")
    lines.append("=" * 72)
    for conc in sorted(df["concurrency"].unique()):
        baseline_par = cell(df, lib, "C0", "parallel", conc)
        c3_par       = cell(df, lib, "C3", "parallel", conc)
        c3_ser       = cell(df, lib, "C3", "serial",   conc)
        if len(baseline_par) == 0 or len(c3_par) == 0:
            continue
        u1, p1 = stats.mannwhitneyu(c3_par, baseline_par, alternative="two-sided")
        d1 = cliffs_delta(c3_par, baseline_par)
        lines.append(
            f"  c={conc}  C3-parallel vs C0-parallel: "
            f"median {np.median(c3_par):.2f} vs {np.median(baseline_par):.2f} ms  "
            f"p={p1:.4f}  cliffs_delta={d1:.3f} ({cliffs_magnitude(d1)})"
        )
        if len(c3_ser) > 0:
            u2, p2 = stats.mannwhitneyu(c3_ser, baseline_par, alternative="two-sided")
            d2 = cliffs_delta(c3_ser, baseline_par)
            lines.append(
                f"  c={conc}  C3-serial   vs C0-parallel: "
                f"median {np.median(c3_ser):.2f} vs {np.median(baseline_par):.2f} ms  "
                f"p={p2:.4f}  cliffs_delta={d2:.3f} ({cliffs_magnitude(d2)})"
            )

    lines.append("\n" + "=" * 72)
    lines.append("RQ3 CROSS-LIBRARY TEST  (C3 serial vs parallel per library)")
    lines.append("=" * 72)
    conc = max(df["concurrency"].unique())
    for lib_name in ["pytorch", "numpy", "sklearn"]:
        if lib_name not in df["library"].unique():
            continue
        ser = cell(df, lib_name, "C3", "serial",   conc)
        par = cell(df, lib_name, "C3", "parallel", conc)
        if len(ser) == 0 or len(par) == 0:
            lines.append(f"  {lib_name}: insufficient data")
            continue
        u, p = stats.mannwhitneyu(par, ser, alternative="two-sided")
        d = cliffs_delta(par, ser)
        direction = "par > ser (pathology)" if np.median(par) > np.median(ser) else "ser > par (unexpected)"
        lines.append(
            f"  {lib_name} (c={conc}): ser={np.median(ser):.2f} ms  par={np.median(par):.2f} ms  "
            f"p={p:.4f}  cliffs_delta={d:.3f} ({cliffs_magnitude(d)})  [{direction}]"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, out_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[info] Saved: {path}")
    plt.close(fig)


def print_data_summary(df: pd.DataFrame) -> None:
    print("\n[data summary]")
    g = df.groupby(["library", "config", "thread_mode", "concurrency"])
    for name, grp in g:
        lib, cfg, tm, c = name
        med = float(np.median(grp["p99_ms"]))
        print(f"  {lib:12s}  {cfg}  {tm:8s}  c={c:2d}  "
              f"n={len(grp):2d}  median_p99={med:.2f} ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Statistical analysis for OS scheduling paper")
    parser.add_argument("--input",      required=True, help="Path to raw_data.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory for figures")
    args = parser.parse_args()

    csv_path = Path(args.input)
    out      = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        sys.exit(f"[error] Input file not found: {csv_path}")

    df = load_data(csv_path)
    print_data_summary(df)

    np.random.seed(42)

    print("\n[info] Generating figures...")
    plot_fig1(df, out)
    plot_fig2(df, out)
    plot_fig3(df, out)
    plot_fig4(df, out)
    plot_fig5(df, out)
    plot_fig6(df, out)
    plot_fig7(df, out)
    plot_fig8(df, out)

    print("\n[info] Running statistical tests...")
    stats_text = run_tests(df)
    stats_file = out / "summary_stats.txt"
    stats_file.write_text(stats_text, encoding="utf-8")
    print(stats_text)
    print(f"\n[info] Stats saved to: {stats_file}")

    print(f"\n[done] All outputs in: {out}")


if __name__ == "__main__":
    main()

