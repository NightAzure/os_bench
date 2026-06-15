#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$REPO_ROOT/results/smoke"
RAW="$OUT_DIR/raw_data.csv"
PYTHON_BIN="${PYTHON:-python3}"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

for script in "$SCRIPT_DIR"/*.sh; do
  bash -n "$script"
done
"$PYTHON_BIN" -m py_compile "$SCRIPT_DIR/analyze.py"
bash "$SCRIPT_DIR/run_targeted_matrix.sh" --help | grep -q -- "--server-cpus"
bash "$SCRIPT_DIR/run_all.sh" --help | grep -q -- "--client-cpus"

"$PYTHON_BIN" - "$RAW" <<'PY'
import csv
import sys
from pathlib import Path

out = Path(sys.argv[1])
fields = [
    "host_label", "library", "config", "thread_mode", "thread_setting",
    "concurrency", "trial", "p50_ms", "p90_ms", "p99_ms", "rps",
    "invol_csw_per_s", "cpu_migrations_per_s", "notes",
]
rows = []
configs = ["C0", "C1", "C2", "C3", "C4"]
for c in [10, 20]:
    for cfg_i, cfg in enumerate(configs):
        for mode, setting, mode_penalty in [
            ("parallel", "default", 80),
            ("limited", "2", 40),
            ("serial", "1", 0),
        ]:
            for trial in range(1, 4):
                base = 90 + c * 2 + trial
                cfg_penalty = {"C0": 0, "C1": 35, "C2": 5, "C3": 55, "C4": 20}[cfg]
                p99 = base + cfg_penalty + mode_penalty
                rows.append({
                    "host_label": "smoke",
                    "library": "pytorch",
                    "config": cfg,
                    "thread_mode": mode,
                    "thread_setting": setting,
                    "concurrency": c,
                    "trial": trial,
                    "p50_ms": p99 * 0.45,
                    "p90_ms": p99 * 0.75,
                    "p99_ms": p99,
                    "rps": 5000 / p99,
                    "invol_csw_per_s": 1 + cfg_i + (mode != "serial") * 2,
                    "cpu_migrations_per_s": {"C0": 6, "C1": 2, "C2": 6, "C3": 0, "C4": 1}[cfg],
                    "notes": "",
                })

for lib in ["numpy", "sklearn"]:
    for cfg in ["C0", "C3"]:
        for mode, setting, mode_penalty in [("parallel", "default", 60), ("serial", "1", 0)]:
            for trial in range(1, 4):
                p99 = 120 + trial + (cfg == "C3") * 45 + mode_penalty
                rows.append({
                    "host_label": "smoke",
                    "library": lib,
                    "config": cfg,
                    "thread_mode": mode,
                    "thread_setting": setting,
                    "concurrency": 20,
                    "trial": trial,
                    "p50_ms": p99 * 0.45,
                    "p90_ms": p99 * 0.75,
                    "p99_ms": p99,
                    "rps": 5000 / p99,
                    "invol_csw_per_s": 2,
                    "cpu_migrations_per_s": 1,
                    "notes": "",
                })

with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
PY

"$PYTHON_BIN" "$SCRIPT_DIR/analyze.py" --input "$RAW" --output-dir "$OUT_DIR" >/dev/null

for artifact in \
  fig1_affinity_spectrum_p99.png \
  fig2_p99_vs_concurrency.png \
  fig3_thread_interaction.png \
  fig5_throughput.png \
  fig6_invol_ctx.png \
  fig7_cross_library.png \
  fig8_cpu_migrations.png \
  summary_stats.txt; do
  [[ -s "$OUT_DIR/$artifact" ]] || {
    echo "[error] Missing smoke artifact: $artifact"
    exit 1
  }
done

echo "[ok] Smoke test passed. Artifacts: $OUT_DIR"
