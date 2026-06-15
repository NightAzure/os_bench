#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

MODE="quick"
TRIALS=""
PORT=8000
WORKERS=4
HOST_LABEL="$(hostname)"
OUTPUT_DIR=""
RUN_SMOKE="yes"
PYTHON_BIN="${PYTHON:-python3}"
FASTAPI_THREAD_LIMIT="${FASTAPI_THREAD_LIMIT:-40}"
SERVER_CPUS=""
CLIENT_CPUS=""

usage() {
  cat <<EOF
Usage: bash scripts/run_all.sh [--smoke | --quick | --full] [options]

Modes:
  --smoke              Validate scripts and analysis only; no server benchmark.
  --quick              Smoke test, then PyTorch C0-C4 at c=10,20 with 1 trial.
  --full               Smoke test, then CSAI-aligned benchmark flow.

Options:
  --trials N           Override trials per benchmark cell.
  --port N             Uvicorn port (default: 8000).
  --workers N          Uvicorn workers (default: 4).
  --host-label LABEL   Host label stored in raw_data.csv.
  --fastapi-thread-limit N
                       AnyIO worker-thread limit for FastAPI sync endpoints.
  --server-cpus LIST   CPUs used by Uvicorn workers/master.
  --client-cpus LIST   CPUs used by wrk and monitor helpers.
  --output-dir DIR     Output directory (default: results/repro_<timestamp>).
  --skip-smoke         Run benchmarks without the smoke test.
  --help               Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) MODE="smoke"; shift ;;
    --quick) MODE="quick"; shift ;;
    --full) MODE="full"; shift ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --host-label) HOST_LABEL="$2"; shift 2 ;;
    --fastapi-thread-limit) FASTAPI_THREAD_LIMIT="$2"; shift 2 ;;
    --server-cpus) SERVER_CPUS="$2"; shift 2 ;;
    --client-cpus) CLIENT_CPUS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --skip-smoke) RUN_SMOKE="no"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[error] Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ "$MODE" == "smoke" ]]; then
  exec bash "$SCRIPT_DIR/smoke_test.sh"
fi

if [[ "$RUN_SMOKE" == "yes" ]]; then
  bash "$SCRIPT_DIR/smoke_test.sh"
fi

export FASTAPI_THREAD_LIMIT

TS="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$REPO_ROOT/results/repro_$TS"
fi
mkdir -p "$OUTPUT_DIR"

bash "$SCRIPT_DIR/record_env.sh" "$OUTPUT_DIR/env_snapshot.txt" >/dev/null 2>&1

run_matrix() {
  local -a cpu_args=()
  if [[ -n "$SERVER_CPUS" ]]; then cpu_args+=(--server-cpus "$SERVER_CPUS"); fi
  if [[ -n "$CLIENT_CPUS" ]]; then cpu_args+=(--client-cpus "$CLIENT_CPUS"); fi
  bash "$SCRIPT_DIR/run_targeted_matrix.sh" \
    --output-dir "$OUTPUT_DIR" \
    --append yes \
    --analyze no \
    --port "$PORT" \
    --workers "$WORKERS" \
    --host-label "$HOST_LABEL" \
    --fastapi-thread-limit "$FASTAPI_THREAD_LIMIT" \
    "${cpu_args[@]}" \
    "$@"
}

if [[ "$MODE" == "quick" ]]; then
  TRIALS="${TRIALS:-1}"
  run_matrix \
    --experiment quick_csai \
    --library pytorch \
    --configs C0,C1,C2,C3,C4 \
    --threads default,1 \
    --concurrency 10,20 \
    --trials "$TRIALS" \
    --warmup-secs 5 \
    --duration-secs 15
else
  TRIALS="${TRIALS:-15}"
  run_matrix \
    --experiment full_csai_pytorch \
    --library pytorch \
    --configs C0,C1,C2,C3,C4 \
    --threads default,2,1 \
    --concurrency 1,5,10,20 \
    --trials "$TRIALS" \
    --warmup-secs 30 \
    --duration-secs 120

  for lib in numpy sklearn; do
    run_matrix \
      --experiment "full_csai_${lib}" \
      --library "$lib" \
      --configs C0,C3 \
      --threads default,1 \
      --concurrency 20 \
      --trials "$TRIALS" \
      --warmup-secs 30 \
      --duration-secs 120
  done
fi

"$PYTHON_BIN" "$SCRIPT_DIR/analyze.py" --input "$OUTPUT_DIR/raw_data.csv" --output-dir "$OUTPUT_DIR"

echo ""
echo "[done] Reproducibility flow completed."
echo "[done] Outputs: $OUTPUT_DIR"
