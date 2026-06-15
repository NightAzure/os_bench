#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${BLUE}>>> $* <<<${NC}"; }
die()       { log_error "$*"; exit 1; }

EXPERIMENT="targeted_matrix"
LIBRARY="pytorch"
CONFIGS="C0,C3,C4"
THREADS="default,1"
CONCURRENCY="20"
TRIALS=5
PORT=8000
WORKERS=4
WARMUP_SECS=15
DURATION_SECS=60
HOST_LABEL="$(hostname)"
OUT_DIR=""
DO_ANALYZE="yes"
APPEND="yes"
PYTHON_BIN="${PYTHON:-python3}"
FASTAPI_THREAD_LIMIT="${FASTAPI_THREAD_LIMIT:-40}"
SERVER_CPUS=""
CLIENT_CPUS=""

usage() {
  cat <<EOF
Usage: bash scripts/run_targeted_matrix.sh [options]

Options:
  --experiment NAME          Output name prefix (default: targeted_matrix)
  --library LIB              pytorch | numpy | sklearn  (default: pytorch)
  --configs LIST             Comma list, e.g. C0,C3,C4   (default: C0,C3,C4)
  --threads LIST             default,1,2,...             (default: default,1)
  --concurrency LIST         Comma list, e.g. 20,10      (default: 20)
  --trials N                 Trials per cell             (default: 5)
  --workers N                Uvicorn workers             (default: 4)
  --port N                   Port                        (default: 8000)
  --warmup-secs N            Warmup duration             (default: 15)
  --duration-secs N          Measurement duration        (default: 60)
  --host-label LABEL         Label written to CSV        (default: hostname)
  --fastapi-thread-limit N   AnyIO worker-thread limit   (default: 40)
  --server-cpus LIST         CPUs used by Uvicorn workers/master
  --client-cpus LIST         CPUs used by wrk and monitor helpers
  --output-dir DIR           Explicit output dir         (default: results/<experiment>_<ts>)
  --analyze yes|no           Run scripts/analyze.py      (default: yes)
  --append yes|no            Append to existing raw CSV  (default: yes)
  --help                     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment) EXPERIMENT="$2"; shift 2 ;;
    --library) LIBRARY="$2"; shift 2 ;;
    --configs) CONFIGS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --warmup-secs) WARMUP_SECS="$2"; shift 2 ;;
    --duration-secs) DURATION_SECS="$2"; shift 2 ;;
    --host-label) HOST_LABEL="$2"; shift 2 ;;
    --fastapi-thread-limit) FASTAPI_THREAD_LIMIT="$2"; shift 2 ;;
    --server-cpus) SERVER_CPUS="$2"; shift 2 ;;
    --client-cpus) CLIENT_CPUS="$2"; shift 2 ;;
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    --analyze) DO_ANALYZE="$2"; shift 2 ;;
    --append) APPEND="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

for tool in wrk uvicorn pidstat taskset mpstat perf curl ps lscpu shuf; do
  command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
"$PYTHON_BIN" --version >/dev/null 2>&1 || die "Python interpreter not found: $PYTHON_BIN"
export FASTAPI_THREAD_LIMIT
if [[ ",$CONFIGS," == *",C2," ]]; then
  command -v sudo >/dev/null 2>&1 || die "sudo not found; C2 requires sudo -n renice"
  sudo -n renice -n -10 -p $$ >/dev/null || die "C2 requires passwordless sudo renice; configure sudoers or omit C2"
  renice -n 0 -p $$ >/dev/null 2>&1 || sudo -n renice -n 0 -p $$ >/dev/null || true
fi

case "$LIBRARY" in
  pytorch) SERVER_MODULE="app.server_pytorch:app" ;;
  numpy) SERVER_MODULE="app.server_numpy:app" ;;
  sklearn) SERVER_MODULE="app.server_sklearn:app" ;;
  *) die "Unsupported library: $LIBRARY" ;;
esac

TS="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$REPO_ROOT/results/${EXPERIMENT}_$TS"
fi
mkdir -p "$OUT_DIR"
RAW_CSV="$OUT_DIR/raw_data.csv"
FAILED_CSV="$OUT_DIR/failed_trials.csv"

if [[ "$APPEND" != "yes" || ! -f "$RAW_CSV" ]]; then
  echo "host_label,library,config,thread_mode,thread_setting,concurrency,trial,p50_ms,p90_ms,p99_ms,rps,invol_csw_per_s,cpu_migrations_per_s,notes" > "$RAW_CSV"
fi
if [[ "$APPEND" != "yes" || ! -f "$FAILED_CSV" ]]; then
  echo "host_label,library,config,thread_mode,thread_setting,concurrency,trial,reason" > "$FAILED_CSV"
fi

SERVER_PID=""
WORKER_PIDS_ARR=()
ALLOWED_CPUS_ARR=()
PROCESS_CPUS_ARR=()
ONLINE_CPU_LIST=""
CLIENT_CPU_LIST=""
IFS=',' read -r -a THREAD_LIST <<< "$THREADS"
IFS=',' read -r -a CONC_LIST <<< "$CONCURRENCY"
IFS=',' read -r -a CONFIG_LIST <<< "$CONFIGS"

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  pkill -f "uvicorn ${SERVER_MODULE}" 2>/dev/null || true
}
trap cleanup EXIT

parse_wrk_output() {
  local wrk_file="$1"
  "$PYTHON_BIN" - "$wrk_file" <<'PYEOF'
import re, sys
content = open(sys.argv[1], encoding='utf-8', errors='ignore').read()

def to_ms(m):
    val, unit = float(m.group(1)), m.group(2)
    return val / 1000 if unit == 'us' else val * 1000 if unit == 's' else val

def find_pct(pct):
    pat = rf'\s+{pct}%\s+(\d+\.?\d*)(us|ms|s)'
    m = re.search(pat, content)
    return f"{to_ms(m):.3f}" if m else "0"

rps_m = re.search(r'Requests/sec:\s+([\d.]+)', content)
rps = rps_m.group(1) if rps_m else "0"
err_m = re.search(r'\[wrk\] done:\s+\d+\s+requests,\s+\d+\s+errors\s+\(([\d.]+)% error rate\)', content)
err = err_m.group(1) if err_m else "MISSING"
print(find_pct(50), find_pct(90), find_pct(99), rps, err)
PYEOF
}

csv_join() {
  local IFS=,
  echo "$*"
}

expand_cpu_list() {
  local spec="$1"
  "$PYTHON_BIN" - "$spec" <<'PYEOF'
import sys

cpus = []
for part in sys.argv[1].split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part:
        start, end = map(int, part.split("-", 1))
        cpus.extend(range(start, end + 1))
    else:
        cpus.append(int(part))
print(" ".join(str(cpu) for cpu in sorted(set(cpus))))
PYEOF
}

detect_allowed_cpus() {
  local line spec
  line="$(taskset -pc $$ 2>/dev/null || true)"
  spec="${line##*: }"
  if [[ -z "$spec" || "$spec" == "$line" ]]; then
    spec="$(lscpu -p=CPU | grep -v '^#' | paste -sd, -)"
  fi
  [[ -n "$spec" ]] || die "Could not determine allowed CPU list"
  read -r -a PROCESS_CPUS_ARR <<< "$(expand_cpu_list "$spec")"
  [[ ${#PROCESS_CPUS_ARR[@]} -gt 0 ]] || die "Process allowed CPU list is empty"

  if [[ -n "$SERVER_CPUS" ]]; then
    read -r -a ALLOWED_CPUS_ARR <<< "$(expand_cpu_list "$SERVER_CPUS")"
  else
    ALLOWED_CPUS_ARR=("${PROCESS_CPUS_ARR[@]}")
  fi
  [[ ${#ALLOWED_CPUS_ARR[@]} -gt 0 ]] || die "Allowed CPU list is empty"
  ONLINE_CPU_LIST="$(csv_join "${ALLOWED_CPUS_ARR[@]}")"

  if [[ -n "$CLIENT_CPUS" ]]; then
    local -a client_arr=()
    read -r -a client_arr <<< "$(expand_cpu_list "$CLIENT_CPUS")"
    CLIENT_CPU_LIST="$(csv_join "${client_arr[@]}")"
  fi

  "$PYTHON_BIN" - "$ONLINE_CPU_LIST" "$(csv_join "${PROCESS_CPUS_ARR[@]}")" "$CLIENT_CPU_LIST" <<'PYEOF'
import sys

server = {int(x) for x in sys.argv[1].split(",") if x}
allowed = {int(x) for x in sys.argv[2].split(",") if x}
client = {int(x) for x in sys.argv[3].split(",") if x}

bad_server = sorted(server - allowed)
bad_client = sorted(client - allowed)
overlap = sorted(server & client)
if bad_server:
    raise SystemExit(f"server CPUs not allowed by current affinity: {bad_server}")
if bad_client:
    raise SystemExit(f"client CPUs not allowed by current affinity: {bad_client}")
if client and overlap:
    raise SystemExit(f"server and client CPU pools must be disjoint; overlap={overlap}")
PYEOF

  if (( WORKERS > ${#ALLOWED_CPUS_ARR[@]} )); then
    die "workers=$WORKERS exceeds allowed logical CPUs=${#ALLOWED_CPUS_ARR[@]} ($ONLINE_CPU_LIST)"
  fi
}

run_on_client() {
  if [[ -n "$CLIENT_CPU_LIST" ]]; then
    taskset -c "$CLIENT_CPU_LIST" "$@"
  else
    "$@"
  fi
}

parse_pidstat() {
  local ps_file="$1"
  "$PYTHON_BIN" - "$ps_file" <<'PYEOF'
import json
import sys

try:
    data = json.load(open(sys.argv[1], encoding="utf-8", errors="ignore"))
except Exception:
    print("0")
    raise SystemExit

vals = []
for host in data.get("sysstat", {}).get("hosts", []):
    for stat in host.get("statistics", []):
        for proc in stat.get("processes", []):
            val = proc.get("nvcswch/s")
            if val is not None:
                vals.append(float(val))
print(f"{(sum(vals) / len(vals)):.2f}" if vals else "0")
PYEOF
}

parse_migrations() {
  local perf_file="$1"
  "$PYTHON_BIN" - "$perf_file" "$DURATION_SECS" "$WORKERS" <<'PYEOF'
import math
import sys

path = sys.argv[1]
duration = max(float(sys.argv[2]), 1.0)
workers = max(float(sys.argv[3]), 1.0)
migrations = None
try:
    for raw in open(path, encoding="utf-8", errors="ignore"):
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 3:
            continue
        event = parts[2]
        if event == "cpu-migrations":
            value = parts[0].replace(",", "")
            if value and value != "<not counted>":
                migrations = float(value)
except Exception:
    migrations = None

if migrations is None or math.isnan(migrations):
    print("NA")
else:
    print(f"{migrations / duration / workers:.2f}")
PYEOF
}

apply_thread_env() {
  local thread_setting="$1"
  unset PYTORCH_NUM_THREADS TORCH_NUM_INTEROP_THREADS OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS BLAS_NUM_THREADS 2>/dev/null || true
  export FASTAPI_THREAD_LIMIT
  if [[ "$thread_setting" == "default" ]]; then
    THREAD_MODE="parallel"
  else
    export PYTORCH_NUM_THREADS="$thread_setting"
    export TORCH_NUM_INTEROP_THREADS="1"
    export OMP_NUM_THREADS="$thread_setting"
    export OPENBLAS_NUM_THREADS="$thread_setting"
    export MKL_NUM_THREADS="$thread_setting"
    export BLAS_NUM_THREADS="$thread_setting"
    if [[ "$thread_setting" == "1" ]]; then
      THREAD_MODE="serial"
    else
      THREAD_MODE="limited"
    fi
  fi
}

check_workers_alive() {
  local label="$1"
  [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null || die "Server master died before $label"
  local alive=0
  for wpid in "${WORKER_PIDS_ARR[@]}"; do
    if kill -0 "$wpid" 2>/dev/null; then
      alive=$((alive+1))
    fi
  done
  (( alive == WORKERS )) || die "Expected $WORKERS live workers before $label, found $alive"
  run_on_client curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || die "Health check failed before $label"
}

discover_worker_pids() {
  local -a candidates=()
  local -a workers=()
  local pid args
  mapfile -t candidates < <(pgrep -P "$SERVER_PID" || true)
  for pid in "${candidates[@]}"; do
    args="$(ps -o args= -p "$pid" 2>/dev/null || true)"
    [[ -n "$args" ]] || continue
    case "$args" in
      *multiprocessing.resource_tracker*|*multiprocessing.semaphore_tracker*)
        continue
        ;;
    esac
    workers+=("$pid")
  done
  if (( ${#workers[@]} > 0 )); then
    printf "%s\n" "${workers[@]}"
  fi
}

describe_server_children() {
  local -a candidates=()
  mapfile -t candidates < <(pgrep -P "$SERVER_PID" || true)
  if (( ${#candidates[@]} == 0 )); then
    echo "no child processes"
  else
    ps -o pid=,ppid=,stat=,args= -p "$(IFS=,; echo "${candidates[*]}")" 2>/dev/null || true
  fi
}

start_server() {
  local thread_setting="$1"
  apply_thread_env "$thread_setting"
  cleanup
  cd "$REPO_ROOT"
  taskset -c "$ONLINE_CPU_LIST" uvicorn "$SERVER_MODULE" --host 127.0.0.1 --port "$PORT" --workers "$WORKERS" --log-level error > "$OUT_DIR/server_${LIBRARY}_${thread_setting}.log" 2>&1 &
  SERVER_PID=$!
  local attempts=0
  until run_on_client curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    sleep 1
    attempts=$((attempts+1))
    [[ $attempts -ge 60 ]] && die "Server did not start for $LIBRARY / thread=$thread_setting"
  done
  attempts=0
  while true; do
    mapfile -t WORKER_PIDS_ARR < <(discover_worker_pids)
    if (( ${#WORKER_PIDS_ARR[@]} == WORKERS )); then
      break
    fi
    sleep 1
    attempts=$((attempts+1))
    [[ $attempts -ge 20 ]] && break
  done
  [[ ${#WORKER_PIDS_ARR[@]} -gt 0 ]] || die "No worker PIDs found"
  if (( ${#WORKER_PIDS_ARR[@]} != WORKERS )); then
    die "Expected $WORKERS worker PIDs, found ${#WORKER_PIDS_ARR[@]}: ${WORKER_PIDS_ARR[*]}. Children: $(describe_server_children | tr '\n' '; ')"
  fi
  check_workers_alive "first trial"
}

apply_affinity() {
  local pid="$1"
  local mask="$2"
  taskset -a -cp "$mask" "$pid" >/dev/null || die "taskset failed for pid=$pid mask=$mask"
}

apply_nice() {
  local pid="$1"
  local nice_value="$2"
  if (( nice_value < 0 )); then
    sudo -n renice -n "$nice_value" -p "$pid" >/dev/null || die "sudo renice failed for pid=$pid nice=$nice_value; configure passwordless sudo or omit C2"
  else
    renice -n "$nice_value" -p "$pid" >/dev/null 2>&1 || sudo -n renice -n "$nice_value" -p "$pid" >/dev/null || die "renice failed for pid=$pid nice=$nice_value"
  fi
}

build_c4_masks() {
  "$PYTHON_BIN" - "$ONLINE_CPU_LIST" <<'PYEOF'
import collections
import subprocess
import sys

allowed = [int(x) for x in sys.argv[1].split(",") if x]
allowed_set = set(allowed)
groups = []
try:
    out = subprocess.check_output(
        ["lscpu", "-p=CPU,CORE,SOCKET,NODE"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    by_core = collections.defaultdict(list)
    for line in out.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        cpu = int(parts[0])
        if cpu not in allowed_set:
            continue
        key = tuple(parts[1:4])
        by_core[key].append(cpu)
    groups = [sorted(v)[:2] for v in by_core.values() if len(v) >= 2]
except Exception:
    groups = []

if not groups:
    groups = [allowed[i:i+2] for i in range(0, len(allowed) - 1, 2)]

for group in groups:
    if len(group) >= 2:
        print(",".join(str(cpu) for cpu in group[:2]))
PYEOF
}

apply_config() {
  local config="$1"
  for wpid in "${WORKER_PIDS_ARR[@]}"; do
    apply_affinity "$wpid" "$ONLINE_CPU_LIST"
    apply_nice "$wpid" 0
  done
  case "$config" in
    C0) ;;
    C1)
      (( ${#ALLOWED_CPUS_ARR[@]} >= 2 )) || die "C1 requires at least 2 allowed CPUs"
      local coarse_mask
      coarse_mask="$(csv_join "${ALLOWED_CPUS_ARR[@]:0:2}")"
      for wpid in "${WORKER_PIDS_ARR[@]}"; do
        apply_affinity "$wpid" "$coarse_mask"
      done
      ;;
    C2)
      for wpid in "${WORKER_PIDS_ARR[@]}"; do
        apply_nice "$wpid" -10
      done
      ;;
    C3)
      local core_idx=0
      for wpid in "${WORKER_PIDS_ARR[@]}"; do
        apply_affinity "$wpid" "${ALLOWED_CPUS_ARR[$core_idx]}"
        core_idx=$((core_idx+1))
      done
      ;;
    C4)
      local pair_idx=0
      local -a c4_masks=()
      mapfile -t c4_masks < <(build_c4_masks)
      (( ${#c4_masks[@]} > 0 )) || die "C4 requires at least one two-CPU sibling or adjacent mask within allowed CPUs"
      for wpid in "${WORKER_PIDS_ARR[@]}"; do
        apply_affinity "$wpid" "${c4_masks[$((pair_idx % ${#c4_masks[@]}))]}"
        pair_idx=$((pair_idx+1))
      done
      ;;
    *) die "Unsupported config: $config" ;;
  esac
}

capture_diag() {
  local tag="$1"
  {
    echo "=== date ==="
    date
    echo "=== host label ==="
    echo "$HOST_LABEL"
    echo "=== cpu pools ==="
    echo "process_allowed_cpus=$(csv_join "${PROCESS_CPUS_ARR[@]}")"
    echo "server_cpus=$ONLINE_CPU_LIST"
    echo "client_cpus=${CLIENT_CPU_LIST:-unset}"
    echo "=== lscpu ==="
    lscpu
    echo "=== ps -eLo ==="
    ps -eLo pid,ppid,tid,psr,pcpu,comm --sort=pid | awk -v pids="$(IFS=,; echo "${WORKER_PIDS_ARR[*]}")" '
      BEGIN { split(pids, a, ","); for (i in a) keep[a[i]]=1 }
      NR==1 || keep[$1] { print }
    '
    echo "=== taskset -a -pc ==="
    for wpid in "${WORKER_PIDS_ARR[@]}"; do taskset -a -pc "$wpid"; done
    echo "=== runtime thread pools ==="
    run_on_client curl -sf "http://127.0.0.1:$PORT/runtime" || true
  } > "$OUT_DIR/diag_${tag}.txt" 2>&1 || true
}

bash "$SCRIPT_DIR/record_env.sh" "$OUT_DIR/env_snapshot.txt" >/dev/null 2>&1
detect_allowed_cpus

log_step "Running $EXPERIMENT into $OUT_DIR"
log_info "Process allowed CPUs: $(csv_join "${PROCESS_CPUS_ARR[@]}")"
log_info "Server CPUs: $ONLINE_CPU_LIST"
if [[ -n "$CLIENT_CPU_LIST" ]]; then
  log_info "Client CPUs: $CLIENT_CPU_LIST"
else
  log_warn "Client CPUs not set; wrk and monitor helpers are not isolated from the server pool"
fi
mapfile -t SHUFFLED_THREADS < <(printf "%s\n" "${THREAD_LIST[@]}" | shuf)
log_info "Thread-setting order: ${SHUFFLED_THREADS[*]}"
for THREAD_SETTING in "${SHUFFLED_THREADS[@]}"; do
  start_server "$THREAD_SETTING"
  mapfile -t SHUFFLED_CONFIGS < <(printf "%s\n" "${CONFIG_LIST[@]}" | shuf)
  for CONFIG in "${SHUFFLED_CONFIGS[@]}"; do
    apply_config "$CONFIG"
    mapfile -t SHUFFLED_CONCURRENCY < <(printf "%s\n" "${CONC_LIST[@]}" | shuf)
    for C in "${SHUFFLED_CONCURRENCY[@]}"; do
      WRK_THREADS=$(( C < 2 ? 1 : 2 ))
      run_on_client wrk -t"$WRK_THREADS" -c"$C" -d"${WARMUP_SECS}s" --timeout 10s -s "$REPO_ROOT/workloads/mixed.lua" "http://127.0.0.1:$PORT" >/dev/null 2>&1 || die "Warmup failed for $LIBRARY/$CONFIG/thread=$THREAD_SETTING/c=$C"
      for TRIAL in $(seq 1 "$TRIALS"); do
        TAG="${LIBRARY}_${CONFIG}_${THREAD_SETTING}_c${C}_t${TRIAL}"
        WRK_FILE="$OUT_DIR/wrk_${TAG}.txt"
        PS_FILE="$OUT_DIR/pidstat_${TAG}.txt"
        MP_FILE="$OUT_DIR/mpstat_${TAG}.txt"
        PERF_FILE="$OUT_DIR/perf_${TAG}.txt"
        PID_LIST="$(IFS=,; echo "${WORKER_PIDS_ARR[*]}")"
        check_workers_alive "$TAG"
        capture_diag "$TAG"
        run_on_client pidstat -w -o JSON -p "$PID_LIST" 1 "$DURATION_SECS" > "$PS_FILE" 2>&1 &
        PS_PID=$!
        run_on_client mpstat -P ALL 1 "$DURATION_SECS" > "$MP_FILE" 2>&1 &
        MP_PID=$!
        run_on_client perf stat -x, -e context-switches,cpu-migrations -p "$PID_LIST" -- sleep "$DURATION_SECS" > "$PERF_FILE" 2>&1 &
        PERF_PID=$!
        WRK_STATUS=0
        run_on_client wrk -t"$WRK_THREADS" -c"$C" -d"${DURATION_SECS}s" --latency --timeout 10s -s "$REPO_ROOT/workloads/mixed.lua" "http://127.0.0.1:$PORT" > "$WRK_FILE" 2>&1 || WRK_STATUS=$?
        wait "$PERF_PID" || die "perf stat failed for $TAG; check perf permissions"
        kill "$PS_PID" "$MP_PID" 2>/dev/null || true
        wait "$PS_PID" 2>/dev/null || true
        wait "$MP_PID" 2>/dev/null || true
        read P50 P90 P99 RPS ERR_RATE < <(parse_wrk_output "$WRK_FILE")
        if [[ "$WRK_STATUS" -ne 0 || "$P99" == "0" || "$RPS" == "0" ]]; then
          log_warn "$TAG failed or produced empty wrk metrics; skipping CSV row"
          printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "$HOST_LABEL" "$LIBRARY" "$CONFIG" "$THREAD_MODE" "$THREAD_SETTING" "$C" "$TRIAL" "failed_wrk_status_${WRK_STATUS}" \
            >> "$FAILED_CSV"
          continue
        fi
        if [[ "$ERR_RATE" == "MISSING" ]]; then
          log_warn "$TAG missing wrk Lua summary; skipping CSV row"
          printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "$HOST_LABEL" "$LIBRARY" "$CONFIG" "$THREAD_MODE" "$THREAD_SETTING" "$C" "$TRIAL" "missing_wrk_lua_summary" \
            >> "$FAILED_CSV"
          continue
        fi
        if ! "$PYTHON_BIN" - "$ERR_RATE" <<'PYEOF'
import sys
raise SystemExit(0 if float(sys.argv[1]) <= 0.1 else 1)
PYEOF
        then
          log_warn "$TAG had wrk error rate ${ERR_RATE}% > 0.1%; skipping CSV row"
          printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "$HOST_LABEL" "$LIBRARY" "$CONFIG" "$THREAD_MODE" "$THREAD_SETTING" "$C" "$TRIAL" "high_wrk_error_rate_${ERR_RATE}" \
            >> "$FAILED_CSV"
          continue
        fi
        INVOL=$(parse_pidstat "$PS_FILE")
        MIGR=$(parse_migrations "$PERF_FILE")
        [[ "$MIGR" != "NA" ]] || die "Could not parse cpu-migrations from $PERF_FILE"
        echo "$HOST_LABEL,$LIBRARY,$CONFIG,$THREAD_MODE,$THREAD_SETTING,$C,$TRIAL,$P50,$P90,$P99,$RPS,$INVOL,$MIGR," >> "$RAW_CSV"
        log_info "$TAG p99=${P99}ms rps=${RPS} migr/s=${MIGR} host=${HOST_LABEL}"
        sleep 3
      done
    done
  done
done

if [[ "$DO_ANALYZE" == "yes" ]]; then
  log_step "Analyzing results"
  "$PYTHON_BIN" "$SCRIPT_DIR/analyze.py" --input "$RAW_CSV" --output-dir "$OUT_DIR"
fi

echo "Results written to: $OUT_DIR"
