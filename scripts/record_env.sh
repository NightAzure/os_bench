#!/usr/bin/env bash
# record_env.sh - Snapshot all system parameters required for the
# reproducibility record. Run once at the start of each experimental session.
#
# Usage:
#   bash scripts/record_env.sh [output_file]
#   Default output: results/env_<timestamp>.txt

set -euo pipefail

OUT="${1:-results/env_$(date +%Y%m%d_%H%M%S).txt}"
mkdir -p "$(dirname "$OUT")"
exec > >(tee "$OUT") 2>&1

echo "==================================================================="
echo "os_bench OS Scheduling Experiment - System Configuration Snapshot"
echo "Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Host: $(hostname)"
echo "==================================================================="

echo ""
echo "--- OS and Kernel ---"
uname -a
lsb_release -a 2>/dev/null || (cat /etc/os-release 2>/dev/null | head -6)


echo ""
echo "--- AWS Instance Metadata (best effort) ---"
TOKEN="$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' || true)"
if [[ -n "$TOKEN" ]]; then
  md() { curl -s -m 2 -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/$1" 2>/dev/null || true; }
  echo "instance-type: $(md instance-type)"
  echo "instance-id: $(md instance-id)"
  echo "placement/availability-zone: $(md placement/availability-zone)"
else
  echo "AWS IMDS unavailable (normal outside EC2 or when IMDS is disabled)"
fi

echo ""
echo "--- CPU Topology (lscpu) ---"
lscpu

echo ""
echo "--- CPU Topology Table ---"
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE 2>/dev/null || echo "lscpu -e topology table not available"

echo ""
echo "--- Physical Core Count ---"
echo "Physical cores: $(cat /sys/devices/system/cpu/cpu*/topology/core_id 2>/dev/null | sort -u | wc -l || echo 'N/A (VM or no topology sysfs)')"
echo "Logical CPUs:   $(nproc)"

echo ""
echo "--- CPU Frequency Governor ---"
if ls /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1; then
  cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort -u
  echo ""
  echo "Max frequency (kHz):"
  cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq 2>/dev/null | sort -u || echo "N/A"
else
  echo "cpufreq not available (VM or no cpufreq sysfs - normal on EC2)"
fi

echo ""
echo "--- Scheduler Sysctls ---"
# Kernel >=6.6 replaced CFS with EEVDF; sysctl names changed.
# We probe both old (CFS) and new (EEVDF) names.
for key in \
  kernel.sched_latency_ns \
  kernel.sched_min_granularity_ns \
  kernel.sched_wakeup_granularity_ns \
  kernel.sched_migration_cost_ns \
  kernel.sched_nr_migrate \
  kernel.sched_tunable_scaling \
  kernel.sched_child_runs_first \
  kernel.sched_base_slice_ns; do
  # sysctl || true prevents set -e from killing the script when key is missing
  val=$(sysctl -n "$key" 2>/dev/null) || true
  if [[ -n "$val" ]]; then
    echo "$key = $val"
  else
    procfs_path="/proc/sys/${key//./\/}"
    if [[ -f "$procfs_path" ]]; then
      echo "$key = $(cat "$procfs_path")"
    else
      echo "$key = N/A (not present on kernel $(uname -r))"
    fi
  fi
done

echo ""
echo "--- NUMA ---"
sysctl kernel.numa_balancing 2>/dev/null || echo "kernel.numa_balancing = N/A"
numactl --hardware 2>/dev/null || echo "numactl not available or single NUMA node"

echo ""
echo "--- Memory ---"
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree|HugePages_Total|Hugepagesize)' /proc/meminfo

echo ""
echo "--- Transparent Huge Pages ---"
cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo "THP info not available"

echo ""
echo "--- Cgroup CPU Quota ---"
# cgroup v1
if [[ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
  echo "cgroup v1 cpu.cfs_quota_us: $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)"
  echo "cgroup v1 cpu.cfs_period_us: $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)"
# cgroup v2
elif [[ -f /sys/fs/cgroup/cpu.max ]]; then
  echo "cgroup v2 cpu.max: $(cat /sys/fs/cgroup/cpu.max)"
else
  echo "No cgroup CPU quota file found"
fi

echo ""
echo "--- Cgroup Memory Limit ---"
if [[ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
  echo "cgroup v1 memory.limit_in_bytes: $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
elif [[ -f /sys/fs/cgroup/memory.max ]]; then
  echo "cgroup v2 memory.max: $(cat /sys/fs/cgroup/memory.max)"
else
  echo "No cgroup memory limit file found"
fi

echo ""
echo "--- System Load at Snapshot Time ---"
uptime
cat /proc/loadavg

echo ""
echo "--- Running Processes (Top CPU consumers) ---"
ps -eo pid,ni,pcpu,pmem,comm --sort=-pcpu 2>/dev/null | head -15

echo ""
echo "--- Python and Package Versions ---"
python3 --version 2>/dev/null || echo "python3 not found"
python3 -c "import torch; print('torch:', torch.__version__)" 2>/dev/null || echo "torch: not installed"
python3 -c "import numpy; print('numpy:', numpy.__version__)" 2>/dev/null || echo "numpy: not installed"
python3 -c "import sklearn; print('scikit-learn:', sklearn.__version__)" 2>/dev/null || echo "scikit-learn: not installed"
python3 -c "import sentence_transformers; print('sentence-transformers:', sentence_transformers.__version__)" 2>/dev/null || echo "sentence-transformers: not installed"
python3 -c "import fastapi; print('fastapi:', fastapi.__version__)" 2>/dev/null || echo "fastapi: not installed"
python3 -c "import uvicorn; print('uvicorn:', uvicorn.__version__)" 2>/dev/null || echo "uvicorn: not installed"
python3 -c "import threadpoolctl; print('threadpoolctl:', threadpoolctl.__version__)" 2>/dev/null || echo "threadpoolctl: not installed"

echo ""
echo "--- Native Thread Runtime Introspection ---"
python3 - <<'PY' 2>/dev/null || echo "native thread introspection unavailable"
import json
import os

keys = [
    "PYTORCH_NUM_THREADS",
    "TORCH_NUM_INTEROP_THREADS",
    "FASTAPI_THREAD_LIMIT",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLAS_NUM_THREADS",
]
print("thread_env:", {k: os.environ.get(k) for k in keys})
try:
    import torch
    print("torch_num_threads:", torch.get_num_threads())
    print("torch_num_interop_threads:", torch.get_num_interop_threads())
except Exception as exc:
    print("torch_thread_info:", f"unavailable ({exc})")
try:
    from threadpoolctl import threadpool_info
    print("threadpoolctl_info:", json.dumps(threadpool_info(), sort_keys=True))
except Exception as exc:
    print("threadpoolctl_info:", f"unavailable ({exc})")
PY

echo ""
echo "--- Tool Versions ---"
# wrk exits non-zero when called without a URL even though it prints its version,
# so we check for existence separately to avoid a spurious "not found" echo.
# head -1 extracts just the version line (wrk also prints full usage help after it).
if command -v wrk > /dev/null 2>&1; then
  wrk --version 2>/dev/null | head -1 || true
else
  echo "wrk not found"
fi
# pidstat (sysstat) uses -V not --version; also exits non-zero on some builds
if command -v pidstat > /dev/null 2>&1; then
  pidstat -V 2>&1 | head -1 || true
else
  echo "pidstat not found"
fi
# perf: the linux-tools-common wrapper looks for /usr/lib/linux-tools/$(uname -r)/perf
# and fails if the kernel-versioned directory is absent. Fall back to the versioned
# path directly so we get a version even when the wrapper isn't in PATH.
_PERF_BIN="/usr/lib/linux-tools/$(uname -r)/perf"
if command -v perf > /dev/null 2>&1; then
  perf --version 2>/dev/null || true
elif [[ -x "$_PERF_BIN" ]]; then
  "$_PERF_BIN" --version 2>/dev/null || true
else
  # Report the mismatch explicitly so it's visible in the env file
  _PERF_FOUND=$(ls /usr/lib/linux-tools/*/perf 2>/dev/null | head -1 || true)
  if [[ -n "$_PERF_FOUND" ]]; then
    echo "perf: kernel-mismatched binary at $_PERF_FOUND (running $(uname -r))"
  else
    echo "perf not found"
  fi
fi
taskset --version 2>/dev/null | head -1 || echo "taskset: $(which taskset 2>/dev/null || echo not found)"

echo ""
echo "--- Network Interface (loopback) ---"
ip addr show lo 2>/dev/null | head -4 || echo "loopback info not available"

echo ""
echo "=== END OF ENVIRONMENT SNAPSHOT ==="
echo "Snapshot written to: $OUT"
