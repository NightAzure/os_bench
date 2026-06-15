# Reproducibility Guide


> When Pinning Hurts: CPU Affinity, Native Thread Pools, and Tail Latency in
> Python ML Inference Services

The maintained flow is intentionally single-entry:

```bash
bash scripts/run_all.sh --smoke
bash scripts/run_all.sh --quick
bash scripts/run_all.sh --full
```

## 1. Expected Host

The paper was designed around CPU-only Amazon Elastic Compute Cloud (EC2)
instances with four logical CPUs, especially `c5.xlarge` and `c6i.xlarge`.
The scripts also run on other Linux hosts, but the quantitative results should
be interpreted as topology-dependent.

Record these host details for every run:

- logical CPUs, physical cores, Simultaneous Multithreading (SMT) siblings,
  sockets, and Non-Uniform Memory Access (NUMA) nodes
- kernel version and scheduler generation
- container or cgroup CPU limits, if any
- Python package versions
- native-thread environment variables and runtime thread-pool introspection
- FastAPI/AnyIO sync-endpoint threadpool limit

`scripts/run_all.sh` calls `scripts/record_env.sh` automatically and writes the
snapshot into the result directory.
Environment capture is required: if `record_env.sh` fails, the benchmark run
stops instead of continuing with an incomplete reproducibility record.

## 2. Install Dependencies

On a fresh Ubuntu host:

```bash
bash scripts/setup_aws.sh
source /opt/os_bench_venv/bin/activate
```

Manual setup is also possible:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

System tools required for benchmark execution:

```text
bash, wrk, uvicorn, pidstat, mpstat, perf, taskset, renice, sudo, curl,
ps, lscpu, shuf, unzip
```

The smoke test does not require `wrk`, an EC2 host, or model downloads.
The benchmark runner uses `perf stat` for `cpu-migrations`; the recorded value
is divided by duration and worker count, so `cpu_migrations_per_s` means
migrations per worker per second. If the kernel blocks software counters, lower
`perf_event_paranoid` for the benchmark.
C2 uses `sudo -n renice`, so configure passwordless sudo for `/usr/bin/renice`
or omit C2 from `--configs`.
FastAPI sync endpoints use AnyIO's worker-thread limiter. The runner exports
`FASTAPI_THREAD_LIMIT` (default `40`) and each server reports it through
`/runtime`, so this additional thread layer is visible in diagnostics.

For final same-host runs on a 4-vCPU instance, isolate the load generator from
the server pool:

```bash
bash scripts/run_all.sh --full --trials 15 \
  --workers 3 \
  --server-cpus 0,1,2 \
  --client-cpus 3 \
  --host-label c5.xlarge
```

This launches Uvicorn master/workers on `--server-cpus` and runs `wrk`,
`pidstat`, `mpstat`, `perf stat`, health checks, and runtime probes on
`--client-cpus`. Server and client CPU pools must be disjoint. Without
`--client-cpus`, same-host client activity can compete with server workers and
should be treated as pilot data only.

## 3. Smoke Test

Run:

```bash
bash scripts/run_all.sh --smoke
```

This creates a synthetic `results/smoke/raw_data.csv`, compiles Python and Bash
scripts, runs the analyzer, and verifies that the expected figure files exist.
Use this before any long benchmark run.

## 4. Quick Benchmark

Run:

```bash
bash scripts/run_all.sh --quick
```

Quick mode runs a one-trial PyTorch benchmark over:

- `C0`: Linux default scheduler, soft affinity and migration-capable placement
- `C1`: coarse shared mask using the first two allowed logical CPUs
- `C2`: uniform `nice -10` priority control
- `C3`: one worker per allowed logical CPU
- `C4`: paired constrained affinity masks
- native-thread modes: runtime default and one native thread
- concurrency: 10 and 20

Quick mode is for sanity checking the host and scripts, not for final paper
statistics.

## 5. Full Benchmark

Run:

```bash
bash scripts/run_all.sh --full --trials 15
```

Full mode runs:

- PyTorch/SentenceTransformer over `C0`-`C4`, native-thread settings
  `default`, `2`, and `1`, and concurrency levels `1,5,10,20`
- NumPy/OpenBLAS and scikit-learn replication over `C0` and `C3` at
  concurrency `20`
- one combined `raw_data.csv`
- one environment snapshot
- one final analysis pass

Outputs are written to `results/repro_<timestamp>/` unless `--output-dir` is
provided.
Failed `wrk` trials are written to `failed_trials.csv` and are not included in
`raw_data.csv`.

The `2` native-thread setting is exploratory threshold/sensitivity evidence.
The main claim should be based on the contrast between runtime-default native
threads and one native thread unless the manuscript explicitly elevates the
two-thread sweep into a stated hypothesis.

## 6. Generated Artifacts

The maintained analyzer generates:

```text
fig1_affinity_spectrum_p99.{png,pdf}
fig2_p99_vs_concurrency.{png,pdf}
fig3_thread_interaction.{png,pdf}
fig4_bootstrap_ci_c20.{png,pdf}
fig5_throughput.{png,pdf}
fig6_invol_ctx.{png,pdf}
fig7_cross_library.{png,pdf}
fig8_cpu_migrations.{png,pdf}
summary_stats.txt
```

Use these figures in the revised manuscript as needed. The key new CSAI-plan
artifact is `fig1_affinity_spectrum_p99`, which compares Linux default
scheduling against per-logical-CPU pinning and paired constrained affinity.

## 7. Reanalyze Existing Data

To regenerate figures from an existing CSV:

```bash
python3 scripts/analyze.py \
  --input results/<run>/raw_data.csv \
  --output-dir results/<run>
```

The analyzer validates CSVs strictly. Missing columns, malformed numeric cells,
nonpositive latency/throughput values, or negative counters stop analysis.

## 8. Result Schema

`raw_data.csv` contains one row per benchmark trial:

```text
host_label, library, config, thread_mode, thread_setting, concurrency, trial,
p50_ms, p90_ms, p99_ms, rps, invol_csw_per_s, cpu_migrations_per_s, notes
```

`thread_mode` values:

- `parallel`: runtime-default native threads
- `limited`: explicitly capped native thread count greater than one
- `serial`: one native thread per worker

## 9. Notes On C4

`C4` is a paired constrained-affinity condition. It uses paired affinity masks
derived from the process's allowed CPUs and, where available, `lscpu` sibling
topology. On small SMT instances this often means migration is preserved only
between sibling logical CPUs on the same physical core, so C4 can mix migration
freedom with same-core SMT contention. Treat it as a paired-mask diagnostic, not
as a pure scheduler-migration experiment. It is not a cpuset/cgroup experiment
and is not equivalent to CPU isolation or `isolcpus`.

The runner randomizes thread-setting, configuration, and concurrency order to
reduce sensitivity to host drift while still avoiding unnecessary server
restarts inside a thread-setting block.

## 10. Practical Run Order

Use this order for final reproduction:

```bash
source /opt/os_bench_venv/bin/activate
bash scripts/run_all.sh --smoke
bash scripts/run_all.sh --quick --trials 1 --workers 3 --server-cpus 0,1,2 --client-cpus 3
bash scripts/run_all.sh --full --trials 15 --workers 3 --server-cpus 0,1,2 --client-cpus 3 --host-label c5.xlarge
```
