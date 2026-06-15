#!/usr/bin/env bash
# setup_aws.sh - One-time setup for Ubuntu 22.04 EC2 instance.
#
# Recommended instance: c5.xlarge (4 vCPU, 8 GB RAM)
#   - Dedicated vCPUs (no burstable credit), consistent CPU performance
#   - ~$0.17/hr on-demand; use the reduced topology check for the final validation pass
#   - Launch with Ubuntu 22.04 LTS AMI, 20 GB gp3 root volume
#
# Run as: bash scripts/setup_aws.sh
# Then activate venv: source /opt/os_bench_venv/bin/activate

set -euo pipefail

echo "================================================"
echo " os_bench: AWS EC2 setup (Ubuntu 22.04)"
echo " Instance should be: c5.xlarge or larger"
echo "================================================"

# ---- System packages ----
echo ""
echo "[1/5] Installing system packages..."
sudo apt-get update -y -q
sudo apt-get install -y -q \
    python3-pip python3-venv python3-dev \
    build-essential libssl-dev git curl wget \
    sysstat \
    linux-tools-common "linux-tools-$(uname -r)" linux-tools-generic \
    numactl \
    htop

echo "  System packages installed."

# ---- wrk load generator ----
echo ""
echo "[2/5] Installing wrk..."
if command -v wrk &>/dev/null; then
    echo "  wrk already installed: $(wrk --version 2>&1 | head -1)"
else
    git clone --depth 1 https://github.com/wg/wrk.git /tmp/wrk_build
    cd /tmp/wrk_build
    make -j"$(nproc)"
    sudo cp wrk /usr/local/bin/wrk
    cd - > /dev/null
    rm -rf /tmp/wrk_build
    echo "  wrk installed: $(wrk --version 2>&1 | head -1)"
fi

# ---- Python virtual environment ----
echo ""
echo "[3/5] Setting up Python venv at /opt/os_bench_venv..."
VENV="/opt/os_bench_venv"
if [ ! -d "$VENV" ]; then
    sudo python3 -m venv "$VENV"
    sudo chown -R "$(whoami):$(whoami)" "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip -q

# ---- Python packages ----
echo ""
echo "[4/5] Installing Python packages..."

# PyTorch CPU-only first (smaller download, ~200 MB vs 700 MB)
pip install torch --index-url https://download.pytorch.org/whl/cpu -q

# Remaining requirements
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
pip install -r "$REPO_ROOT/requirements.txt" -q

echo "  Python packages installed."

# ---- Pre-download SentenceTransformer model ----
echo ""
echo "[5/5] Pre-downloading SentenceTransformer model (all-MiniLM-L6-v2)..."
python3 - <<'PYEOF'
from sentence_transformers import SentenceTransformer
import numpy as np

print("  Downloading model weights...")
m = SentenceTransformer("all-MiniLM-L6-v2")

print("  Warming up (first inference is slow due to JIT)...")
_ = m.encode("warmup text", convert_to_numpy=True)

# Verify numpy BLAS
a = np.random.randn(256, 256).astype(np.float32)
b = np.random.randn(256, 256).astype(np.float32)
_ = np.dot(a, b)

print("  Model and BLAS ready.")
PYEOF

# ---- sudo for nice ----
echo ""
echo "================================================"
echo " Setup complete!"
echo ""
echo " IMPORTANT: C2 config (nice -10) requires sudo."
echo " Add this line to /etc/sudoers via visudo:"
echo "   $(whoami) ALL=(ALL) NOPASSWD: /usr/bin/renice"
echo ""
echo " perf is required for CPU migration counters. If perf_event_paranoid"
echo " blocks software counters on your image, lower it for the benchmark:"
echo "   echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid"
echo ""
echo " To activate venv in a new shell:"
echo "   source $VENV/bin/activate"
echo ""
echo " To run experiments:"
echo "   cd $REPO_ROOT"
echo "   bash scripts/run_all.sh --smoke"
echo "   bash scripts/run_all.sh --quick --trials 1"
echo "   bash scripts/run_all.sh --full --trials 15 --host-label c5.xlarge"
echo "================================================"
