#!/bin/bash
# Parameterized SLURM launcher for payload-transport training on KIAC.
#
# Adapted from mj-adapt-drones/scripts/train_quadrotor_rma_phase1.sh, extended so
# one script can launch many configurations and pin each to a specific commit.
# That matters for a sweep whose results are reviewed later: without a recorded
# commit there is no way to tell which code produced which curve.
#
# Usage:
#   sbatch --job-name=<name> scripts/kiac_train.sh <RUN_NAME> <GIT_REF> [extra args...]
#
# Example:
#   sbatch scripts/kiac_train.sh baseline a3d190f
#   sbatch scripts/kiac_train.sh wide_kernel a3d190f --agent.algorithm.learning-rate 1e-3
#
# The run gets its own checkout under $WORKDIR/runs/<RUN_NAME>, so concurrent
# jobs never share a working tree or a .venv.

#SBATCH --partition=ada
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn9        # old NVIDIA driver (CUDA 12.2): mujoco_warp loses CUDA graphs
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

RUN_NAME="${1:?usage: kiac_train.sh <RUN_NAME> <GIT_REF> [extra args...]}"
GIT_REF="${2:?usage: kiac_train.sh <RUN_NAME> <GIT_REF> [extra args...]}"
shift 2
EXTRA_ARGS=("$@")

REPO_URL="https://github.com/varadVaidya/payload-drones.git"
BASE_REPO="$HOME/robotics/mj-adapt-drones"
WORKDIR="$HOME/robotics/payload-runs"
RUN_DIR="$WORKDIR/$RUN_NAME"

source ~/.local/bin/env
export PYTHONUNBUFFERED=1
export GIT_PYTHON_REFRESH=quiet

echo "=========================================================="
echo "run          : $RUN_NAME"
echo "git ref      : $GIT_REF"
echo "extra args   : ${EXTRA_ARGS[*]:-<none>}"
echo "node         : $(hostname)"
echo "gpu          : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "started      : $(date -Is)"
echo "=========================================================="

# The base repo is an editable path dependency; without it uv sync cannot resolve.
if [ ! -d "$BASE_REPO" ]; then
  echo "FATAL: base repo missing at $BASE_REPO" >&2
  exit 1
fi

mkdir -p "$WORKDIR"
if [ ! -d "$RUN_DIR/.git" ]; then
  git clone --quiet "$REPO_URL" "$RUN_DIR"
fi
cd "$RUN_DIR"
git fetch --quiet origin
git checkout --quiet --detach "$GIT_REF"
echo "checked out  : $(git log --oneline -1)"

# pyproject pins the base repo by absolute path, which differs from the dev box.
# Rewrite it for this machine rather than editing the committed file.
python3 - "$BASE_REPO" <<'PY'
import pathlib, re, sys
base = sys.argv[1]
p = pathlib.Path("pyproject.toml")
s = p.read_text()
s2 = re.sub(r'path = "[^"]*mj-adapt-drones"', f'path = "{base}"', s)
if s2 != s:
    p.write_text(s2)
    print(f"pyproject    : repointed mj-adapt-drones -> {base}")
PY

uv sync --quiet
echo "python       : $(uv run --no-sync python -V 2>&1)"
uv run --no-sync python -c "import torch; print('torch        :', torch.__version__, '| cuda', torch.cuda.is_available())"

echo "----------------------------------------------------------"
uv run --no-sync train Mjlab-PayloadTransport-RMA-Phase1 \
  --agent.max-iterations 1500 \
  --agent.logger tensorboard \
  --agent.experiment-name "$RUN_NAME" \
  "${EXTRA_ARGS[@]}"

echo "=========================================================="
echo "finished     : $(date -Is)"
echo "=========================================================="
