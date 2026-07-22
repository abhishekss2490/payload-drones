#!/bin/bash
# SLURM launcher for payload-transport training on KIAC.
#
# Usage:
#   sbatch --job-name=<name> scripts/kiac_train.sh <RUN_NAME> [extra args...]
#
# Example:
#   sbatch --job-name=pd_baseline scripts/kiac_train.sh baseline \
#     --agent.algorithm.learning-rate 1e-3
#
# Runs from the submitting checkout directly. payload-drones is self-contained
# now that the mj-adapt-drones surface is vendored, so there is no external
# checkout to locate and no pyproject to rewrite. There is also no reason to
# clone per run: mjlab keys its outputs by --agent.experiment-name, unique per
# run, so concurrent jobs read the tree without writing to it. The earlier
# per-run clone plus `uv sync` cost four identical ~8 GB syncs for no benefit.
#
# The commit is recorded in the log rather than passed in. Sweeps get reviewed
# hours later, and a curve with no provenance is not evidence.

#SBATCH --partition=ada
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn9        # old NVIDIA driver (CUDA 12.2): mujoco_warp loses CUDA graphs
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

RUN_NAME="${1:?usage: kiac_train.sh <RUN_NAME> [extra args...]}"
shift
EXTRA_ARGS=("$@")

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source ~/.local/bin/env
export PYTHONUNBUFFERED=1
export GIT_PYTHON_REFRESH=quiet

echo "=========================================================="
echo "run        : $RUN_NAME"
echo "commit     : $(git log --oneline -1)"
echo "extra args : ${EXTRA_ARGS[*]:-<none>}"
echo "node       : $(hostname)"
echo "gpu        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "started    : $(date -Is)"
echo "=========================================================="

uv run --no-sync python -c "import torch; print('torch      :', torch.__version__, '| cuda', torch.cuda.is_available())"

echo "----------------------------------------------------------"
uv run --no-sync train Mjlab-PayloadTransport-RMA-Phase1 \
  --agent.max-iterations 1500 \
  --agent.logger tensorboard \
  --agent.experiment-name "$RUN_NAME" \
  "${EXTRA_ARGS[@]}"

echo "=========================================================="
echo "finished   : $(date -Is)"
echo "=========================================================="
