#!/bin/bash
# SLURM launcher for payload-transport training on KIAC.
#
# Usage:
#   sbatch --job-name=<name> scripts/kiac_train.sh <RUN_NAME> [extra args...]
#
# Example:
#   sbatch --job-name=pd_baseline scripts/kiac_train.sh baseline \
#     --phases.teacher.agent.algorithm.learning-rate 1e-3
#
# Runs from the submitting checkout through robot-learning's shared environment.
#
# The commit is recorded in the log rather than passed in. Sweeps get reviewed
# hours later, and a curve with no provenance is not evidence.

#SBATCH --partition=ada
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn8,cn9
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

RUN_NAME="${1:?usage: kiac_train.sh <RUN_NAME> [extra args...]}"
shift
EXTRA_ARGS=("$@")

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON="$(dirname "$REPO")/.venv/bin/python"
cd "$REPO"
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

"$PYTHON" -c "import torch; print('torch      :', torch.__version__, '| cuda', torch.cuda.is_available())"

echo "----------------------------------------------------------"
"$PYTHON" -m payload_drones.train Mjlab-PayloadTransport-RMA \
  --phases.teacher.agent.max-iterations 2000 \
  --phases.teacher.agent.experiment-name "$RUN_NAME" \
  "${EXTRA_ARGS[@]}"

echo "=========================================================="
echo "finished   : $(date -Is)"
echo "=========================================================="
