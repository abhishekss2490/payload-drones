# Lab GPU server setup

## Status

The target is `abhishek@10.176.60.200` (`dacasa6000`). The host inventory on
2026-10-05 found an RTX 6000 Ada GPU with 48 GB VRAM, NVIDIA driver 570.211.01,
32 logical CPUs, 124 GiB RAM, and about 61 GiB free on the system filesystem.
No Conda installation or SLURM client was found in the inspected locations.
The GPU was in use by another user. At the user's request, environment setup
proceeds while GPU validation and training remain pending. Setup records under
`~/robot-learning/.setup/` and `logs/server/setup/` contain the installation
results; this hardware inventory alone does not confirm GPU compatibility.

## Environment design

Use a dedicated Conda environment for Python and install the project's locked
Python packages into that environment. Conda and uv serve different purposes:
Conda provides the interpreter and isolation; `uv.lock` fixes the project's
runtime dependencies, including the CUDA-enabled PyTorch wheel and the RSL-RL
Git revision. Use the existing lab GPU driver and the lab's GPU allocation
mechanism.

The local stack checked on 2026-10-05 is:

| Component | Version |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.13.0+cu129 |
| PyTorch CUDA runtime | 12.9 |
| MJLab | 1.6.0 |
| MuJoCo | 3.11.0 |
| MuJoCo-Warp | 3.11.0 |
| Warp | 1.16.0 |
| RSL-RL | 5.4.2, Git commit `e3886a49257f53939711ed9898125bd989138d49` |
| NumPy | 2.5.2 |
| W&B | 0.28.2 |
| Installation tool: uv | 0.12.2 |

`robotics-common` must be a sibling checkout. The currently validated common
commit is `7ad2ef1edb052029ca51f13dfb5b326c7ed22d99`. Record both repositories'
commits for each experiment. Transfer the source checkouts with their Git
history; recreate the environment on the server from the lock.

## 1. Inspect the server

From a server checkout, run this read-only inventory:

```bash
python3 scripts/server-audit
```

Inspect the GPU model, free VRAM, other GPU workloads, driver version, available
RAM/disk, Conda installation, and SLURM tools. If the lab allocates GPUs through
SLURM, perform GPU validation and training inside an allocation.

The CUDA runtime in the PyTorch wheel and the host NVIDIA driver are separate
components. CUDA minor-version compatibility has limits, especially when code
is compiled to PTX at runtime. Verify PyTorch, Warp compilation, and an actual
MuJoCo-Warp rollout on the target GPU instead of treating a successful import as
proof that the stack works.

## 2. Create a dedicated Conda environment

The selected distribution is Miniforge 26.7.2-0, installed under
`~/miniforge3/` after verifying the official installer's SHA-256 checksum.
Load Conda in a new shell with:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
```

After inspecting the installed Conda distribution, create an unused environment
name and activate it. The example name is `payload-drones`:

```bash
conda create -n payload-drones --override-channels -c conda-forge python=3.12.13 pip
conda activate payload-drones
python -m pip install uv==0.12.2
cd ~/robot-learning/payload-drones
scripts/setup-conda
```

`setup-conda` refuses the base environment and an unexpected Python version. It
exports `uv.lock` as `pylock.toml` with `--locked`, installs the exact artifact
URLs and hashes, then installs both local editable packages without resolving
another dependency tree. It checks the installed dependencies afterward. The
Conda interpreter is targeted explicitly; the local laptop `.venv` is not used.

Audit files go under `logs/server/setup/<timestamp>-<pid>/`: setup output,
Conda package specification, exported package lock, installed package
list, key version checks, lock hash, and both source commits. Subsequent
experiments should recreate this environment from the recorded inputs when
changing dependencies.

## 3. Validate before full training

Within the allocated GPU environment:

1. Confirm `torch.cuda.is_available()` and execute a CUDA tensor operation.
2. Run the environment and model tests: `python -m pytest tests/test_env_smoke.py
   tests/test_marl_models.py -q`.
3. Run a bounded teacher-to-student pipeline to check optimization, checkpoint
   handoff, logging, and resume. Choose the environment count from measured
   VRAM use on the target GPU.
4. Record iteration times and peak memory. Estimate the full run using steady
   iteration times and report that estimate before committing to a long run.

The default full pipeline is 2,000 teacher iterations (256 rollout steps per
iteration) and 500 distillation iterations (48 rollout steps). The package
default is 4,096 environments; explicitly override both phases when benchmarking
another size. Fewer environments change the samples per iteration.

## 4. Training and outputs

The existing cluster wrapper calls the project's `.venv/bin/python`; it does
not select a Conda environment. For this setup, execute the Conda interpreter
inside the lab's allocated GPU session:

```bash
mkdir -p logs/server
WANDB_DIR="$PWD/logs/server" \
  "$CONDA_PREFIX/bin/python" -m payload_drones.train Mjlab-PayloadTransport-RMA \
  --log-root logs/server
```

This is the full-run command, to use after validation. The shared pipeline
requires clean project and `robotics-common` Git trees. Configure W&B login in
the server terminal, or explicitly use offline mode with a unique `WANDB_NAME`;
offline W&B does not assign the run name required by this pipeline.

Use `logs/server/` for server runs and retain `logs/laptop/` for laptop runs.
For an unmanaged lab workstation, run the validated command in tmux and capture
console output. For a scheduled machine, use the scheduler's job and output
settings. Keep GPU selection, environment count, seed, and source versions in
the run record.

## SSH access for assisted setup

A successful password login in one terminal does not authenticate another SSH
process. To let the assistant use your connection, run this in a second local
WSL terminal and enter the password in that terminal:

```bash
ssh -M -S /tmp/payload-gpu-ssh -o ControlPersist=30m -fN abhishek@10.176.60.200
```

The assistant can then use the connection with `ssh -S /tmp/payload-gpu-ssh`.
After setup, close it from the local WSL terminal:

```bash
ssh -S /tmp/payload-gpu-ssh -O exit abhishek@10.176.60.200
```

## Reference documentation

- [uv: Conda environments and explicit interpreter selection](https://docs.astral.sh/uv/pip/environments/)
- [uv: exporting the existing lockfile](https://docs.astral.sh/uv/concepts/projects/export/)
- [NVIDIA: CUDA minor-version compatibility and PTX limitations](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
