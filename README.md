# payload-drones

Cooperative aerial payload-transport training and evaluation on MJLab. Install
from this directory; `robotics-common` must be checked out beside this repository.
The cluster and laptop launchers both use this project's `.venv`.

```bash
uv sync --locked
.venv/bin/wandb login
```

Training and evaluation require a clean Git repository. Commit your changes
before starting a run. Training records the project and `robotics-common`
commits, logs to W&B, and tags the project commit as
`runs/<pipeline>/<run-name>`.

## Train on the laptop GPU

Run from a Linux or WSL2 terminal with an NVIDIA GPU visible to PyTorch:

```bash
scripts/laptop-train baseline Mjlab-PayloadTransport-RMA
```

This uses the same `payload_drones.train` entry point as `scripts/slurm-train`:
PPO teacher training, automatic checkpoint handoff, then student distillation.
It retains W&B logging, phase checkpoints, the run manifest, resume, video
options, and all phase-specific training arguments. The launcher checks CUDA,
prints visible GPUs and free memory, and saves console output while training
runs in the foreground. Training failures return a nonzero exit status.

New laptop runs default to **1,024 environments per phase**, compared with the
training configuration's 4,096. Rollout lengths, iteration counts, rewards,
networks, and optimizer settings keep their existing defaults. Consequently,
the laptop collects one quarter as many environment steps per iteration; compare
experiments by total samples as well as iterations.

A local check on an 8 GiB RTX 4070 Laptop GPU completed one full training iteration
in each phase at 1,024 environments, including loading the teacher weights into
the student runner. Peak PyTorch reserved memory was 2.41 GiB for the teacher and
2.92 GiB for distillation. These figures exclude memory allocated directly by
the simulator and do not measure a complete training run or video recording.

### Preview and customize

```bash
# Print the command without checking CUDA or starting training.
scripts/laptop-train --dry-run baseline Mjlab-PayloadTransport-RMA

# Use fewer environments if other applications need GPU memory.
LAPTOP_NUM_ENVS=512 scripts/laptop-train smaller Mjlab-PayloadTransport-RMA

# Explicit phase options override the launcher defaults.
scripts/laptop-train custom Mjlab-PayloadTransport-RMA \
  --phases.teacher.env.scene.num-envs 512 \
  --phases.distill.env.scene.num-envs 256

# Show all options accepted by the shared pipeline.
scripts/laptop-train help Mjlab-PayloadTransport-RMA --help
```

GPU selection also passes through, for example
`--phases.teacher.gpu-ids '[0]' --phases.distill.gpu-ids '[0]'`. Both phases must select
the same single GPU. For the cluster, the equivalent command retains the wall
time argument and submits through SLURM:

```bash
scripts/slurm-train baseline 04:00:00 Mjlab-PayloadTransport-RMA
```

### Logs and resume

- `JOB_NAME` (such as `baseline`) labels the console log. W&B assigns the run name.
- Console logs: `logs/laptop/console/<timestamp>-<job-name>-<pid>.log`.
- Run manifest and checkpoints: `logs/laptop/Mjlab-PayloadTransport-RMA/<wandb-name>/`,
  with `phases/01-teacher/` and `phases/02-distill/` subdirectories.
- W&B metrics and debug logs: `logs/laptop/wandb/`. Upload staging, caches, and
  downloaded artifacts also stay under `logs/laptop/`.
- `--log-root <path>` moves all these outputs together. Cluster runs retain their
  existing output layout.

```bash
scripts/laptop-train continued Mjlab-PayloadTransport-RMA --resume <wandb-name>
```

Resume uses the saved manifest and latest checkpoint, skips completed phases,
and requires the original project and `robotics-common` commits. Do not add
phase overrides to a resume command; the shared pipeline rejects them. The
laptop launcher adds no environment defaults on resume, so a cluster run retains
its original environment count. A run that needs more memory than the laptop
provides cannot be made smaller through this resume option. When using a custom
output root, pass the same `--log-root` to locate the run.

For local logging without a W&B account:

```bash
WANDB_MODE=offline scripts/laptop-train baseline Mjlab-PayloadTransport-RMA
# After logging in, sync the specific offline run directory printed in the log:
.venv/bin/wandb sync logs/laptop/wandb/offline-run-<timestamp>-<run-id>
```

Offline runs get a unique `laptop-<job>-<timestamp>-<pid>` name. W&B records a
separate local log segment when an offline run resumes; checkpoint recovery is
still handled by the pipeline. Use `WANDB_MODE=offline` again when resuming an
unsynced offline run.

To keep training running after disconnecting the terminal, start it inside
`tmux` if installed:

```bash
tmux new -s payload
scripts/laptop-train baseline Mjlab-PayloadTransport-RMA
# Detach with Ctrl-b, then d. Reconnect with: tmux attach -t payload
```

### Evaluate and inspect

```bash
.venv/bin/python -m payload_drones.evaluate Mjlab-PayloadTransport-RMA --log-root logs/laptop --run-name <wandb-name>
.venv/bin/python -m payload_drones.play Mjlab-PayloadTransport-RMA-Phase1
.venv/bin/python -m payload_drones.play_stats Mjlab-PayloadTransport-RMA-Phase1
```

Evaluation runs one fresh episode for both phases.

## Reset crash after the shared-environment migration

### Symptom

The payload model imported correctly and both registered RSL-RL configurations
could be constructed, but the first real environment reset terminated Python
with exit code 139. This was a native segmentation fault, not a catchable Python
exception. The failing path was:

```text
ManagerBasedRlEnv.reset()
  -> CommandManager.reset()
  -> PayloadTrajectoryCommand._resample_command()
  -> PayloadTrajectoryCommand._initialize_team_at_reference()
  -> settle_quads_for_cable_slack()
  -> Simulation.forward()
  -> mujoco_warp.forward()
  -> collision / convex_narrowphase
  -> SIGSEGV
```

The failure appeared after moving the project from its private environment to
the current shared stack:

| Environment | MJLab | MuJoCo | MuJoCo-Warp | Warp | Result |
|---|---:|---:|---:|---:|---|
| Original project venv | 1.4.0 | 3.8.1 | 3.8.1 | 1.12.1 | Test passed |
| Shared workspace, before fix | 1.6.0 | 3.11.0 | 3.11.0 | 1.16.0 | Segmentation fault |

The failing shared-stack run used a freshly generated, versioned Warp 1.16.0
kernel cache, ruling out stale kernels from the original environment. The same
test and source passed under the original venv. MuJoCo-Warp officially supports
CPU execution for development and debugging, although its primary target is an
NVIDIA GPU; therefore a CPU native crash was not treated as an expected test
skip. See the [MuJoCo-Warp project documentation](https://github.com/google-deepmind/mujoco_warp).

### Why the full forward pass was the wrong operation

`settle_quads_for_cable_slack()` runs in the middle of reset. At that point the
new payload pose has been written, but the three quadrotors have not yet been
moved over their corresponding payload attachment sites. This transient,
pre-settle state is intentional.

The function needs current world-space hook and attachment positions to compute
the final quadrotor poses. Those values are derived from `qpos`, so they must be
refreshed after writing the payload pose. The old implementation refreshed them
with `env.sim.forward()`. A full forward does much more than update transforms:
it also enters collision detection, contact generation, constraints, and the
rest of MuJoCo-Warp's forward pipeline. Consequently, convex collision kernels
were being run on the temporary geometry before cable-safe placement was
complete. In MuJoCo-Warp 3.11's CPU path, that unnecessary collision pass was
the immediate trigger for the segmentation fault.

### Fix

The reset now calls MuJoCo-Warp's public kinematics operation directly:

```python
with wp.ScopedDevice(env.sim.wp_device):
    mjwarp.kinematics(env.sim.wp_model, env.sim.wp_data)
```

Kinematics refreshes the body and site transforms needed by
`quad.data.site_pos_w` and `payload.data.site_pos_w` without running collision
detection on the incomplete layout. The function then centres each quad hook
over its payload attachment, applies the independently sampled cable length,
and writes the final root poses and velocities. Normal simulator forward/step
processing still occurs once the reset geometry is valid; no collision physics
or training-time simulation was disabled.

This was preferable to either globally disabling collisions or downgrading the
shared environment. Disabling collisions would change the task, while MJLab
1.6 requires the MuJoCo 3.11 generation and the workspace deliberately owns one
consistent simulator stack.

### Verification

After the change, the complete copied payload suite passes on CPU:

```bash
uv run --project .. --locked python -m pytest \
  tests/test_env_smoke.py tests/test_marl_models.py -q
# 30 passed, 3 skipped
```

The three skips are explicitly CUDA-only memory/rollout checks. MuJoCo-Warp
still emits a warning that its MultiCCD path generates at most one contact for
the scene's cylinder-cylinder and cylinder-box pairs. That warning is separate
from the reset crash and remains visible rather than being suppressed.
