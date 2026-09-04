# payload-drones

Cooperative aerial payload-transport training and evaluation on MJLab. This
package is installed editable into the parent `robot-learning` uv workspace.

```bash
cd ..
uv sync --all-packages --locked

cd payload-drones
uv run --project .. --locked python -m payload_drones.train Mjlab-PayloadTransport-RMA-Phase1
uv run --project .. --locked python -m payload_drones.play Mjlab-PayloadTransport-RMA-Phase1
uv run --project .. --locked python -m payload_drones.play_stats Mjlab-PayloadTransport-RMA-Phase1
```

Training and evaluation require a clean nested git repository. Training records
the commit and branch in W&B and tags the commit as `runs/<task>/<run-name>`.

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
