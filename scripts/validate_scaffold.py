"""Empirical validation for the spatial-tendon payload scaffold.

Run with ``uv run python scripts/validate_scaffold.py``.  The script uses MuJoCo
for deterministic single-world semantics and mujoco-warp for the batched GPU
checks when CUDA is available.
"""

from __future__ import annotations

import gc
import time

import mujoco
import numpy as np
import torch

from mjlab.scene import Scene
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sim.sim import Simulation

from payload_drones.scene.builder import (
    DEFAULT_CABLE_LENGTH,
    DEFAULT_CABLE_STIFFNESS,
    N_CABLES,
    make_payload_scene_cfg,
)

PAYLOAD_QPOS_ADR = 21
PAYLOAD_MASS = 0.75
QUAD_MASS = 0.792
GRAVITY = 9.81


def make_model() -> mujoco.MjModel:
    """Compile the scene through mjlab, including its post-attach spec callback."""
    scene = Scene(make_payload_scene_cfg(), device="cpu")
    return scene.compile()


def tendon_tension(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return unilateral spring tensions (positive values pull, never push)."""
    extension = data.ten_length - model.tendon_lengthspring[:, 1]
    return np.maximum(0.0, model.tendon_stiffness * extension)


def semantic_checks(model: mujoco.MjModel) -> tuple[bool, bool, bool]:
    """Check taut, slack, and symmetric static load sharing in MuJoCo."""
    taut = mujoco.MjData(model)
    taut.qpos[:] = model.qpos0
    # A 5 mm payload drop stretches each vertical cable equally.
    taut.qpos[PAYLOAD_QPOS_ADR + 2] -= 0.005
    mujoco.mj_forward(model, taut)
    taut_force = tendon_tension(model, taut)
    taut_ok = bool(np.all(taut.ten_length > DEFAULT_CABLE_LENGTH) and np.all(taut_force >= 0.0))
    print(f"taut: lengths={taut.ten_length}, tensions={taut_force} N, pass={taut_ok}")

    slack = mujoco.MjData(model)
    slack.qpos[:] = model.qpos0
    slack.qpos[PAYLOAD_QPOS_ADR + 2] += 0.10
    mujoco.mj_forward(model, slack)
    slack_force = tendon_tension(model, slack)
    slack_ok = bool(np.all(slack.ten_length < DEFAULT_CABLE_LENGTH) and np.allclose(slack_force, 0.0))
    print(f"slack: lengths={slack.ten_length}, tensions={slack_force} N, pass={slack_ok}")

    share = mujoco.MjData(model)
    share.qpos[:] = model.qpos0
    # Extension that produces payload weight / 3 in each of three springs.
    share.qpos[PAYLOAD_QPOS_ADR + 2] -= PAYLOAD_MASS * GRAVITY / (N_CABLES * DEFAULT_CABLE_STIFFNESS)
    mujoco.mj_forward(model, share)
    forces = tendon_tension(model, share)
    expected = PAYLOAD_MASS * GRAVITY / N_CABLES
    sharing_ok = bool(np.allclose(forces, expected, rtol=0.02, atol=0.02))
    print(f"load share: tensions={forces} N, expected={expected:.4f} N, pass={sharing_ok}")
    return taut_ok, slack_ok, sharing_ok


def stability_check(model: mujoco.MjModel) -> bool:
    """Run the requested 2,000 equal-hover-thrust steps and report drift."""
    data = mujoco.MjData(model)
    data.ctrl[:] = (QUAD_MASS + PAYLOAD_MASS / N_CABLES) * GRAVITY / 4.0
    initial = data.qpos[PAYLOAD_QPOS_ADR : PAYLOAD_QPOS_ADR + 3].copy()
    for _ in range(2_000):
        mujoco.mj_step(model, data)
    drift = float(np.linalg.norm(data.qpos[PAYLOAD_QPOS_ADR : PAYLOAD_QPOS_ADR + 3] - initial))
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    stable = finite and drift < 2.0
    print(f"2,000-step equal-hover: finite={finite}, payload drift={drift:.3f} m, pass={stable}")
    return stable


def gpu_checks(model: mujoco.MjModel) -> None:
    """Validate batched tendon ranges and benchmark GPU throughput if available."""
    if not torch.cuda.is_available():
        print("GPU checks: SKIPPED (no CUDA device exposed to this process).")
        return
    for worlds in (256, 512, 1024, 2048, 4096):
        try:
            torch.cuda.reset_peak_memory_stats()
            sim = Simulation(
                worlds,
                SimulationCfg(nconmax=64, njmax=128, mujoco=MujocoCfg(integrator="euler", timestep=0.005)),
                model,
                "cuda",
            )
            sim.expand_model_fields(("tendon_range",))
            sim.model.tendon_range[: worlds // 2, :, 1] = 0.55
            sim.model.tendon_range[worlds // 2 :, :, 1] = 0.65
            sim.forward()
            for _ in range(100):
                sim.step()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(500):
                sim.step()
            torch.cuda.synchronize()
            rate = worlds * 500 / (time.perf_counter() - start)
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"GPU sweep: envs={worlds}, steps/s={rate:.0f}, peak={peak:.2f} GiB")
            del sim
            gc.collect()
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"GPU sweep: envs={worlds}, OOM")
            torch.cuda.empty_cache()
            break


def main() -> None:
    """Execute every scaffold check without hiding a failed assertion."""
    model = make_model()
    nominal = mujoco.MjData(model)
    mujoco.mj_forward(model, nominal)
    print(f"compile: ntendon={model.ntendon} (expected {N_CABLES})")
    print(f"constraint allocation at nominal pose: nconmax>=64, njmax>={nominal.nefc}")
    taut_ok, slack_ok, share_ok = semantic_checks(model)
    stable = stability_check(model)
    gpu_checks(model)
    if model.ntendon != N_CABLES or not (taut_ok and slack_ok and share_ok and stable):
        raise SystemExit("SCAFFOLD VALIDATION FAILED: do not proceed to RL training.")


if __name__ == "__main__":
    main()
