"""Corrected stability check for the cable/payload subsystem.

The original open-loop test commanded equal hover thrust on three *free-flying*
quadrotors and called the resulting drift a scaffold failure. That test is
invalid: a quadrotor is open-loop unstable in attitude, so three free quads plus
a free 6-DoF payload at fixed equal thrust is an unstable equilibrium. Drift is
correct physics, not a defect in the cable model.

This script isolates the question we actually care about — *is the cable and
payload physics stable and well conditioned?* — by pinning the quadrotors
kinematically each step and letting only the payload move. If the payload
settles to a static hang with steady tensions and no energy injection, the
cable representation is sound.

Run: ``uv run --no-sync python scripts/check_cable_physics.py``
"""

from __future__ import annotations

import mujoco
import numpy as np

from mjlab.scene import Scene

from payload_drones.scene.builder import DEFAULT_CABLE_LENGTH, N_CABLES, make_payload_scene_cfg

PAYLOAD_QPOS_ADR = 21
PAYLOAD_MASS = 0.75
GRAVITY = 9.81
N_STEPS = 4000  # 20 s at 5 ms


def _tensions(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Read unilateral tensions from active tendon-limit constraint rows.

    The cable has zero explicit stiffness: ``stiffness * extension`` would
    therefore (incorrectly) report zero load.  MuJoCo exposes the constraint
    row for each active tendon through ``tendon_efcadr``; an inactive (slack)
    tendon has address -1 and, correctly, zero tension.
    """
    tension = np.zeros(model.ntendon)
    tendon_limit = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)
    for tendon_id, row in enumerate(data.tendon_efcadr):
        if row < 0:
            continue
        if data.efc_type[row] != tendon_limit or data.efc_id[row] != tendon_id:
            raise RuntimeError(f"unexpected constraint row for tendon {tendon_id}: {row}")
        tension[tendon_id] = max(0.0, data.efc_force[row])
    return tension


def main() -> None:
    scene = Scene(make_payload_scene_cfg(), device="cpu")
    model = scene.compile()
    # Energy diagnostics are off by default; without this mj_step leaves
    # data.energy at zero and any energy assertion is vacuous.
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0

    # Snapshot the three quad free joints so we can re-pin them every step.
    # Quad free joints occupy qpos[0:21] (3 quads x 7) and qvel[0:18] (3 x 6).
    quad_qpos0 = data.qpos[:PAYLOAD_QPOS_ADR].copy()

    # Displace the payload laterally so it must swing and settle — a static
    # start would pass trivially.
    data.qpos[PAYLOAD_QPOS_ADR + 0] += 0.10
    mujoco.mj_forward(model, data)

    energies: list[float] = []
    payload_z: list[float] = []
    tension_trace: list[np.ndarray] = []
    extension_trace: list[np.ndarray] = []

    for step in range(N_STEPS):
        # Kinematic pin: hold quads exactly fixed, let only the payload move.
        data.qpos[:PAYLOAD_QPOS_ADR] = quad_qpos0
        data.qvel[: PAYLOAD_QPOS_ADR - 3] = 0.0  # 18 dof of quad velocity
        mujoco.mj_step(model, data)

        if step % 10 == 0:
            ke = float(data.energy[1]) if data.energy.size > 1 else 0.0
            pe = float(data.energy[0]) if data.energy.size > 0 else 0.0
            energies.append(ke + pe)
            payload_z.append(float(data.qpos[PAYLOAD_QPOS_ADR + 2]))
            tension_trace.append(_tensions(model, data).copy())
            extension_trace.append((data.ten_length - model.tendon_range[:, 1]).copy())

    tensions = np.array(tension_trace)
    extensions = np.array(extension_trace)
    finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))

    # Settling: the payload starts displaced, so it must *decay*, not already be
    # still. Compare oscillation amplitude early vs late — a ratio < 1 means the
    # swing is damping out, which is what stability actually means here.
    quarter = max(1, len(tensions) // 4)
    head_std = tensions[:quarter].std(axis=0)
    tail = tensions[-quarter:]
    tail_mean = tail.mean(axis=0)
    tail_std = tail.std(axis=0)
    steady_peak_extension = float(extensions[-quarter:].max())
    decay_ratio = tail_std / np.maximum(head_std, 1e-9)
    expected_total = PAYLOAD_MASS * GRAVITY

    print("=" * 66)
    print("CABLE / PAYLOAD SUBSYSTEM STABILITY (quads kinematically pinned)")
    print("=" * 66)
    print(f"finite after {N_STEPS} steps            : {finite}")
    print(f"payload z: start={payload_z[0]:.4f}  end={payload_z[-1]:.4f}")
    print(f"payload z drift over last 25%         : "
          f"{abs(payload_z[-1] - payload_z[-len(payload_z) // 4]):.5f} m")
    print()
    print(f"steady tensions (mean)                : {np.round(tail_mean, 4)} N")
    print(f"osc. amplitude early (std)            : {np.round(head_std, 5)} N")
    print(f"osc. amplitude late  (std)            : {np.round(tail_std, 5)} N")
    print(f"decay ratio (late/early, <1 = damping): {np.round(decay_ratio, 4)}")
    print(f"sum of tensions                       : {tail_mean.sum():.4f} N")
    print(f"expected (payload weight)             : {expected_total:.4f} N")
    print(f"relative error                        : "
          f"{abs(tail_mean.sum() - expected_total) / expected_total * 100:.2f} %")
    print(f"peak extension, final 25%             : {steady_peak_extension * 1e3:.3f} mm")
    print()
    print(f"energy: initial={energies[0]:.4f}  final={energies[-1]:.4f}")
    print(f"energy growth (final/initial)         : "
          f"{energies[-1] / energies[0] if energies[0] else float('nan'):.4f}")
    print(f"max |energy| over trace               : {max(abs(e) for e in energies):.4f}")
    print()

    settled = bool(np.all(decay_ratio < 1.0))
    balanced = bool(abs(tail_mean.sum() - expected_total) / expected_total < 0.02)
    # Energy must not grow: a numerically unstable constraint injects energy.
    # Constraint projection is not perfectly energy-conserving.  The actual
    # stability evidence is decaying motion; permit sub-1% numerical drift.
    no_blowup = bool(abs(energies[-1]) <= 1.01 * abs(energies[0]))

    print(f"  [1/4] finite                        : {'PASS' if finite else 'FAIL'}")
    print(f"  [2/4] oscillation decaying          : {'PASS' if settled else 'FAIL'}")
    print(f"  [3/4] tension sum = payload weight  : {'PASS' if balanced else 'FAIL'}")
    print(f"  [4/4] no energy injection           : {'PASS' if no_blowup else 'FAIL'}")
    print()
    verdict = finite and settled and balanced and no_blowup
    print("VERDICT:", "CABLE PHYSICS SOUND" if verdict else "CABLE PHYSICS PROBLEM")
    print("=" * 66)


if __name__ == "__main__":
    main()
