# Emergent Cooperative Payload Transport with Physically Heterogeneous Quadrotors

**Status**: design locked, implementation starting
**Target**: ICRA (deadline ~mid-Sept 2026)
**Base repo**: `/scratch/robotics/mj-adapt-drones` (mjlab / mujoco-warp, GPU)

---

## 1. The two parents

| | [CoRL 2025 — Zeng et al.](https://arxiv.org/abs/2508.01522) | [IROS 2025 — Vaidya & Keshavan](https://arxiv.org/abs/2503.09622) |
|---|---|---|
| Setting | 3 MAVs, cable-suspended load, 6-DoF | Single quadrotor, trajectory tracking |
| Action space | linear acceleration + body rates | **direct motor forces** |
| Adaptation | robustness via randomization | **scale-aware DR + RMA latent** (arm length 0.05–0.16 m) |
| Coordination | implicit, via load pose observation only | n/a |
| Heterogeneity | **algorithmic** (one MAV runs a different controller) | **physical** (30 g → 2.1 kg, same policy) |
| Role assignment | **one-hot identity vector `e_i`** | n/a |
| Adaptation input | obs history H=3 | TCN over 50-step state-action history |

### CoRL setup, for baseline fidelity
3 MAVs @ **0.6 kg each — fully homogeneous**. Payload 1.4 kg, cable 1 m. MAPPO, CTDE, centralized
critic + parameter-shared actor, MLP [1024, 512, 256, 128]. Policy 100 Hz, INDI low-level 300 Hz.
Isaac Lab, 4096 envs, RTX 3090, 17 h. Rewards: position, orientation, **downwash avoidance**, action
smoothness, body-rate penalty, thrust efficiency. **No inertia randomization during training** —
robustness comes from the INDI controller's accelerometer feedback, not from learned adaptation.

## 2. The gap

CoRL's action space **hides physical heterogeneity**: acceleration + body-rate commands are absorbed
by a per-platform low-level controller, so the MARL policy never sees whether it is flying a 30 g or a
2 kg airframe. Their "heterogeneity" result is about *policies*, not *platforms*.

Direct motor-level control removes that shield. The policy *becomes* the low-level controller, so a
40× mass spread across teammates is now the policy's problem — which is exactly the problem the IROS
scale-aware RMA formulation already solves for one drone.

Two further shields in CoRL that we remove:
- **The INDI low-level controller absorbs model error.** They apply no inertia randomization at all;
  accelerometer feedback handles disturbances. Motor-level control has no such crutch, so the learned
  adaptation must carry that weight itself.
- **A one-hot identity vector `e_i` assigns roles.** In a physically homogeneous team this is the
  *only* available symmetry-breaker — there is nothing else distinguishing agent 1 from agent 2.

**Nobody has done physically heterogeneous multi-quad payload transport at motor level.**
(Checked: RoVerFly `2509.11149`, FLARE `2508.09797`, ASTER — all single-quadrotor.)

## 3. Core claim

> In a cable-coupled team, an RMA latent trained **only** to identify *ego* dynamics is forced to
> also encode *team* state — because cable tension makes the payload a communication channel and the
> state-action history its receiver.

This reframes RMA from *system identification of self* to *system identification of self-in-a-team*.
It is what makes this a merge rather than a concatenation: neither parent paper predicts it.

## 4. Falsifiable hypotheses

**H1 — Emergent capability-proportional load sharing.**
With **no load-allocation term anywhere in the reward**, agent *i*'s steady-state cable tension share
correlates with its thrust headroom `(TWR_i − 1) · m_i · g`.
*Metric*: Pearson r between tension share and headroom share over randomized teams.
*Null*: r ≈ 0 (equal sharing regardless of capability).

**H2 — Emergent redistribution under agent loss.**
On mid-flight disablement of one agent, survivors redistribute tension without communication, because
the disturbance propagates through the payload into their state-action history.
*Metric*: redistribution settling time + payload pose deviation, vs. an adaptation-module-ablated baseline.

**H3 — The latent does double duty.**
`z_dyn` linearly decodes payload mass and team composition, despite being trained **only** on an
ego-dynamics distillation target.
*Metric*: linear-probe R² for payload mass / teammate arm lengths vs. shuffled-label control.
*Reuse*: `codex-dynamics-eval/` maps ~1:1 (`representation_in_context`, `correlation_tests`,
`latent_distribution`, `umap_visualization`, `memorization_tests`).

**H4 — The capability latent replaces the one-hot role index.**
CoRL breaks agent symmetry with a one-hot identity `e_i`, necessary because their team is physically
homogeneous. We claim that in a heterogeneous team the platform's own dynamics break the symmetry, so
`z_dyn` subsumes `e_i`: roles become continuous and physically grounded rather than discrete and assigned.
*Metric*: ours vs. ours+one-hot vs. one-hot-only (no `z_dyn`). Prediction — adding `e_i` on top of
`z_dyn` yields no significant gain, while `e_i` alone degrades badly under heterogeneity, **and**
one-hot-only fails to transfer when agent indices are permuted at test time whereas `z_dyn` is permutation-robust.

H3 is the direct mechanistic evidence for the core claim. H1/H2 are the behavioral consequences.
H4 is the cleanest head-to-head against the CoRL formulation.

## 5. Experimental grid

Primary 2×2 — the claim is that the bottom-right cell is where baselines break:

| | Homogeneous team | **Heterogeneous team** |
|---|---|---|
| acc + body-rate (CoRL-style) | baseline A | baseline B |
| **motor-level (ours)** | baseline C | **ours** |

Ablations: (a) no dynamics encoder, (b) frozen/blind adaptation module, (c) explicit load-allocation
reward term — H1 should show this is *unnecessary*, (d) teammate-state observation oracle — upper bound
on what implicit communication costs, (e) one-hot `e_i` variants for H4, (f) history length H ∈ {3, 50}
— isolates whether the win comes from *adaptation* or merely from *more history* (CoRL uses H=3).

**Baseline fidelity**: baselines A/B reproduce CoRL's formulation — MAPPO, CTDE centralized critic,
parameter-shared actor, one-hot `e_i`, their six reward terms including downwash avoidance, acc+body-rate
actions through an INDI-equivalent low-level controller. We must implement that low-level layer honestly,
or baseline B is a strawman and a reviewer will say so.

## 6. Scope decisions (locked)

- **Task**: 6-DoF payload pose tracking. 3 cables at 3 distinct attachment points → 6-DoF controllable.
- **Team**: fixed N=3, per-agent dynamics randomized every reset. Agent-masking infra built from day
  one (required for H2 regardless; leaves a variable-N ablation cheap).
- **Sim2real**: sim-only for this paper, but every choice stays transfer-compatible — motor-level
  actions, observation delays, motor lag `τ`, no privileged info at deployment.
- **No explicit communication, no teammate-state observations** at deployment. Payload pose only.

## 7. Physics scaffold (verified against installed mujoco-warp)

- Cables = **spatial tendons** with `limited="true" range="0 L"`. Unilateral: pulls when taut, slack
  otherwise — physically correct, and cheaper than a rigid-rod chain.
  Confirmed kernels: `_spatial_site_tendon`, `_spatial_tendon_wrap`, `_limit_tendon`.
- `tendon_range`, `tendon_length0`, `tendon_stiffness/damping` are per-world batched
  (`array("*", "ntendon", ...)`) → **cable length randomizable per env**, same pattern as the existing
  `site_pos` randomization.
- Cables cross entity boundaries, so tendons must be added at **scene level post-`attach()`**.
  Precedent already in the base repo: `SceneCfg(spec_fn=_set_air_drag)`.
- Integrator stays **Euler** (mujoco-warp does not support implicit + fluid model; base repo already
  relies on this for inertia-box air drag).

## 8. Known risks

| Risk | Mitigation |
|---|---|
| **Compute**: RTX A4000 16 GB; 3 quads + payload + tendons per env | Benchmark `num_envs` early; expect ~1–2 k, not 4 k. Constraint solver cost from tendon limits is the unknown. |
| Tendon limit constraints may need `njmax` tuning / cause solver stalls | Part of the scaffold validation task — measure before building on it. |
| 6-DoF from cables is underactuated & yaw-weak | May need payload attachment geometry tuning; check controllability early. |
| "Emergence" reads as hand-waving | Every claim is pre-registered above with a metric and a null. |
| Training instability from simultaneous multi-agent motor-level exploration | Curriculum: start payload-light + cables short; reuse existing curriculum infra. |

## 9. Build order

1. **Physics scaffold + validation** ← highest risk, do first
2. Port DR to per-agent heterogeneous teams (extend `quad_scaling.py`)
3. Payload 6-DoF command/reward/termination terms
4. Multi-agent policy wiring (shared policy, agents folded into batch dim)
5. Phase-1 PPO (teacher) → Phase-2 latent distillation (student)
6. Baselines + ablations + H1/H2/H3 analyses
