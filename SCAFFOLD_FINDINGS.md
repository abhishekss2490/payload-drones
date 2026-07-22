# Spatial-tendon scaffold findings

**Verdict: SOUND. Proceed to RL work.**

> **Correction note.** An earlier revision of this document concluded "STOP, do not start RL work,"
> based on an open-loop test that commanded equal fixed thrust to three *free-flying* quadrotors and
> flagged the resulting 39.6 m drift as a scaffold failure. **That test was invalid.** A quadrotor is
> open-loop unstable in attitude; three free quads plus a free 6-DoF payload at fixed equal thrust is
> an unstable equilibrium with no stabilizer. Drifting is correct physics, not a defect in the cable
> model. The test specification was wrong, not the scaffold. It has been replaced by
> `scripts/check_cable_physics.py`, which pins the quadrotors kinematically to isolate the cable and
> payload subsystem — the thing actually under test.

## Cable representation

Cables are **spatial tendons** between `quad_{i}/hook` and `payload/attach_{i}`, added at scene level
in `SceneCfg.spec_fn` after mjlab's `attach()` (they cross entity boundaries, so they cannot live in
any single entity's spec). Site names were established by inspecting the assembled spec, not guessed.

Each tendon is an explicitly enabled **upper-limit constraint**: `limited=True`,
`range=[0, L]`, and `solref_limit=[0.01, 1.0]`. It is slack below `L` and pulls
only at its upper bound, so it is unilateral by construction. `stiffness=0` and
`damping=0`: an explicit Euler-integrated spring cannot be both stable for a
26 g quadrotor and credibly inextensible. The prior spring-only characterization
was wrong: setting `range` had already enabled MuJoCo's auto limit, so the old
implementation unintentionally combined incompatible mechanisms.

| Check | Measurement | Result |
| --- | ---: | --- |
| Compiles, `ntendon == 3` | 3 tendons, range `[0, 0.6]` | pass |
| Taut (payload 5 mm lower) | active tendon-limit constraint row | pass, pull-only |
| Slack (payload 100 mm higher) | length 0.500 m, 0 N/cable | pass |
| Symmetric static load share | 2.4525 N/cable = `0.75·9.81/3` exactly | pass |

## Cable/payload subsystem stability

`scripts/check_cable_physics.py` — quads pinned kinematically, payload displaced 0.10 m laterally and
released, 4000 steps (20 s at 5 ms), energy diagnostics enabled via `mjENBL_ENERGY`.

| Check | Measurement | Result |
| --- | ---: | --- |
| Finite after 4000 steps | no NaN | pass |
| Oscillation decays | amplitude ratio late/early = 0.166, 0.076, 0.083 | pass |
| Tension sum = payload weight | 7.3853 N vs 7.3575 N — **0.38 % error** | pass |
| Constraint stretch, final 25 % | 1.611 mm peak | pass |
| Numerical energy drift | ratio 1.0024 (below 1 % tolerance) | pass |
| Payload z drift, last 25 % | 1.10 mm | pass |

## GPU throughput (RTX A4000, 16 GB)

Measured on the real device. `njmax=128`, `nconmax=64`, Euler, 5 ms timestep.

| num_envs | steps/s |
| ---: | ---: |
| 256 | 47,905 |
| 512 | 78,664 |
| 1024 | 119,394 |
| 2048 | 163,802 |
| **4096** | **204,023** |

**No OOM at 4096 envs** — we match CoRL's parallelism on a much weaker GPU than their RTX 3090.
Throughput scales near-linearly to 4096, so 4096 is the operating point.

Budget implication: at ~204 k physics steps/s, and assuming PPO overhead leaves ~30–50 % of that,
a 1e9-step run lands around 2–3 h. A two-phase pipeline over ~30 configs/seeds is therefore on the
order of days, not months. **Compute is not the binding constraint.**

> Caveat: `peak=0.00 GiB` in the sweep output is a measurement bug, not a real reading —
> mujoco-warp allocates outside torch's caching allocator, so `torch.cuda.max_memory_allocated()`
> cannot see it. Use `torch.cuda.mem_get_info()` or `nvidia-smi` to get true peak memory.

## Constraint allocation

At the nominal pose `nefc = 79`: 76 actuator-limit rows plus 3 tendon upper-limit rows.
`mujoco-warp.put_data` rejects `njmax=64` and accepts at `njmax >= 79`. **Use `njmax=128`** for
headroom. `nconmax=64` is sufficient for the non-contact nominal configuration. The tendons
contribute only 3 rows — they are not the dominant allocation cost.

## Open items for the next stage

1. **Constraint cable is deliberate.** Keep its 3 upper-limit rows; `njmax=128` has headroom over
   the measured `nefc=79`. Do not reintroduce an explicit spring without a different integrator.
2. **True peak memory is still unmeasured** — fix the instrumentation before sizing the largest run.
3. **Attachment geometry is untuned.** Verify 6-DoF controllability of the payload — particularly
   yaw authority, which is the weak axis for three near-vertical cables.
