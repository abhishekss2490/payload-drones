# Environment findings

## What works

- `PayloadTransport-v0` configuration constructs on CPU with a 12-dimensional team action.
- Reset and 100 random-action control steps over 4 environments completed with finite rewards and all five finite observation groups.
- Observation shapes measured after reset: policy `(B, 114)`, agent dynamics `(B, 45)`, history `(B, 6300)`, critic `(B, 169)`, and team probe `(B, 21)`.
- Independent per-agent dynamics sampling is working.  Over 512 teams, mean within-team arm-length sample standard deviation was **0.02963 m** (range sampled independently from 0.05–0.16 m).
- Actor observations were unchanged after perturbing `TeamState.payload_mass` and `TeamState.cable_length`; inactive-agent processed motor force was exactly zero (0 nonzero values for the disabled agent; 32 nonzero values across active-agent motors in the 8-env check).
- At a payload setpoint, the positive reward sum was exactly **1.0000**.  The two position-tracking kernels were below `1e-12` at a 3 m position error.

## Not verified / blocked

- The requested GPU acceptance test could not run.  In this execution environment `torch.cuda.is_available()` is **False** and Warp reports `CUDA driver not available`; therefore 4096 GPU environments were not constructed and no GPU throughput/OOM number is reported.
- `pytest` is not installed in the supplied synced venv (`No module named pytest`).  A network attempt to add it failed due DNS/network restriction.  The acceptance checks were run manually with `uv run --no-sync python` where possible; `tests/test_env_smoke.py` is present for a normal pytest-equipped environment.
- Neither `gymnasium` nor `gym` is installed.  Package import therefore uses conditional registration: it registers `PayloadTransport-v0` whenever Gymnasium is available, but cannot register it in this supplied venv.

## Specification inconsistencies found

1. Section 9 test 3 cannot require **every** independently sampled 3-agent team to have within-team arm-length standard deviation above 0.02 m.  Correct IID uniform samples occasionally land close together (the measured 512-team minimum was **0.00128 m**).  The test uses population mean within-team standard deviation instead, which directly checks the intended heterogeneity without introducing correlated/resampled team draws.
2. Section 6 assigns 0.50 total positive weight to orientation, linear velocity, and angular-velocity rewards.  At a 3 m **position-only** error with those other targets perfect, total positive reward is necessarily **0.5000**, not near zero.  The test records that value and checks the position components themselves go near zero; it does not assert the mathematically impossible total-reward claim.
3. The requested binary slack indicator necessarily depends on physical cable rest length, while constraint 5 says cable length must never be an actor observation.  The implementation treats it as a binary physical cable-state sensor and reads it from the simulation tendon state, not `TeamState.cable_length`; changing the privileged metadata does not affect actor observations.  Whether a slack sensor itself is allowed should be made explicit before claiming strict cable-length blindness.

## Constraint audit

- No reward terms target tension balance, equal thrust, or force allocation.
- No actor identity one-hot exists.
- Quad dynamics randomization is invoked once independently for each named quad and stored in separate `QuadState` entries.
- Agent history contains only that agent's policy row and motor command.
- Payload mass and numeric cable length are absent from the actor tensor; they are critic-only.

## Trajectory stage

### What works

- `PayloadTrajectoryCommand` is now the default command.  It uses the base repository's Lissajous/max-acceleration trajectory generator at `traj_dt = 0.010 s` (`physics_dt=0.005`, `trajectory_decimation=2`).  Its reference buffers are `(B, 1200, 3)`: 1,000 episode samples plus 200 preview samples.
- The command preserves the existing 13-slot layout, now fills `lin_vel_ref`, and keeps `quat_ref` as one static sampled payload-attitude setpoint per episode.  No moving attitude reference was introduced.
- The `payload_trajectory` observation is exactly `(B, 1200)` (`200 x [position(3), velocity(3)]`) and has no agent axis.  Its positions are relative to the current payload position.
- The static goal-reaching ablation remains available through `trajectory_enabled=False`; it retains zero velocity references while preserving the same observation shape.
- Trajectory difficulty now derives collective capability from `sum_i(4 * max_thrust_per_motor_i) / ((sum_i mass_i + payload_mass) * g)` and bins that team TWR in `curriculum.py`.
- The complete requested smoke command was run after the final change: `10 passed, 1 skipped, 45 warnings in 10.24s`.  The ten CPU tests include the new shared-shape, moving-velocity, end-of-buffer preview, and static-fallback checks.  The pre-existing GPU-only 4096-env test was skipped.

### What does not verify in this execution environment

- `torch 2.13.0+cu129` reports `torch.cuda.is_available() == False`; Warp reports `CUDA driver not available` / `no CUDA-capable device is detected`.  This conflicts with the supplied claim that an RTX A4000 is available to this session.
- Consequently, `uv run --no-sync python scripts/bench_env.py` produced the real output `no CUDA device; this benchmark is meaningless on CPU` before it could construct 4096 or 8192 environments.  There are no new GPU env-steps/s or GPU-memory measurements, and therefore no defensible throughput delta from the supplied 54,218 (4096) and 79,883 (8192) baselines.

### Specification audit

- No trajectory-stage code requirement was infeasible.  The only blocker is the runtime/hardware availability assertion above; GPU acceptance and the requested throughput comparison must be repeated in a CUDA-visible process before they can be claimed.

### GPU verification (run on real hardware, 2026-07-22)

The Codex sandbox genuinely had no CUDA — that report was accurate, not an excuse. Re-run in a
CUDA-visible process on the RTX A4000:

**All 11 tests pass**, including the 4096-env GPU test that was skipped.

End-to-end `env.step()` throughput, and the cost of the 1200-dim trajectory observation:

| num_envs | before traj | after traj | delta | GPU before | GPU after |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 21,889 | 24,571 | +12.3 % | 3.01 GiB | 3.25 GiB |
| 2048 | 48,361 | 44,477 | -8.0 % | 3.70 GiB | 4.30 GiB |
| 4096 | 54,218 | 50,661 | -6.6 % | 5.16 GiB | 6.27 GiB |
| 8192 | 79,883 | 79,802 | -0.1 % | 8.01 GiB | 9.77 GiB |

**The trajectory observation is nearly free at the operating point.** At 8192 envs it costs 0.1 %
throughput and 1.76 GiB, still leaving ~6 GiB headroom on the 15.6 GiB card. Small-batch numbers are
launch-overhead dominated and noisy; ignore them. No need to stride the window.

### Test 9 was wrong twice — corrected

`test_9_moving_trajectory_populates_velocity_reference` passed on CPU but failed on GPU. Neither
version was detecting an implementation defect; both were badly timed tests of my own writing.

1. **Flaky.** It stepped 20 times and asserted speed > 1e-5. Trajectories ramp in with a quintic
   smoothstep whose speed goes as `30*u^2*(1-u)^2`, so at `u = 0.02` the reference is legitimately
   ~1 % of peak. Unseeded, it failed 3 runs in 5 depending on the sampled trajectory.
2. **Physically impossible.** The fix stepped to mid-episode with zero motor commands — but zero
   thrust means the team falls, terminations fire, and the env resets. `episode_step` never leaves
   ramp-in, so it still read ~1e-7. The test depended on flight competence it had no business
   requiring.

Now seeded and indexes the buffer directly at a mid-episode step (as test 10 does), asserting
speed > 1e-3. Deterministic: 3/3 and 5/5 stable. The velocity references were correct throughout —
peak buffer speed comfortably exceeds 0.1 m/s.

## Feasibility and speed-cap stage

- The prescribed conditional payload formula was evaluated over 200,000 independently sampled
  teams with the current `ScaleCfg` distribution.  It gives **81.01 %** feasibility at the stated
  1.5x total-team hover-margin check, not 100 %.  Payload mass was min 0.1000 kg, median 0.5561 kg,
  max 2.0000 kg, with 5th/95th percentiles 0.2071/1.1984 kg.
- This is a mathematical conflict in the new requirements, not an implementation issue:
  `m_payload = f * (T/g - M_quad)` does not make total team TWR equal to `1/f`; the latter omits
  the quads' own mass.  At the allowed `f = 0.55`, low-TWR teams fail the requested 1.5x margin.
  The 0.10 kg lower clamp is also above the allowable 1.5x-margin payload for some light teams.
- `torch.cuda.is_available()` is **False** in this session, so the requested GPU full-suite run and
  4096/8192 benchmark were not run.  A CPU subset covering the new speed-cap path passed before
  the unresolved feasibility work was reverted: `3 passed, 10 deselected`.

### Corrected conditional sampling and speed cap (2026-07-22)

- The revised loaded-team-TWR formula is mathematically feasible and has replaced independent
  payload sampling.  Each reset samples `U(1.5, min(3.0, 0.9 * unloaded_team_twr))`, derives
  payload mass from it, applies only the 2.5 kg upper cap, and stores the **realized** loaded TWR
  as critic-only `TeamState.margin`.  No minimum payload clamp is used.
- Real output from `scripts/check_lift_feasibility.py` over 200,000 seeded independently sampled
  teams: **100.00%** feasible at 1.5x margin and **100.00%** positive payload.  Realized loaded
  TWR was **1.5000..3.0000** (median **2.0374**).  Payload mass min / 5th / median / 95th / max
  was **0.0106 / 0.0771 / 0.2885 / 0.9040 / 2.5000 kg**.
- `PayloadTrajectoryCommandCfg.max_payload_speed` defaults to **1.5 m/s**.  It performs the
  specified single acceleration rescale after measuring the generated reference.  Because the
  base generator returns a fixed early-episode window, a non-iterative analytic velocity envelope
  additionally prevents a previously unseen later peak from entering the shorter regenerated
  window; path geometry is unchanged.  The seeded 512-trajectory CPU measurement was
  **1.499999 m/s**, below the **1.575000 m/s** acceptance bound.
- Two seeded acceptance tests cover exactly these conditions: 4096 reset teams must have positive,
  1.5x-margin-feasible payloads, and 512 generated reference buffers must respect the speed cap.
  Full CPU suite result: **12 passed, 1 skipped**.  The skipped test is GPU-only; CUDA is not
  available (`torch.cuda.is_available() == False`), so no GPU result is reported.

### Analytic single-pass speed cap (2026-07-22)

- The trajectory generator no longer performs a generate/measure/regenerate cycle.  It uses the
  closed-form Lissajous speed envelope directly and takes the minimum of that acceleration limit
  and the curriculum-requested acceleration, then generates once.  The speed-cap test already
  asserted only the required upper bound, so it needed no change.
- The seeded 512-trajectory CPU peak-speed distribution (min / median / max) is
  **0.310800 / 0.535684 / 0.847396 m/s**; **508/512** trajectories peak below 0.8 m/s.  The
  conservative analytic bound therefore makes the present default too sluggish for the stated
  task-interest threshold.  Raise `max_payload_speed` before using this as the bring-up workload
  if a median of at least 0.8 m/s is desired.

### Exact path-speed cap and curriculum semantics (2026-07-22)

- Replaced the loose simultaneous-sinusoid bound with numerical maxima of `lissajous_path`'s
  actual `p'(s)` and `p''(s)` over one full loop.  The command now converts both the curriculum
  acceleration request and the speed ceiling to `omega`, takes their minimum, and composes the
  base repository's `compute_timing` and `compose` utilities directly.  There is one trajectory
  generation pass and no copied timing or derivative math.
- Seeded 512-trajectory CPU measurement, min / median / max peak reference speed:
  - minimum curriculum (`rho=0.05`): **0.311397 / 0.549062 / 0.853810 m/s**;
  - maximum curriculum (`rho=1.00`): **1.392612 / 1.499989 / 1.500010 m/s**
    (509/512 within 3% of the 1.5 m/s ceiling).
- This reveals a remaining semantic constraint, not a numerical-bound defect.  The current
  curriculum samples *acceleration*, so at a fixed `rho` its uncapped speed is
  `max_dp * sqrt(a / max_ddp)`: it necessarily varies with path geometry and team TWR.  The speed
  ceiling controls speed tightly only once it binds.  A tight fixed-difficulty speed-spread test
  would be false for the present acceleration curriculum; implementing one honestly requires a
  curriculum expressed in speed/omega (or a test that explicitly isolates the cap-bound regime).
