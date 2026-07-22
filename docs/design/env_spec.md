# Environment specification — `PayloadTransport-v0`

Implementation spec for the cooperative heterogeneous payload-transport RL environment.
Read `RESEARCH_BRIEF.md` for the hypotheses this environment must be able to test, and
`docs/design/marl_wiring.md` for the actor/critic wiring.

**Base repo:** `/scratch/robotics/mj-adapt-drones` (editable dep, importable as `mj_adapt_drones`).
Reuse aggressively; do not reimplement anything that already exists there.

---

## 0. Non-negotiable design constraints

These exist because the science depends on them. Violating any one invalidates a hypothesis.

1. **No explicit load-sharing / force-allocation reward term.** H1 asks whether load sharing
   *emerges* from payload tracking alone. A reward that pays for balanced tension makes H1
   unfalsifiable and the result trivial. Reward the payload's motion, never the allocation.
2. **No one-hot agent identity in the actor observation.** H4 is precisely the claim that an
   inferred capability latent replaces CoRL's assigned index `e_i`. A one-hot must exist only as
   an *ablation-time* switch, off by default.
3. **Per-agent independent dynamics sampling.** Each of the 3 quads draws its own `arm_length`
   independently, so teams are genuinely heterogeneous (a 0.05 m and a 0.16 m quad on the same
   payload). Sampling one team-wide value would silently kill H1/H2/H3.
4. **The dynamics latent is supervised with ego-only targets.** Agent *i*'s adaptation module sees
   only agent *i*'s own state/action history and is regressed only onto agent *i*'s own parameters.
   H3 (the latent nevertheless encodes team state) is only meaningful under this constraint.
5. **Payload mass and cable length are never in the actor observation.** They are privileged.

---

## 1. Scene

Already implemented in `src/payload_drones/scene/builder.py`. Do not rewrite. It provides
3 quads (`quad_0`, `quad_1`, `quad_2`) + `payload`, joined by 3 spatial-tendon cables, with
air drag set on the root spec.

Cables are **deadband springs** (`springlength = [0, L]`), not limit constraints — validated in
`SCAFFOLD_FINDINGS.md`. Simulation config: `njmax=128`, `nconmax=64`, Euler, `timestep=0.005`.

Entity naming is fixed: quads are `quad_{i}` for `i in 0..2`, payload is `payload`.
Cable tendons are `cable_{i}`. Sites are `quad_{i}/hook` and `payload/attach_{i}`.

---

## 2. Team state and domain randomization

**New module:** `src/payload_drones/mdp/team_scaling.py`

### `TeamState`

```python
@dataclass
class TeamState:
    agents: list[QuadState]          # length N_AGENTS, each is the base repo's QuadState
    payload_mass: torch.Tensor       # (num_envs,)
    payload_inertia: torch.Tensor    # (num_envs, 3)
    cable_length: torch.Tensor       # (num_envs, N_AGENTS)
    active_mask: torch.Tensor        # (num_envs, N_AGENTS) bool — False = failed agent
```

Reuse `mj_adapt_drones.mdp.quad_scaling.QuadState` verbatim; do not redefine it.

### Randomization event

`team_dynamics_randomization(env, env_ids, scale_cfg, ...)`:

- Calls the base repo's `quadrotor_dynamics_randomization` **once per quad entity**, with
  `asset_cfg=SceneEntityCfg(f"quad_{i}")` and matching site/actuator cfgs. That function is already
  parameterized by `asset_cfg`, so per-agent application needs no fork of its internals.
- Each call must draw its **own independent** `arm_length`. Verify this — if the base function
  writes into a single `env.quad_state`, capture the result per agent into `TeamState.agents[i]`
  rather than letting calls clobber each other.
- **Payload mass is conditioned on team lift capacity — NOT sampled independently.** See 2b.
- Randomizes cable length over `cable_length_range = (0.5, 0.8)` m by writing `tendon_springlength`
  (upper entry) and `tendon_range` per env.
- Populates `env.team_state`.

Startup event `initialize_team_state` mirrors `initialize_quad_state`.

**Agent failure (for H2):** `active_mask` starts all-True. A separate event
`sample_agent_failure(env, env_ids, p_failure, min_active=2)` optionally flips one agent to False.
Default `p_failure=0.0` — failure is an *evaluation* condition, not a training one, unless a config
explicitly enables it.

---

## 2b. Payload mass must be conditioned on team capacity (added 2026-07-22)

**The original spec sampled payload mass independently over `(0.2, 1.5)` kg. That is wrong and must
be replaced.** Measured over 200k sampled teams with `scripts/check_lift_feasibility.py`:

| Quantity | Measured |
| --- | --- |
| Per-quad mass | 0.021 .. 0.831 kg |
| Team spare lift capacity | 0.140 .. 5.602 kg (median 1.435) |
| **Teams that cannot hover with their payload** | **20.23 %** |
| Teams infeasible at a 1.5x hover margin | 40.74 % |

One episode in five was physically unlearnable. Worse, infeasibility correlates with small/weak
teams — precisely the population the paper is about — so it would have biased results toward
"heterogeneous teams perform poorly" when the real cause is that we asked impossible teams to do
impossible things. That is a fabricated result, not a finding.

The cause is structural: team spare capacity spans a 40x range, so **no fixed payload range can be
both feasible for weak teams and non-trivial for strong ones.**

### Required approach — randomize the LOADED team thrust-to-weight directly

> **An earlier revision of this section was mathematically wrong.** It sampled
> `payload = load_factor * (T/g - M)` with `load_factor ~ U(0.25, 0.55)` and claimed loaded team TWR
> equals `1/load_factor`. That ignores that the quads must also lift *themselves*. Solving the real
> constraint gives `S > 1.5*M*(1-lf)/(1-1.5*lf)`, which at `lf = 0.55` demands unloaded team TWR
> > 3.86 — outside `ScaleCfg.twr_range = (2.0, 4.0)` for most teams. Measured feasibility was
> **81.01 %**, not 100 %. The 0.10 kg minimum clamp was also infeasible for the lightest teams,
> whose entire margin-safe payload is ~27 g.

Randomize the loaded team thrust-to-weight ratio itself. It is the physically meaningful quantity,
it is what the curriculum bins on (section 4b), and it makes feasibility automatic:

```
S        = sum_i(max_total_thrust_i) / g     # total liftable mass
M        = sum_i(m_i)                        # total quad mass
team_twr = S / M                             # unloaded, in (2, 4) via ScaleCfg

margin   ~ U(1.5, min(3.0, 0.9 * team_twr))  # LOADED team TWR
payload_mass = min(S / margin - M, 2.5)      # > 0 by construction
```

The `0.9 * team_twr` upper bound guarantees `margin < team_twr`, hence `payload_mass > 0`, for every
team including the weakest. **Clamp the maximum only** — an upper clamp reduces payload and so only
increases margin, while a *minimum* clamp is exactly what broke the previous version.

Store `margin` (the realized loaded team TWR) in `TeamState`. It is privileged/critic-only and the
natural analysis variable.

Verified over 200k sampled teams:

| Quantity | Old formula | This formula |
| --- | ---: | ---: |
| Feasible at 1.5x margin | 81.01 % | **100.00 %** |
| Realized loaded team TWR | — | 1.500 .. 3.000 (median 2.040) |
| Payload mass | — | median 0.288 kg, 95th pct 0.904 kg |
| Payload / team mass | — | median 0.417, 5-95 pct 0.138 .. 1.017 |
| Payloads under 50 g | — | 1.31 % |

The task does not collapse to trivial payloads: the median payload is 42 % of total team mass and
sometimes exceeds it.

**Acceptance:** over 4096 sampled teams, 100 % must satisfy
`sum_i(max_thrust_i) > 1.5 * (sum_i(m_i) + m_payload) * g`, and 100 % must have `payload_mass > 0`.

## 2bb. Cable stiffness and damping must scale with the lightest quad (added 2026-07-22)

**The fixed cable parameters `stiffness=2000 N/m`, `damping=80 N.s/m` are numerically unstable for
the lighter half of the randomization range, and diverge within 2 simulation steps.**

Found by running the actual trainer, not by any unit test. Diagnostic at 4096 envs with random
actions: step 0 max quad-payload distance 0.775 m; **step 2: 6.922 m with max |qvel| 47,497**; step
19: 2600 m. A quadrotor cannot fly 6.9 m in 0.01 s — the cable spring launches it, then the growing
extension feeds back as a larger force.

Those values were chosen against the 0.75 kg payload of the scaffold test, where the quads were
*pinned* (effective mass 0.75 kg, `omega*dt = 0.26`, stable). They were never re-checked against the
**lightest randomized quad at 0.021 kg**. For explicit Euler at `dt = 0.005`:

| | value | Euler stability limit at m = 0.021 kg | over by |
| --- | ---: | ---: | ---: |
| stiffness `k` | 2000 N/m | `0.25*m/dt^2` = 210 N/m | 9.5x |
| damping `c` | 80 N.s/m | `0.5*m/dt` = 2.1 N.s/m | **38x** |

Damping is the worse offender: `(c/m)*dt = 19` where stability needs `< 2`.

`implicitfast` would absorb this, but mujoco-warp raises
`NotImplementedError: Implicit integrators and fluid model not implemented` — it is incompatible
with the air-drag fluid model (`density=1.2`, `viscosity=1.8e-5`) inherited from the base repo. Do
not silently drop air drag to enable it.

### Required approach

Scale the cable parameters per env with **that team's lightest quad**, in
`team_dynamics_randomization` after the quads are sampled, writing `tendon_stiffness` and
`tendon_damping`:

```
m_min = min_i(m_i)                                  # per env
k = min(2000.0, K_SAFETY * m_min / dt**2)           # K_SAFETY = 0.25
c = min(  80.0, C_SAFETY * m_min / dt)              # C_SAFETY = 0.5
```

This holds `omega*dt` roughly constant across the randomization range instead of letting it vary
40x, and it is physically sensible: a tether sized for a 20 g micro-quad is not the one used for an
800 g machine. The existing constants become ceilings for heavy teams rather than fixed values.

Do **not** fix this by narrowing `arm_length_range` — heterogeneity is the entire point of the
project, and trading it away for numerical convenience would hollow out H1-H4.

**Acceptance:** 4096 envs, random actions, 400 steps: all states finite, and max quad-payload
distance stays bounded (no env exceeds ~5 m, versus 2600 m before).

## 2c. Trajectory speed cap for initial testing (added 2026-07-22)

Cap payload reference speed at **~1.5 m/s** while bringing the system up. This is a soft target for
early testing, not a physical constraint — expose it as a config parameter
(`max_payload_speed: float = 1.5`) so it can be raised once tracking works.

The base generator is parameterized by `max_acceleration`, not speed, so derive it. For a fixed path
shape, scaling time by `k` gives `v -> v/k` and `a -> a/k^2`. So generate, measure peak speed, then
rescale acceleration by `(v_target / v_peak)^2`. Because the path geometry is unchanged and only the
angular rate moves, **one correction is exact** — do not iterate or binary-search.

Keep trajectories geometrically simple at this stage; aggressive maneuvers come later.

**Acceptance:** over 512 sampled trajectories, peak reference speed must not exceed
`1.05 * max_payload_speed`.

## 3. Actions

**New module:** `src/payload_drones/mdp/actions.py`

`TeamThrustAction`: 12-dim (3 agents x 4 motors). Wrap the base repo's
`QuadrotorThrustAction` per entity rather than reimplementing motor mapping, thrust curves, or
motor-lag handling.

Inactive agents (`active_mask == False`) have their commanded thrust forced to zero before it
reaches the actuators. Apply the mask in the action term, not the policy — a failed rotor produces
no thrust regardless of what the network outputs.

---

## 4. Commands

**New module:** `src/payload_drones/mdp/payload_command.py`

`PayloadPoseCommand` — a 6-DoF payload pose setpoint, following CoRL's goal-sampling structure
rather than the base repo's full trajectory machinery (which is built for single-quad differential
flatness and does not transfer to a cable-suspended load).

- Position: uniform in a box around the env origin, `x,y in (-1.5, 1.5)`, `z in (0.8, 2.0)`.
- Yaw: uniform in `(-pi, pi)`. Roll/pitch: uniform in `(-0.3, 0.3)` rad, small — three near-vertical
  cables have weak roll/pitch authority and unreachable goals just add reward noise.
- Resample on reset; optionally resample mid-episode on a timer (`resampling_time_range`).
- `command` tensor layout: `[pos_err_w (3), quat_ref (4), lin_vel_ref (3), ang_vel_ref (3)]`.
  Velocity references are zero for setpoints; keep the slots so trajectory-following can be
  dropped in later without changing observation shapes.

Curriculum (`src/payload_drones/mdp/curriculum.py`): start with a tight goal box and expand as
tracking error falls. Reuse the base repo's curriculum term structure.

---

## 4b. Payload reference trajectory (added 2026-07-22)

The static setpoint of section 4 is **superseded** by a time-varying payload trajectory. The
setpoint version remains valid as an ablation ("goal-reaching, no preview") but is no longer the
default.

**Why preview matters here.** A cable-suspended payload is a pendulum with period
`T = 2*pi*sqrt(L/g)` — 1.55 s at L = 0.6 m, 1.79 s at L = 0.8 m. Control runs at 100 Hz
(`decimation=2` on a 5 ms step). A 200-step window is therefore 2.0 s, or ~1.3 pendulum periods:
precisely the anti-sway preview horizon. Reactive setpoint tracking on a pendulum overshoots and
pumps swing; anticipation lets the team pre-shape motion. This is classic crane input-shaping.

### Content and horizon (decided)

- **Trajectory carries position + velocity only: 6 dims per waypoint**, matching the base repo
  exactly so `TrajectoryEncoder`'s defaults (`num_waypoints=200`, `features_per_waypoint=6`) work
  unchanged.
- **Orientation stays a static setpoint** (the existing quat slots). Three near-vertical cables
  have weak roll/pitch authority over the load; a time-varying attitude reference would be largely
  untrackable and unreachable references mostly inject reward noise. The paper therefore claims
  6-DoF *stabilization* with 3-DoF *trajectory tracking* — state this explicitly, do not blur it.
- **Horizon: 200 steps / 2.0 s.**

### Implementation

`src/payload_drones/mdp/payload_command.py` — extend to `PayloadTrajectoryCommand`.

- Reuse the base repo's `mj_adapt_drones.mdp.trajectory` generators (`generate_trajectory`,
  `lissajous_path`, `compose`, `compute_timing`, `generate_trajectory_for_max_acceleration`)
  to produce smooth payload paths with derivatives. Do not write new trajectory math.
- Mirror `TrajectoryCommand`'s buffer structure: a buffer at `traj_dt = physics_dt *
  trajectory_decimation` resolution, sized `N_episode + observation_horizon` so the lookahead never
  runs off the end.
- Populate the previously-zero velocity slots of the command tensor.

### Curriculum by team thrust-to-weight

The base repo bins curriculum difficulty by single-quad TWR. The team analogue is

```
team_twr = sum_i(max_thrust_i) / ((sum_i(m_i) + m_payload) * g)
```

Bin on this. It is a natural extension and a genuinely good fit: heterogeneous teams have different
collective capability, so trajectory difficulty should scale with what the *team* can actually do,
not with any individual agent.

### New observation group `payload_trajectory` — SHARED, not per-agent

Shape `(num_envs, 200 * 6) = (num_envs, 1200)`. Waypoint positions are expressed **relative to the
current payload position** so the encoder sees a translation-invariant window.

**This group has no agent axis.** All three agents track the same payload reference, so the encoder
runs *once per env* and its latent is broadcast to all agents — one third the cost of a naive
per-agent encoding. Do not replicate this tensor across agents.

Resulting per-agent model latent (for the model stage, not this one):
`[own_policy_obs (38), z_traj (16, shared), z_dyn (8)] = 62`.

## 5. Observations

**New module:** `src/payload_drones/mdp/observations.py`

Per-agent quantities are computed for all agents and returned **stacked**, shape
`(num_envs, N_AGENTS, per_agent_dim)`, flattened to `(num_envs, N_AGENTS * per_agent_dim)` only at
the manager boundary. Keep the agent axis explicit inside the functions — the models fold it into
the batch dimension and silent reshapes here are the easiest way to corrupt that.

### Group `policy` (actor, per-agent, **non-privileged**)

| Term | Dim | Notes |
| --- | ---: | --- |
| own position relative to payload, world frame | 3 | |
| own linear velocity relative to payload | 3 | |
| own orientation (6D rotation repr, not quat) | 6 | 6D avoids quaternion sign ambiguity |
| own body angular velocity | 3 | |
| own last action | 4 | |
| own cable unit vector + slack indicator | 4 | direction hook→attach, and a taut/slack flag |

**Cable sensing is an explicit modeling assumption.** The slack flag is a *physical* cable-state
sensor (read from the simulation's tendon state — equivalent to a load cell or cable-angle encoder
at the hook), not a comparison against the privileged `TeamState.cable_length`. Perturbing that
metadata leaves the actor observation unchanged, so constraint 5 holds. This is deliberate and
strengthens rather than weakens the mechanism claim: agent *i* sensing its own cable is ego
information, and H3 asserts that the *history* of such ego sensations reveals *team* state. An
ablation with the cable channel removed tests whether it is load-bearing.
| payload pose error vs goal (pos 3 + 6D rot) | 9 | |
| payload linear + angular velocity | 6 | |

Total **38** per agent. **No one-hot identity. No payload mass. No cable length. No other agent's
state** — inter-agent coordination must route through the payload, which is the whole point.

### Group `agent_dynamics` (privileged, per-agent — the RMA teacher target)

The base repo's 15-dim ego parameter vector, per agent: `arm_length, mass, inertia(3), ipos(3),
iquat(4), km_kf, motor_tau, twr`. Reuse the base repo's observation function per entity.

### Group `agent_history` (adaptation-module input, per-agent)

Rolling window of length 50 over that agent's **own** `(policy_obs, action)`. Shape
`(num_envs, N_AGENTS, 50, 38+4)`. Ego-only, per constraint 4.

### Group `critic` (centralized, privileged)

Everything in `policy` for all agents, plus all agents' true 15-dim parameters, plus payload mass,
payload inertia, cable lengths, `active_mask`, and the realized loaded team TWR (`margin`).

### Group `team_probe` (analysis only, never fed to any network)

Per-cable tension, per-agent thrust fraction, payload wrench residual. This is the measurement
channel for H1/H2 — it must exist from day one or the hypotheses cannot be evaluated later.

---

## 6. Rewards

**New module:** `src/payload_drones/mdp/rewards.py`

Reuse `_exp_kernel` and the Gaussian-kernel convention from the base repo's `rewards.py`.
Positive weights sum to 1.0; penalties are negative on top.

| Term | Weight | Definition |
| --- | ---: | --- |
| `payload_position_tracking` | 0.30 | `exp_kernel(||p_load - p_ref||^2, std=0.5)` |
| `payload_position_fine` | 0.20 | same, `std=0.1` |
| `payload_orientation_tracking` | 0.20 | `exp_kernel(quat_error^2, std=0.3)` |
| `payload_velocity_tracking` | 0.15 | `exp_kernel(||v_load - v_ref||^2, std=0.5)` |
| `payload_ang_vel_penalty` | 0.15 | `exp_kernel(||w_load||^2, std=1.0)` — settle the load |
| `action_rate` | -0.01 | per-agent, masked, mean over active agents |
| `body_rate_penalty` | -0.01 | per-agent quad angular velocity |
| `quad_proximity_penalty` | -0.05 | pairwise; hinge below `min_separation=0.5` m |
| `cable_slack_penalty` | -0.02 | discourages sustained slack (snap loads) |

Per-agent penalties must be **masked and averaged over active agents only**, so a failed agent does
not dilute the penalty for the survivors.

**Explicitly forbidden:** any term rewarding equal tension, balanced thrust, or a target force
allocation. See constraint 1.

---

## 7. Terminations

**New module:** `src/payload_drones/mdp/terminations.py`

Reuse `nan_detection` and `time_out` from mjlab. New terms:

- `payload_ground_contact` — payload z below 0.15 m relative to env origin.
- `payload_far_from_goal` — `||p_load - p_ref|| > 3.0` m.
- `any_quad_tumbling` — any *active* quad with `||w|| > 50` rad/s. Ignore inactive agents.
- `quad_collision` — any pair closer than 0.25 m.
- `cable_overextension` — any cable stretched beyond `1.5 * L` (indicates a solver problem).

---

## 8. Env config and registration

**New module:** `src/payload_drones/payload_env_cfg.py`, mirroring the base repo's
`traj_env_cfg.py` structure (`ObservationsCfg`, `RewardsCfg`, `EventCfg`, `TerminationsCfg`,
`CommandsCfg`, `CurriculumCfg` dataclasses composed into a `ManagerBasedRlEnvCfg`).

- `episode_length_s = 10.0`, `decimation = 2` (control at 100 Hz on a 5 ms physics step).
- `SimulationCfg(njmax=128, nconmax=64, integrator="euler", timestep=0.005)`.
- Registration uses **mjlab's own registry**, not Gymnasium:
  `mjlab.tasks.registry.register_mjlab_task` called from `payload_drones/config/__init__.py`,
  exposed via `[project.entry-points."mjlab.tasks"]` in `pyproject.toml`, launched as
  `uv run train <task-id>` — mirroring `mj_adapt_drones/config/__init__.py`. mjlab does not use
  Gymnasium and it is not installed.

  > Corrected 2026-07-22. This section originally said "gym registration pattern", which the base
  > repo does not use. `register_mjlab_task` additionally requires an `rl_cfg`, so registration is
  > deferred to the model/training stage; until then build the env directly from
  > `make_payload_env_cfg`.

Event ordering matters and must be: `reset_scene_to_default` → `team_dynamics_randomization` →
`sample_agent_failure` → root-state reset. The base repo's DR docstring states this ordering
requirement; violating it randomizes into a stale model.

---

## 9. Acceptance tests

`tests/test_env_smoke.py` — all must pass before any RL work begins.

1. Env constructs, `reset()` and 100 `step()` calls with random actions produce no NaN.
2. Observation group shapes match this document exactly; assert on the numbers.
3. **Heterogeneity check:** over 512 envs, the *population mean* of within-team `arm_length`
   standard deviation is > 0.02 m. This is the direct test of constraint 3 — if agents are
   accidentally correlated it fails here rather than silently producing a null result three weeks
   later.

   > Corrected 2026-07-22. This originally demanded that *every* team exceed 0.02 m, which is
   > statistically impossible: correct IID uniform draws occasionally land close together, and the
   > measured 512-team minimum was 0.00128 m. Asserting the per-team bound would have forced
   > correlated or resampled team draws — destroying the very property the test exists to protect.
   > Measured population mean: **0.02963 m**.
4. **Privilege leak check:** assert the `policy` group tensor is unchanged when payload mass and
   cable length are perturbed, holding all else fixed. Guards constraints 2 and 5.
5. Masking: with an agent inactive, its commanded thrust reaches the actuators as exactly zero.
6. Reward sanity: payload held at goal yields positive reward summing to exactly 1.0; at 3 m
   position error the two *position* kernels fall below 1e-12.

   > Corrected 2026-07-22. This originally demanded near-zero *total* reward at 3 m error, which is
   > mathematically impossible under section 6's weights: orientation, linear-velocity and
   > angular-velocity terms carry 0.50 of the positive weight and remain satisfied at a
   > position-only error, so the total is necessarily 0.5000. The assertion now targets the
   > position components, which are what a position error is supposed to move.
7. GPU: 4096 envs construct and step without OOM.
