# Design plan: moderate team heterogeneity and payload randomization

**Status:** plan for review; no implementation is authorized by this document.

This plan changes four coupled parts of the environment:

1. arm lengths become correlated within a team while retaining a controllable
   within-team spread;
2. payload mass is sampled against a minimum **loaded** team thrust-to-weight
   ratio (TWR), if that is the intended meaning of the requested 2.5 floor;
3. payload attachment geometry, center of mass (COM), and inertia become
   randomizable without making reset geometry unsafe; and
4. a failed agent and a healthy agent with a slack cable are represented and
   measured differently.

The primary scientific recommendation is to keep attachment geometry symmetric
and COM centered in the main H1 experiment. Attachment/COM asymmetry should be
a controlled second axis or ablation. Otherwise geometric statics prescribe
unequal tension before capability-driven behavior has a chance to emerge.

## Recommended defaults at a glance

| Item | Recommended default | Sweep/ablation values |
| --- | --- | --- |
| Arm sampling | correlated; base `U(0.060, 0.145)` m; per-agent multiplicative spread `±0.10` | spread `0`, `0.10`, `0.30`, and IID-independent |
| Clean arm-spread sweep | use common base range `U(0.072, 0.123)` m for both `±0.10` and `±0.30` | separate broad-scale `±0.10` run uses `U(0.060, 0.145)` |
| Per-agent TWR | **provisionally** change from `(2.0, 4.0)` to `(3.0, 4.0)` if 2.5 is a hard loaded-TWR floor | preserve `(2.0, 4.0)` by conditioning on unloaded team TWR `>= 2.8` |
| Loaded team TWR | sample `U(2.5, min(3.0, 0.9 * unloaded_team_twr))` | floor sweep `2.0`, `2.25`, `2.5` |
| Payload mass cap | `2.5 kg`, upper cap only | report cap rate |
| Main-H1 attachment geometry | fixed equilateral triangle, radius `0.300 m`; centered COM | asymmetric-geometry and off-center-COM ablations |
| Payload inertia | independently vary mass-scaled nominal inertia by approximately `0.72–1.38×`, then project to a valid inertia | off for nominal-physics control |
| COM | `(0,0,0)` in main H1 | horizontal offset radius `0–0.030 m`, vertical `±0.010 m` |
| Slack handling | no slack termination; no slack reward in primary H1; diagnose sustained slack | optional delayed soft penalty after `0.25 s` |
| Failure handling | zero thrust via `active_mask`, body and cable remain in physics, actor never sees the mask | failure time and failed index are analysis-only |

The TWR recommendation is provisional because “TWR stays above 2.5” is
ambiguous. The existing payload sampler and `TeamState.margin` both define TWR
as **loaded** team TWR, so that is the interpretation used below. If 2.5 was
intended as an unloaded-team floor, section 2 should not be adopted as written.

## Quantitative-method note

The arm/mass and payload numbers below come from a 1,000,000-team Monte Carlo
calculation using the polynomial mean and uniform `±std` mass fits in
`quad_scaling.py`, with random seed `20260723`. The calculation omits the later,
small pseudo-inertia `U` perturbation, so these are design estimates rather than
acceptance-test constants. The implementation must remeasure them through the
actual reset event before ranges are locked.

## 1. Correlated arm-length sampling

### Mechanism

Add team-level arm sampling ahead of the three calls to
`quadrotor_dynamics_randomization`:

```text
B ~ Uniform(base_arm_range)
epsilon_i ~ Uniform(-arm_relative_spread, +arm_relative_spread)
L_i = B * (1 + epsilon_i),  i in {0,1,2}
```

The quad randomizer should accept a pre-sampled `arm_length` override. It should
continue to draw each quad's mass residual, inertia residuals, inertial-frame
perturbation, motor geometry noise, `km/kf`, motor time constant, and TWR
independently. Correlating arm length must not accidentally make the complete
quad dynamics identical.

Recommended configuration fields:

```text
arm_sampling_mode: "correlated" | "independent" = "correlated"
arm_hard_range: (0.050, 0.160) m
base_arm_range: (0.060, 0.145) m
arm_relative_spread: 0.10
arm_offset_distribution: "uniform"
min_team_arm_range_fraction: 0.0
```

`min_team_arm_range_fraction` should remain zero for training: rejecting teams
whose three IID offsets happen to be close would distort the declared
distribution. H1 evaluation should instead stratify or bin teams by realized
capability contrast.

At `±10%`, `B in (0.060, 0.145)` produces arm lengths in
`(0.054, 0.1595)` m without clipping. Do not silently clamp arm lengths; clipping
creates point masses at the endpoints and changes the within-team spread as a
function of base size. Validate that the configured base/spread product lies
inside `arm_hard_range` and fail configuration early if it does not.

For a clean `±10%` versus `±30%` spread sweep, use the same base range
`(0.072, 0.123)` m for both: it keeps even `±30%` inside approximately
`(0.0504, 0.1599)` m. Using the broader default base range only for `±10%` is
useful for training coverage, but it cannot be compared directly with a
`±30%` run without controlling for the changed team-scale distribution.

The independent control remains exactly the current
`L_i ~ U(0.05, 0.16)` sampling. Treat it as an extreme-heterogeneity/OOD control,
not the primary training distribution.

### Does `±10%` retain meaningful heterogeneity?

Yes, but the claim changes from “orders-of-magnitude teammates” to moderate,
realistic capability variation. It does not become a homogeneous-team task.

| Within-team quantity | Current IID `0.05–0.16 m` | Correlated `±10%`, base `0.06–0.145 m` |
| --- | ---: | ---: |
| Median arm max/min | `1.733×` | `1.105×` |
| 5–95% arm max/min | `1.149–2.707×` | `1.027–1.189×` |
| Mean within-team sample std | `29.0 mm` | `5.4 mm` |
| Median mass max/min | `3.83×` | `1.57×` |
| 5–95% mass max/min | `1.43–13.37×` | `1.13–2.42×` |
| Median largest headroom share | `54.4%` | `44.1%` |
| Median smallest headroom share | `13.6%` | `22.9%` |
| 5–95% largest headroom share | `39.4–78.6%` | `36.2–57.5%` |

Headroom here is the registered H1 quantity
`(TWR_i - 1) * m_i * g`, with the current independent per-agent
`TWR_i ~ U(2,4)`. The mass polynomial is steep enough that roughly 10% arm
variation gives a median `1.57×` within-team mass ratio. Independent TWR and
the remaining per-quad DR add further capability variation.

One scientific caveat is important: with per-agent TWR still independently
sampled over `(2,4)`, much of H1's capability contrast can come from TWR rather
than arm size. Correlated arms do not imply correlated TWR in the current code.
Report H1 both against total headroom and with mass/arm and TWR contributions
separated. This is also why changing TWR to `(3,4)` in section 2 does not erase
H1: the median arm-driven mass ratio remains non-trivial.

### H1/H2/H3 implications

- **H1:** the expected signal is smaller but more physically credible. Use
  realized continuous headroom contrast, not a binary “heterogeneous” label.
  Pre-register H1 effect size by arm-spread bin.
- **H2:** similar vehicles make the identity of the failed agent less
  determinative, which is desirable for a clean redistribution test. Still
  report the failed agent's pre-failure thrust-capacity share.
- **H3:** teammate arm lengths have less within-team variance, so raw arm-length
  decoding R² is a harder test. Report normalized error and a shuffled-label
  control; do not interpret lower R² under `±10%` as failure without accounting
  for the reduced target variance. Payload mass and team-TWR decoding remain
  meaningful.

### What could break and what to verify

The current quad randomizer samples arm length internally, so merely calling it
three times cannot create team correlation. A small API refactor or an explicit
override is required. This is easier than changing the physics, but it affects
the single-quad test path and partial resets.

Verify:

1. every `L_i` lies in the configured relative interval around the stored base;
2. no clipping or endpoint pile-up occurs;
3. partial resets change only requested environments;
4. per-agent residual dynamics remain independent conditional on arm length;
5. for the broad `±10%` default, measured mean within-team arm std is near
   `5.4 mm`, replacing the old acceptance threshold `>20 mm`, which would now
   reject the intended design;
6. pairwise arm-length correlation is high (design estimate about `0.94`) for
   correlated sampling and near zero for the independent control; and
7. the H1 analysis records base scale, realized arm spread, mass spread, TWR
   spread, and headroom spread separately.

## 2. Payload mass under a team-TWR floor

### First resolve what “2.5 floor” means

Use:

```text
U = sum_i(max_thrust_i) / (g * sum_i(m_i))                  # unloaded team TWR
R = sum_i(max_thrust_i) / (g * (sum_i(m_i) + m_payload))    # loaded team TWR
```

This plan assumes the requested floor is `R >= 2.5`. That is consistent with
the existing sampler, `TeamState.margin`, and the curriculum variable. If the
request instead means `U >= 2.5`, payload mass could still be sampled down to a
loaded TWR of 1.5 under the old formula, so the task difficulty would barely
change.

### Mathematical incompatibility in the current ranges

A positive payload always makes `R < U`. Therefore a hard loaded floor of 2.5
is impossible for any sampled team with `U <= 2.5`, and it produces a nearly
zero payload for `U` only slightly above 2.5.

This is not a corner case. Arm correlation does not solve it because
`quad_scaling.py` samples each quad's TWR independently over `(2,4)`.
In the Monte Carlo design sample, about `29.1%` of correlated-`±10%` teams had
`U < 2.8`. Thus the formula cannot simply replace `1.5` with `2.5`; its
per-team upper bound would be below its lower bound for a material fraction of
episodes.

### Provisional recommended mechanism

If 2.5 is a hard loaded-TWR floor, provisionally use:

```text
per-agent TWR_i ~ U(3.0, 4.0)
U = sum_i(TWR_i * m_i) / sum_i(m_i)
R_target ~ U(2.5, min(3.0, 0.9 * U))
m_payload_raw = sum_i(TWR_i * m_i) / R_target - sum_i(m_i)
m_payload = min(m_payload_raw, 2.5 kg)
R_realized = sum_i(TWR_i * m_i) / (sum_i(m_i) + m_payload)
```

This guarantees `U >= 3.0`, a non-empty target interval, positive payload, and
`R_realized >= 2.5`. The `0.9 * U` upper bound also guarantees
`m_payload / team_mass >= 1/0.9 - 1 = 0.111` before the upper mass cap. As in
the current correct sampler, there must be no minimum payload-mass clamp.

If preserving the current individual TWR support `(2,4)` is scientifically
important, the alternative is to resample only the three TWR draws until
`U >= 2.8`, leaving arms and masses unchanged, then use the same formula.
The `2.8` threshold supplies a small buffer over `2.5/0.9 = 2.777...`.
This rejects about 29% of initial TWR triples and shifts the realized unloaded
team distribution upward; that conditioning must be disclosed.

Recommended configuration knobs:

```text
loaded_team_twr_floor: 2.5
loaded_team_twr_ceiling: 3.0
payload_positive_margin_fraction: 0.9
max_payload_mass: 2.5 kg
team_twr_conditioning: "raise_individual_floor" | "resample_team"
minimum_unloaded_team_twr: 2.8       # used only by resample_team
max_twr_resample_attempts: 64
```

Do not fall back to a lower loaded TWR after exhausting resamples. Fail the
reset with diagnostics; a fallback would violate the declared distribution.

### Estimated payload distributions with correlated `±10%` arms

| Quantity | Current margin `1.5...` and TWR `(2,4)` | Floor 2.5, condition `U>=2.8`, TWR support `(2,4)` | Floor 2.5, TWR `(3,4)` |
| --- | ---: | ---: | ---: |
| Payload mass median | `0.240 kg` | `0.110 kg` | `0.163 kg` |
| Payload mass 5–95% | `0.054–0.907 kg` | `0.032–0.320 kg` | `0.048–0.444 kg` |
| Payload/team-mass median | `0.420` | `0.170` | `0.273` |
| Payload/team-mass 5–95% | `0.138–0.992` | `0.115–0.348` | `0.144–0.433` |
| Loaded TWR median | `2.051` | `2.622` | `2.741` |
| Payload under `50 g` | about `4.2%` | `16.1%` | `5.9%` |
| Hit `2.5 kg` cap in design sample | `0%` | `0%` | `0%` |
| Positive and loaded TWR `>=2.5` | not applicable | `100%` | `100%` |

The task does not collapse under the provisional `(3,4)` recommendation: the
median payload is 27% of team mass and the 95th percentile is 43%. The
conditioned `(2,4)` alternative is relatively lighter and produces many more
absolute payloads below 50 g, although its payload is still at least about 11%
of team mass by construction.

These distributions are a direct consequence of requesting high control
headroom. A loaded TWR floor of 2.5 cannot simultaneously produce the old
median payload/team ratio of 0.42 unless unloaded team TWR is raised
substantially.

### H1/H2/H3 implications

- **H1:** compare load sharing only within narrow loaded-TWR and payload-ratio
  bins. Otherwise a high-headroom, light-payload episode is mixed with a
  lower-headroom, heavier one.
- **H2:** a floor of 2.5 makes one-agent-loss trials more likely to remain
  physically hoverable. For each proposed failed agent compute the post-failure
  capacity
  `sum_active(max_thrust) / ((team_mass + payload_mass) * g)`.
  The main H2 set should require this to be at least `1.15`; trials below 1.0
  are impossible-recovery stress tests, not adaptation tests.
- **H3:** loaded TWR, payload mass, and team mass become statistically coupled
  by construction. A payload-mass probe must therefore include controls that
  distinguish genuine payload inference from decoding ego/team scale. Report a
  conditional probe controlling for ego mass/TWR and base arm length.

### What could break and what to verify

Changing `ScaleCfg.twr_range` changes motor authority, action scaling, phase-1
training, and H1 headroom—not only payload mass. Conditioning TWR triples avoids
changing support but changes their joint distribution. Both alternatives need
an explicit experiment label.

Verify on at least 200,000 actual reset samples:

1. `m_payload > 0` for 100% of samples;
2. `R_realized >= 2.5 - 1e-6` for 100%;
3. the target interval is never inverted;
4. payload/team-mass 5th percentile is at least `0.11` before any upper cap;
5. cap rate, sub-50-g rate, rejection rate, and resample-attempt count;
6. the same properties after partial reset; and
7. post-failure capacity distributions for failure of the smallest, median,
   and largest thrust-capacity agent.

## 3. Randomize payload physical properties

### Separate randomization axes

Do not bundle all payload randomization behind one Boolean. Use independent
axes so the core H1 result can keep statics symmetric:

```text
attachment_mode: "symmetric_fixed" | "symmetric_radius" | "asymmetric"
payload_com_mode: "centered" | "offset"
payload_inertia_mode: "mass_scaled" | "independent_shape"
```

The main H1 configuration should be:

```text
attachment_mode = "symmetric_fixed"
attachment_radius = 0.300 m
payload_com_mode = "centered"
payload_inertia_mode = "independent_shape"
```

Independent inertia changes rotational response but does not prescribe a
static cable-tension allocation. It can therefore be present in main H1 after
a nominal-inertia control verifies that it does not hide the effect.

### 3a. Attachment-point locations

For the symmetric-radius robustness axis, draw a single
`R ~ U(0.290, 0.320) m`, a global triangle yaw
`phi ~ U(-pi, pi)`, and retain exact 120-degree spacing and a centered
triangle. This changes moment arms while keeping the static vertical load
allocation at one third each.

For the asymmetric-geometry ablation, propose:

```text
common radius R                 ~ U(0.295, 0.315) m
per-vertex radial multiplier    ~ U(0.94, 1.06)
per-vertex angular jitter       ~ U(-6 deg, +6 deg), then subtract mean jitter
global triangle yaw             ~ U(-pi, pi)
site z                           = +0.060 m (fixed top plane)
```

Reject and redraw a geometry unless all conditions hold:

1. sites retain their counter-clockwise order;
2. every pairwise site distance is at least `0.500 m`;
3. every site center satisfies `|x| <= 0.325 m` and `|y| <= 0.325 m`,
   leaving margin inside the payload's `±0.34 m` top footprint;
4. the payload COM lies strictly inside the attachment triangle; and
5. the vertical-static barycentric share of every cable is at least `0.20`.

The first three constraints directly protect reset geometry. A quick geometric
Monte Carlo for these proposed ranges accepted about 73% of draws before the
COM constraint, so rejection sampling should not be a reset bottleneck.

Write the three payload-local site positions to the batched `site_pos` model
field during `team_dynamics_randomization`, after
`reset_scene_to_default` and before command-owned spawn. Store the positions in
`TeamState`; they are privileged physical parameters, not actor observations.

This part is easier than it first appears because
`settle_quads_for_cable_slack` already refreshes the live site poses and places
each quad hook directly above its corresponding attachment. It does not use
`builder._QUAD_POSITIONS` at reset. The static builder positions remain model
defaults only.

The safety constraints are still essential. At reset, pairwise quad horizontal
separation equals pairwise attachment separation. The `0.500 m` bound avoids
both physical overlap of two maximum `0.16 m`-arm vehicles and an immediate
`quad_proximity_penalty`. Cables are vertical and therefore cannot cross at
reset. Tendons have no collision geometry, so later cable crossing must be
reported as a geometric diagnostic if it matters; MuJoCo will not resolve it
as contact.

### 3b. COM and inertia independent of mass scaling

Start from the current mass-scaled nominal principal inertia:

```text
J_mass = (m_payload / 0.75 kg) * (0.012, 0.012, 0.020) kg m^2
```

For `independent_shape`, draw:

```text
overall inertia scale k0  ~ log-uniform(0.80, 1.25)
axis scales kx,ky,kz      ~ log-uniform(0.90, 1.10), independently
J_raw                     = J_mass * k0 * (kx, ky, kz)
```

The componentwise support is approximately `0.72–1.38` times the simple
mass-scaled nominal. Project the principal moments to the pseudo-inertia cone
as the quad randomizer already does, and store the projected values, not the
raw samples. Optionally rotate the principal frame with yaw
`U(-15 deg,15 deg)` and roll/pitch `U(-5 deg,5 deg)`.

For the COM-offset ablation:

```text
horizontal COM offset: uniform in disk of radius 0.030 m
vertical COM offset:   U(-0.010, +0.010) m
```

Write the result to payload `body_ipos`. MuJoCo's `body_inertia` is about the
COM, so do not apply a parallel-axis shift to that field. Enforce that the COM
remains inside the payload box and, for a three-taut-cable static check, inside
the attachment triangle.

For symmetric radius `0.30 m`, a 3 cm horizontal COM shift can change the
ideal vertical static allocation from `33/33/33%` to approximately
`40/30/30%` when shifted toward one vertex. This is large enough to be a useful
geometric-load axis but moderate enough that all cables retain positive static
tension.

The geometric reference share should be computed and stored for analysis. For
attachment coordinates `r_i` and COM projection `c`, solve:

```text
sum_i(w_i) = 1
sum_i(w_i * r_i_xy) = c_xy
```

The resulting barycentric weights are the vertical static tension shares due
to geometry alone. They must never enter the reward or actor observation.

### 3c. Other physically meaningful properties

Cable length is already randomized independently over `0.5–0.8 m`, and that is
a major payload-system property. Independent cable lengths alter pendulum
dynamics and can complicate attribution in H1 even when static vertical
tensions remain balanced.

Expose:

```text
cable_sampling_mode: "team_correlated" | "independent"
cable_base_range: (0.55, 0.75) m
cable_relative_spread: 0.02
```

Use team-correlated cable lengths in the clean H1 experiment, and retain the
current independent `(0.5,0.8) m` distribution as a robustness ablation.
Do not randomize tendon stiffness/damping; the tendons are unilateral solver
limits, not springs.

Payload geom size and aerodynamic coefficients should not be added in the
first implementation pass. Changing geom size affects collision, drag, site
footprint, and inertia simultaneously and is much harder to identify cleanly.

### H1/H2/H3 implications

- **H1:** asymmetric attachment or off-center COM is a confound if mixed into
  the primary experiment. It creates unequal load sharing even for identical
  agents. Keep geometry symmetric and COM centered for the main H1 claim.
  Then run a factorial ablation with arm spread and geometric static-share
  spread as separate axes.
- In the asymmetric ablation, regress observed tension share on both headroom
  share and stored geometric reference share. Also analyze the residual
  `observed_share - geometric_share`. Capability-proportional behavior remains
  interesting only if headroom explains variation beyond geometry.
- **H2:** use centered COM/symmetric attachments for the main failure study.
  With only two powered agents after failure, an off-center load can make the
  recovery geometrically impossible or require large payload tilt.
- **H3:** COM, inertia, and attachment geometry remain absent from the actor and
  ego teacher target. Exploratory probes may test whether `z_dyn` decodes them,
  but they should not replace the registered payload-mass/team-composition H3
  metrics.

### What could break and what to verify

Attachment randomization itself is moderate difficulty because `site_pos` is
already batched and spawn uses live sites. Physically valid COM/inertia
randomization is harder: it requires payload `body_ipos`/`body_iquat` state,
valid-inertia projection, critic-shape updates, and direct model-versus-state
tests.

Verify:

1. model `site_pos`, `body_ipos`, `body_inertia`, and `body_iquat` equal stored
   `TeamState` values after full and partial resets;
2. all inertia values are positive and satisfy rigid-body triangle
   inequalities after projection;
3. all attachment geometry constraints and barycentric-share bounds;
4. after command-owned spawn, no payload-ground, quad-ground, or inter-quad
   contact in at least 4096 resets;
5. each cable geometric length is `0.95 ± 0.02` of its own sampled limit at
   reset and initial tendon force is zero;
6. minimum root height remains above `0.25 m`;
7. 400-step random-action stress remains finite with the existing constraint
   capacity;
8. actor observations are unchanged when only stored privileged payload
   parameters are perturbed, and contain no new mass/COM/inertia/site metadata;
9. the centralized critic is deliberately extended with COM, principal-axis
   orientation, and attachment positions or a compact equivalent; and
10. symmetric-geometry static hangs recover approximately equal tension,
    while COM-offset static hangs agree with the barycentric prediction within
    a predeclared tolerance (start with `±5 percentage points` per cable).

## 4. Failed agents versus healthy slack cables

### Keep two distinct state concepts

`active_mask` means actuator health/participation commanded by the experiment.
It must not be inferred from instantaneous cable tension.

Add two analysis-only, time-varying concepts:

```text
is_load_bearing_i =
    tension_i > max(0.05 N, 0.02 * payload_mass * g)
is_contributing_i = active_mask_i AND is_load_bearing_i
```

The threshold and duration counters are analysis state, never policy input and
never a replacement for `active_mask`. A healthy slack agent remains active
even when `is_contributing=False`. A failed agent remains non-contributing even
if its dead body temporarily makes `is_load_bearing=True`; that physical
failed-cable tension is reported separately.

Use these timing definitions at 100 Hz:

- transient slack: continuous slack duration `<=0.20 s`;
- sustained slack: duration `>0.50 s`;
- reset the sustained counter only after tension is above threshold for
  `0.10 s`, providing hysteresis against solver chatter.

The actor's existing own-cable direction plus geometric slack flag can remain.
It is an ego physical sensor, not a teammate-failure signal. Do not add
`active_mask`, teammate cable tension/slack, active-agent count, failed-agent
index, or `is_contributing` to actor observations. Surviving agents must infer
failure through their own cable and payload motion/history, preserving H2/H3.

### Case i: genuinely failed agent

- At the scheduled failure time, set only that entry of `active_mask=False`.
  The action term continues to force its actual motor command and processed
  thrust to exactly zero.
- Keep the quad body, mass, collision geometry, hook, and tendon in the
  simulation. It is dead weight, matching the intended H2 experiment.
- Continue computing its actor output and history only to keep static tensor
  shapes; discard its action. Do not zero its observation in a way visible to
  other agents.
- Keep `active_mask` in the centralized critic. This does not leak at
  deployment, and H2 is planned as zero-shot evaluation. If failures are later
  included in training, revisit PPO loss masking explicitly rather than
  assuming the current stock joint log-prob supports it.
- Exclude the failed agent from active-agent action/body-rate/slack shaping
  means. Do not exclude its body from physical team safety: ground contact or
  collision involving the dead quad should still count as mission failure in
  the recommended default.
- Ignore inactive-agent tumbling as the current termination already does.

The current `sample_agent_failure` is a reset-time sampler. H2 requires
mid-flight disablement, so a scheduled mid-episode failure mechanism and a
failure timestamp/index buffer are still needed. This is more work than merely
changing `p_failure`: the statistics wrapper must retain pre-failure state and
must not confuse an auto-reset's next team with the completed episode.

Recommended H2 protocol:

1. hover/track with all agents for `3.0 s`;
2. require a stable pre-failure window of `1.0 s`;
3. fail one agent at `t=3.0 s`;
4. observe at least `3.0 s` post-failure or until mission termination;
5. stratify failure by the agent's pre-failure max-thrust share; and
6. use only trials with post-failure loaded capacity `>=1.15` for the main
   adaptation metric, reporting lower-capacity trials separately.

### Case ii: healthy agent with a slack cable

Transient slack is physically valid for a unilateral cable during payload
swing, unloading, and reorientation. It should not terminate an episode.
Persistent slack also should not directly terminate: if the team can track
with two taut cables for a while, forced three-cable tension would be an
allocation objective in disguise.

For the primary H1 experiment, set the existing slack reward weight to zero.
The present instantaneous `-0.004` penalty explicitly rewards keeping every
healthy cable taut and can make “all agents contribute” partly reward-driven.
If snap loads destabilize training, expose an optional delayed soft penalty:

```text
penalty_i = clamp((slack_duration_i - 0.25 s) / 0.75 s, 0, 1)
team penalty = mean over active agents
weight = -0.004 maximum
```

Run it as a training-stability ablation and disclose it in H1. A better
long-term safety shaper would penalize measured re-tension impulse rather than
slack itself, but that also must remain outside the core emergence result.

Do not dynamically set `active_mask=False` for a healthy slack agent. Doing so
would remove it from penalties, alter reward normalization, and falsely turn
an emergent behavior into an experimenter-declared failure.

### Rewards and terminations

- Payload tracking, orientation, velocity, alive, and crash terms remain shared
  team rewards and do not use `active_mask`.
- `_masked_mean` remains appropriate for local action-rate and body-rate costs:
  denominator is the number of commanded-active agents.
- Healthy-but-slack agents remain in that denominator. Failed agents do not.
- A slack-specific soft cost, when enabled, averages over active agents only.
- `quad_proximity_penalty` may continue to include an inactive body because the
  collision hazard is physical, not an actuator-effort cost.
- No termination is added for slack or zero tension. Existing payload error,
  contacts, cable overextension, and NaN terms determine mission failure.

### `team_probe` and `play_stats`

The current display calls `tension_i / payload_weight` a “load share.” That is a
load ratio, not a normalized tension share; its sum changes with payload
acceleration, cable angle, and quad/cable dynamics. Preserve it as a useful
diagnostic but rename it.

Add three separate readouts:

```text
load_ratio_i = tension_i / (payload_mass * g)
tension_share_i = tension_i / sum_active(tension)
headroom_share_i = headroom_i / sum_active(headroom)
```

Metric rules:

- **H1:** run only failure-free trials. Every healthy agent stays in the
  denominator; a healthy slack cable contributes exactly zero tension. Do not
  drop slack frames or agents, which would bias H1 toward apparent sharing.
- **H2:** exclude inactive agents from the survivors' normalized
  `tension_share`, but report failed-cable tension separately because the dead
  body remains physically coupled.
- Record `active_mask`, `is_load_bearing`, `is_contributing`, continuous slack
  duration, failure index/time, raw tension, and headroom at every analysis
  step.
- Do not average an entire trajectory into one number before distinguishing
  transient, steady pre-failure, and post-failure windows.

For H1, use dedicated hover or low-acceleration evaluation segments. A
recommended steady window discards the first `1.0 s` and requires for at least
`1.0 s`:

```text
payload velocity error < 0.25 m/s
payload angular speed < 0.50 rad/s
sum cable tension > 0.80 * payload_mass * g
```

Report Pearson and Spearman correlation between tension share and headroom
share, with team-clustered confidence intervals. In asymmetric-geometry
ablations also include the geometric barycentric share.

For H2, define redistribution settling time as the first post-failure time at
which every survivor's tension share stays within 10% of its final
`0.5 s`-window mean for a contiguous `0.25 s`. Report payload position and
attitude peak deviation, failed-cable tension decay, and non-recovery rate.
This definition uses analysis-only failure timing; the policy never receives
it.

### What could break and what to verify

The geometric slack observation currently compares geometric length with the
tendon limit and can chatter at equality; use hysteresis for metrics even if
the one-bit actor channel is retained. Tension comes from solver constraint
rows, so validate the force threshold on CPU and GPU paths.

Verify:

1. an inactive agent produces exactly zero actuator force but remains massive,
   collidable, and cabled;
2. changing `active_mask` or the failed index does not directly change any
   surviving actor observation at the same physical state;
3. a failed cable may report tension without being labeled contributing;
4. a healthy slack cable remains active and contributes zero to H1 tension
   share;
5. masked local reward means use two agents after one failure, while shared
   payload reward is unchanged;
6. no slack-only termination occurs for at least `1 s` of deliberately induced
   slack;
7. transient slack under `0.20 s` incurs no delayed penalty when that ablation
   is enabled; and
8. `play_stats` snapshots and prints the episode's actual time-varying failure
   state rather than the next auto-reset state.

## Interactions between the four changes

1. **Correlated arms + loaded-TWR floor:** moderate arm variation reduces
   team-mass extremes, but independent `(2,4)` TWR sampling still creates teams
   below the requested loaded floor. The floor requires either higher
   individual TWR or explicit conditioning.
2. **High TWR floor + H1:** payloads become lighter relative to capacity.
   Tension differences may shrink because all agents have ample authority.
   Analyze H1 across payload/team-mass bins and include floor `2.0` or `2.25`
   as a sensitivity sweep if `2.5` produces a ceiling-effect policy.
3. **High TWR floor + H2:** this is beneficial because two survivors usually
   retain hover authority, but it does not guarantee recovery after loss of the
   strongest agent. Gate the main H2 metric on post-failure feasibility.
4. **Attachment/COM asymmetry + H1:** geometry can produce the same unequal
   tension pattern that H1 attributes to capability. Never mix symmetric and
   asymmetric episodes in the primary H1 correlation without a geometry
   covariate.
5. **Attachment asymmetry + correlated arms:** assigning the largest vehicle
   to the geometrically heaviest vertex by chance can amplify or cancel the
   effect. Randomly permute agent-to-vertex assignment at evaluation and record
   it; do not intentionally match capability to geometry.
6. **COM asymmetry + agent loss:** two surviving cables may not be able to
   support the COM without tilt or help from the failed cable. Keep main H2
   centered and use asymmetry only as a stress test.
7. **Independent cable lengths + geometry/inertia:** all three change pendulum
   modes. Use correlated cable lengths for causal H1 measurements, then restore
   full independent cable DR for robustness.
8. **Slack reward + H1:** even a small persistent-slack cost creates an
   experimenter preference for participation. The clean emergence result needs
   a zero-weight slack term or a clearly separated ablation.
9. **Payload cap + high-scale teams:** if future base/TWR ranges grow, the
   `2.5 kg` cap raises realized loaded TWR above the target. Always analyze
   `R_realized`, not only the sampled target.

## Suggested experiment matrix

The smallest matrix that separates the scientific effects is:

| Experiment | Arm spread | Attach/COM | Loaded floor | Purpose |
| --- | --- | --- | --- | --- |
| H1-main | `±10%` | symmetric/centered | `2.5` | moderate capability-driven sharing |
| H1-homogeneous | `0%` | symmetric/centered | `2.5` | equal-capability null |
| H1-spread | `±30%` | symmetric/centered | `2.5` | dose response |
| H1-extreme | independent `0.05–0.16` | symmetric/centered | `2.5` | old regime/OOD control |
| Geometry ablation | `±10%` | asymmetric or COM offset | `2.5` | separate geometric and capability shares |
| Payload-floor sensitivity | `±10%` | symmetric/centered | `2.0`, `2.25`, `2.5` | detect light-payload ceiling effect |
| H2-main | `±10%` | symmetric/centered | `2.5` | feasible redistribution |
| H2-stress | `±10%` | asymmetric/offset | `2.5` | robustness, not primary mechanism claim |

Use the common safe base-arm range `(0.072,0.123)` for direct spread-dose
comparisons. A separate broad-scale training run may use `(0.060,0.145)` at
`±10%`.

## Risks and open questions for Varad to decide

1. **Does “TWR above 2.5” mean loaded or unloaded team TWR?** This plan assumes
   loaded, matching the current meaning of `margin`.
2. **If loaded, may per-agent TWR change from `(2,4)` to `(3,4)`?** Recommended
   because it guarantees the floor without rejecting almost 29% of initial TWR
   triples and gives a less trivial payload. The alternative is explicit
   `U>=2.8` conditioning.
3. **Should attachment/COM asymmetry be excluded from main H1?** Strong
   recommendation: yes; include it as a controlled second axis with geometric
   barycentric share recorded.
4. **Should the main H1 run remove the current `-0.004` instantaneous slack
   penalty?** Strong recommendation: yes. Keep a delayed soft version only as a
   training-stability ablation.
5. **Which scale coverage matters more for the spread sweep?** The clean
   comparison uses common base range `(0.072,0.123)`; the broad `±10%` default
   `(0.060,0.145)` covers more vehicle scales but cannot support `±30%` without
   clipping or changing support.
6. **Should independent cable lengths remain in main H1?** Recommendation:
   correlate them within `±2%` for causal measurement and retain independent
   `0.5–0.8 m` as robustness DR.
7. **Should contact of the dead failed quad terminate H2?** Recommendation:
   yes, as a mission-safety failure, while inactive-agent tumbling alone
   remains ignored.
8. **Is a 27%-of-team-mass median payload sufficiently non-trivial?** That is
   the design estimate for loaded floor 2.5 with per-agent TWR `(3,4)`. If not,
   the floor must come down or unloaded TWR must rise; the payload formula
   cannot evade that tradeoff.
