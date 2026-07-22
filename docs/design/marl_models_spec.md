# MARL model and training specification

Implementation spec for the models, algorithms and training configs.
Architecture rationale lives in `docs/design/marl_wiring.md` — read it first; this document does not
repeat it. Environment contract is `docs/design/env_spec.md`. Hypotheses are in `RESEARCH_BRIEF.md`.

**Base repo:** `/scratch/robotics/mj-adapt-drones`, importable as `mj_adapt_drones`. Read-only.
Reuse `rl/modules.py` (`CausalTCN`, `TrajectoryEncoder`, `CausalConv1d`), `rl/models.py`
(`RMATeacherModel`, `RMAStudentModel`, `_squashed_forward`, `_rebuild_mlp_with_output_activation`),
`rl/algorithms.py` (`LatentDistillation`), `rl/runner.py`, `rl/configs.py` and
`config/rl_config.py` as the templates. Import the modules; subclass or adapt the models.

---

## 0. Non-negotiable constraints

Carried from `env_spec.md` section 0. Each maps to a hypothesis that dies if violated.

1. **The adaptation module is ego-only, in both input and target.** Agent *i*'s TCN sees only agent
   *i*'s own state-action history, and is regressed only onto agent *i*'s own dynamics latent.
   H3 — that this latent nevertheless encodes team state — is meaningless otherwise. Folding the
   agent axis into the batch enforces this structurally: **each row must contain exactly one agent's
   own data, and no row may reach another agent's.** Assert it, do not assume it.
2. **No one-hot agent identity anywhere in the actor path.** H4 is the claim that an inferred
   capability latent replaces CoRL's assigned index. An `agent_id_onehot` may exist only as an
   ablation flag, default off.
3. **The actor never sees privileged state.** No payload mass, no cable length, no teammate
   dynamics. Only the critic does.
4. **Parameter sharing across agents.** One actor, applied per agent. Not three actors.

---

## 1. Module layout

```
src/payload_drones/rl/
    models.py       MARLTeacherModel, MARLStudentModel
    algorithms.py   TeamLatentDistillation
    runner.py       MARLPhase1Runner, MARLDistillationRunner
    configs.py      model / algorithm / runner cfg dataclasses
src/payload_drones/config/
    __init__.py     register_mjlab_task calls
    rl_config.py    payload_rma_phase1_cfg(), payload_rma_phase2_cfg()
```

## 2. Agent-axis folding

`N_AGENTS = 3`. Groups carrying an agent axis: `policy` (3x38), `agent_dynamics` (3x15),
`agent_history` (3 x 50 x 42). Groups without: `payload_trajectory` (1200), `critic` (170).

Fold at the top of `get_latent`, unfold at the end of `forward`:

```
(B, N*d) -> view(B, N, d) -> reshape(B*N, d)      # fold
(B*N, 4) -> view(B, N, 4) -> reshape(B, N*4)      # unfold
```

**Ordering must be consistent everywhere.** Agent index is the *second* axis before folding, so row
`b*N + i` is env `b`, agent `i`. Any place that folds or unfolds uses exactly this convention —
mixing `(B, N, d)` and `(N, B, d)` silently trains a policy on scrambled agents and still converges
to something plausible-looking. Test 3 exists to catch this.

## 3. Trajectory encoder is shared, encoded once

`payload_trajectory` has **no agent axis** — all three agents track the same payload reference.

```
traj (B, 1200) -> TrajectoryEncoder -> z_traj (B, 16)
                                    -> expand to (B, N, 16) -> reshape (B*N, 16)
```

Encode **once per env, then broadcast**. Do not re-encode per agent: it is 3x the compute for
identical outputs. Use `expand`/`unsqueeze` rather than `repeat` where possible to avoid the copy.

Reuse `TrajectoryEncoder` unmodified with its defaults (`num_waypoints=200`,
`features_per_waypoint=6`, `output_dim=16`).

## 4. Models

### `MARLTeacherModel` (Phase 1)

Per-agent latent, mirroring `RMATeacherModel`:

```
[own_policy_obs (38), z_traj (16, shared), z_dyn (8)] = 62 -> MLP -> 4 motor commands
```

- `dynamics_encoder`: MLP over agent *i*'s own 15-dim params -> `z_dyn` (8). Same shape and role as
  the base repo's.
- **`output_activation=None`**, with `_squashed_forward` applying `tanh` so the policy mean lands in
  `[-1, 1]`. The action term then maps `[-1,1] -> u in [0,1]` via `scale=0.5, offset=0.5`, clamps,
  and applies `F = max_thrust * u^2`. This is the base repo's contract, unchanged.

  > Corrected 2026-07-22. An earlier revision said "keep the base repo's `output_activation`
  > (sigmoid to `[0,1]`)". That is wrong twice over: the base repo's config default is
  > `output_activation=None`, and `_squashed_forward` applies `tanh` unconditionally when a
  > distribution is present. Sigmoid *and* tanh would yield `[0, tanh(1)) ~ [0, 0.762)` — neither the
  > `[-1,1]` the action term expects nor the `[0,1]` motor range, and it would make thrust below
  > hover unreachable.
- `obs_normalizer` sized for the 62-dim per-agent latent, accumulating statistics over `B*N` rows —
  shared across agents, which is correct under parameter sharing.
- `get_dynamics_latent(obs)` returns `(B*N, 8)` for use as the Phase 2 distillation target.

### `MARLStudentModel` (Phase 2)

Identical, with `z_dyn` replaced by `z_dyn_hat` from a `CausalTCN` over agent *i*'s own history:

```
history (B, N*50*42) -> fold -> (B*N, 50, 42) -> CausalTCN -> z_dyn_hat (B*N, 8)
```

Only the TCN trains. The trajectory encoder and actor MLP load from Phase 1 and freeze — mirror the
weight-loading surgery in the base repo's `MjlabDistillationRunner.load`.

### Critic

Centralized, **not folded**: `critic` obs `(B, 170)` -> MLP -> `(B, 1)`. One value per env, since
the team reward is shared. Do not replicate the value per agent.

## 5. Action distribution and masking

The 12-dim action is three independent 4-dim per-agent distributions under a shared actor, so the
joint log-prob is the sum of per-agent log-probs — exactly cooperative parameter-shared MAPPO,
obtained from stock PPO.

**Tie the standard deviation across agents.** In rsl_rl, `std_type="scalar"` means *direct* (as
opposed to log) parameterization, not one shared scalar:
`self.std_param = nn.Parameter(init_std * torch.ones(output_dim))` gives `output_dim` independent
values — 12 of them here. Implement a minimal `GaussianDistribution` subclass holding a 4-dim
parameter tiled to 12, so all three agents share one exploration std.

> Corrected 2026-07-22. `marl_wiring.md` claimed the base repo's scalar-std distribution already
> shares std across agents. It does not. Untied stds would probably converge anyway, since agent
> slots are exchangeable (all three draw arm length from the same distribution), so this is a
> correctness/cleanliness fix rather than a results-threatening one. But "parameter sharing" should
> be exactly true, including the exploration std, not approximately true.

**Agent masking is NOT applied in the PPO loss.** Training runs with all agents active
(`p_failure=0.0`); agent failure is a *zero-shot evaluation* condition for H2, not a training
condition. Failure is already handled where it belongs — the action term forces an inactive agent's
thrust to exactly zero.

> Corrected 2026-07-22. An earlier revision required masking inactive agents' log-prob and entropy
> in the PPO loss. Stock `rsl_rl.algorithms.PPO` computes one already-summed joint log-prob and
> entropy and has no mask field in the transition or loss, so this would have required a custom
> distribution, loss, and rollout storage — for a code path that never executes under the planned
> experiments.

**The actor stays blind to `active_mask`, deliberately.** It is critic-only. For H2 the scientific
question is whether surviving agents *infer* a teammate's failure from payload dynamics through the
cable — which is the mechanism H3 proposes. Telling them directly would answer a much less
interesting question. If a later experiment trains *with* failures, it will need a custom masked
loss; note that then rather than building it speculatively now.

## 6. Two-phase training

Follow `config/rl_config.py` and `rl/runner.py`.

- **Phase 1** — PPO with the teacher. Privileged `z_dyn` from true parameters.
  `MARLPhase1Runner` mirroring `MjlabRMAPhase1Runner`.
- **Phase 2** — `TeamLatentDistillation` subclassing `LatentDistillation`. Student TCN regresses onto
  the frozen teacher's `z_dyn`. Loss is MSE over `(B*N, 8)` — **ego-paired**: row `b*N+i` of the
  prediction must be matched against row `b*N+i` of the target, never a permutation.

Register both via `register_mjlab_task` in `payload_drones/config/__init__.py`, exposed through
`[project.entry-points."mjlab.tasks"]` in `pyproject.toml`, as
`Mjlab-PayloadTransport-RMA-Phase1` / `-Phase2`. Mirrors `mj_adapt_drones/config/__init__.py`.

Operating point: **8192 envs** (measured 83,330 env-steps/s, 10.11 of 15.6 GiB).

## 7. Acceptance tests

`tests/test_marl_models.py`. Report real measured output.

1. **Shapes.** Teacher and student produce `(B, 12)` from a real env observation; critic produces
   `(B, 1)`.
2. **Trajectory encoder runs once.** Patch/count `TrajectoryEncoder.forward` calls during one
   `get_latent`; assert exactly one call, and that the resulting `z_traj` rows are identical across
   the three agents of the same env.
3. **Agent ordering round-trips.** Fold then unfold a tensor of known per-agent values and assert
   exact recovery. Then confirm that permuting agent *j*'s observation block changes only agent *j*'s
   action block, and leaves the other two bitwise identical. This is the test that catches a
   transposed fold.
4. **Ego-only isolation (H3 depends on this).** Perturb agent *j*'s `agent_history` and
   `agent_dynamics`; assert agents *k != j* produce bitwise-identical actions and identical
   `z_dyn_hat`. If any other agent moves, the adaptation module is not ego-only and H3 is void.
5. **No privileged leak.** Perturb payload mass, cable length and teammate dynamics in the critic
   group; assert the actor's output is unchanged.
6. **Distillation pairing.** With a known teacher and a student whose TCN is identity-like, assert
   the distillation loss is zero only for the correct ego pairing and non-zero under an agent
   permutation. Guards against a silently transposed target.
7. **Shared exploration std.** Assert the actor's std tensor has 12 entries that are equal in
   groups of 4 across agents (i.e. backed by a single 4-dim parameter), and that one optimizer step
   keeps them tied.
8. **Phase 2 freezing.** After loading Phase 1 weights, assert `requires_grad` is False on the
   trajectory encoder and actor MLP, True on the TCN, and that one optimizer step changes only TCN
   parameters.
9. **GPU.** Both models forward and backward at 8192 envs without OOM; report peak memory and
   the throughput cost versus the env-only baseline of 83,330 env-steps/s.
