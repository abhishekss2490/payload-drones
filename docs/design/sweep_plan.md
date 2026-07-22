# Phase-1 bring-up sweep plan

All four jobs must run from one commit and seed `4551`.  The environment,
termination economics, contact model, attitude setpoint, reward-tightening
schedule, and reset distribution are identical.  Only the flags shown below
differ.

The common final reward stage begins at iteration 1000, so iterations
1000--1500 are an unchanged final-policy hold.

## Shared launch flags

```bash
uv run --no-sync train Mjlab-PayloadTransport-RMA-Phase1 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 1500 \
  --agent.algorithm.num-mini-batches 16 \
  --agent.seed 4551
```

The cluster wrapper may add logging, GPU, and SLURM flags, but it must not
mutate environment or PPO settings outside the registered flags below.

## 1. Corrected baseline (`baseline`)

```bash
--agent.run-name baseline \
--env.rewards.payload-position-tracking.params.std 0.75 \
--env.commands.payload-pose.hover-fraction-initial 0.40 \
--env.commands.payload-pose.hover-fraction-final 0.15 \
--agent.algorithm.learning-rate 0.001 \
--agent.algorithm.num-learning-epochs 5 \
--agent.algorithm.schedule adaptive \
--agent.algorithm.desired-kl 0.015
```

Hypothesis: aligned reset, non-suicidal termination economics, hover exposure,
and conservative PPO are sufficient for stable suspension followed by moving
tracking.

Falsification: on a fixed deterministic evaluation set, hover survival does
not improve by iteration 300, or moving-reference position RMSE and physical
contact rate remain flat through iteration 750.  A renewed pattern of rising
collision fraction together with falling episode length is an immediate stop,
not merely a failed hypothesis.

## 2. Broad early capture (`broad_capture`)

```bash
--agent.run-name broad_capture \
--env.rewards.payload-position-tracking.params.std 1.50 \
--env.commands.payload-pose.hover-fraction-initial 0.40 \
--env.commands.payload-pose.hover-fraction-final 0.15 \
--agent.algorithm.learning-rate 0.001 \
--agent.algorithm.num-learning-epochs 5 \
--agent.algorithm.schedule adaptive \
--agent.algorithm.desired-kl 0.015
```

This differs from the baseline only during iterations 0--500.  At iteration
500 the shared curriculum sets the coarse std to 0.60, and at iteration 1000
to 0.50.

Hypothesis: even with aligned spawn, early failures leave the 0.75 m basin
often enough that a 1.5 m recovery kernel improves acquisition.

Falsification: no improvement over baseline in deterministic survival or RMSE
by iterations 100--500, or worse final RMSE after both runs share the same
iteration-1000 reward.  Training reward is not comparable across these two
runs and must not be used as the decision metric.

## 3. Aggressive PPO (`aggressive_ppo`)

```bash
--agent.run-name aggressive_ppo \
--env.rewards.payload-position-tracking.params.std 0.75 \
--env.commands.payload-pose.hover-fraction-initial 0.40 \
--env.commands.payload-pose.hover-fraction-final 0.15 \
--agent.algorithm.learning-rate 0.005 \
--agent.algorithm.num-learning-epochs 10 \
--agent.algorithm.schedule fixed \
--agent.algorithm.desired-kl 0.02
```

Hypothesis: the inherited single-quad PPO update is too aggressive for the
three-agent joint distribution.  With a fixed schedule, `desired_kl` is
intentionally inert; the flag records the inherited value exactly.

Falsification: aggressive PPO matches or beats baseline on deterministic
survival, position RMSE, collision fraction per completed episode, action
saturation, curriculum progress, and finite value/surrogate losses.  Because
the installed runner does not emit KL or clip fraction, do not make a decision
from unavailable diagnostics; for the adaptive baseline, the learning-rate
trajectory records whether KL feedback repeatedly drove it to the floor.

## 4. No hover warm-up (`no_hover`)

```bash
--agent.run-name no_hover \
--env.rewards.payload-position-tracking.params.std 0.75 \
--env.commands.payload-pose.hover-fraction-initial 0.0 \
--env.commands.payload-pose.hover-fraction-final 0.0 \
--agent.algorithm.learning-rate 0.001 \
--agent.algorithm.num-learning-epochs 5 \
--agent.algorithm.schedule adaptive \
--agent.algorithm.desired-kl 0.015
```

Hypothesis: asking for moving `rho=0.05` trajectories from the first episode is
too difficult before cable-coupled hover is learned.

Falsification: this run matches baseline on deterministic survival, position
RMSE, contact rate, and trajectory-curriculum promotion.  If it does, the
hover mixture adds no useful curriculum.

## Review measurements

Evaluate checkpoints 0, 100, 250, 500, 1000, and 1500 deterministically on the
same fixed team/trajectory set.  Review:

- full-episode survival and episode length;
- every termination as a fraction of completed episodes, not a raw count;
- position RMSE and 95th percentile, velocity error, and attitude error;
- minimum inter-quad separation and actual quad--quad contacts;
- action mean, learned standard deviation, and saturation;
- value/surrogate losses and the adaptive learning-rate trajectory (the
  installed runner does not log joint KL, clip fraction, or explained variance);
- sampled `rho`, hover-episode fraction, and per-bin curriculum ceilings.

The submission gate is common to all runs: if collision fraction rises while
episode length falls in the short sanity run, do not submit the sweep.

## Preflight result (2026-07-22)

A 35-iteration, 64-environment CPU sanity run completed 573,440 environment
steps after increasing `njmax` to 512.  Comparing the first and last five
logged values:

| Metric | First-five mean | Last-five mean |
| --- | ---: | ---: |
| episode length (steps) | 72.148 | 81.496 |
| payload position error (m) | 0.685 | 0.738 |
| quad--quad contact termination | 0.154 | 0.074 |
| payload-ground contact termination | 0.349 | 0.318 |
| quad-ground contact termination | 0.382 | 0.488 |
| tumbling termination | 0.645 | 0.582 |

Termination metrics can overlap on a terminal step and therefore do not sum
to one.  `payload_far_from_goal` remained about 0.001, while timeouts,
cable-overextension, and NaN termination were all zero.  The submission gate
passes: collision termination fell while episode length rose.  Position error
did not improve in this deliberately under-powered preflight, so it is not
evidence that the tracking task is learned.
