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
saturation, and curriculum progress without excessive joint KL or clip
fraction.

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
- joint KL, PPO clip fraction, and critic explained variance;
- sampled `rho`, hover-episode fraction, and per-bin curriculum ceilings.

The submission gate is common to all runs: if collision fraction rises while
episode length falls in the short sanity run, do not submit the sweep.
