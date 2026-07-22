# MARL implementation findings

## What works

- The MARL teacher and student use a strict env-major fold:
  `(B, 3, D) -> (B*3, D)`, and restore actions with the exact inverse.  The
  shared actor therefore operates only on one agent's own row at a time.
- The shared 1200-dimensional payload trajectory is encoded once per
  environment and broadcast with `expand` to the three folded agent rows.
- The centralized critic remains unfurled: `(B, 170) -> (B, 1)`.
- The student folds each agent's own `50 x 42` history before the unchanged
  base `CausalTCN`; its distillation target uses the same env-major ego row.
- Phase-2 checkpoint loading transfers the Phase-1 trajectory encoder, actor,
  normalizer, and distribution then freezes everything except the TCN.
- `SharedAgentGaussianDistribution` has one learnable four-motor std vector;
  the folded actor tiles it over all three agent action blocks, so it remains
  tied after optimization.
- Both RMA phases are registered as `Mjlab-PayloadTransport-RMA-Phase1` and
  `Mjlab-PayloadTransport-RMA-Phase2` through the `mjlab.tasks` entry-point
  hook.

## Corrected requirements

The earlier contradictions were corrected in the specification before this
implementation:

- Actions use the validated base contract: `output_activation=None`, then
  `_squashed_forward` applies `tanh` to produce `[-1, 1]`, which the action
  term maps to motor command `[0, 1]`.
- Exploration std is truly shared across agents through a four-parameter
  distribution rather than rsl-rl's misleadingly named `std_type="scalar"`
  12-parameter implementation.
- `active_mask` remains critic-only and is not used in PPO training; planned
  training has `p_failure=0`, while evaluation failures are handled by the
  action term's exact zero-thrust mask.

## Real verification

Command:

```text
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl uv run --no-sync pytest -q
```

Result on 2026-07-22:

```text
20 passed, 2 skipped, 48 warnings in 12.16s
```

The MARL-only acceptance suite produced `8 passed, 1 skipped` in `4.16s`.
Both skipped tests are GPU checks: Warp reported `CUDA driver not available`
and `torch.cuda.is_available()` is false in this sandbox.  Consequently no
8192-environment GPU memory or throughput measurement was run or claimed.

The only available GPU reference remains the pre-existing environment-only
measurement from the specification: 83,330 env-steps/s at 8,192 environments
and 10.11 GiB on an RTX A4000.  It is not a new measurement from this run.

## No remaining specification contradiction found

The corrected sections 4, 5, and 7 are implementable with stock PPO.  No
training run was launched.
