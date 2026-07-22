# Spatial-tendon scaffold findings

**Verdict: STOP. Do not start RL work on this scaffold.**  The scene compiles and
the cable law is correct, but the required open-loop equal-hover stability test
fails: after 2,000 steps (10 s at 5 ms) payload drift is **39.601 m**.  The
simulation remains finite, so this is a dynamic/control-coupling failure, not a
NaN or allocation failure.

## What was verified

The scene is assembled through `SceneCfg.spec_fn` after mjlab attaches its
entities.  Inspection of the assembled spec established the actual names:
`quad_0/hook`, `quad_1/hook`, `quad_2/hook`, and
`payload/attach_0..2`.  It compiles with `ntendon == 3`.

Each tendon is spatial, limited, and has range `[0, 0.6]`, stiffness 2,000
N/m, and damping 80.  Direct MuJoCo measurements from
`scripts/validate_scaffold.py`:

| Check | Measurement | Result |
| --- | ---: | --- |
| Taut (payload 5 mm lower) | length 0.605 m; 10.0 N/cable | pass, pull-only |
| Slack (payload 100 mm higher) | length 0.500 m; 0 N/cable | pass |
| Symmetric load share | 2.4525 N/cable; expected 2.4525 N | pass |
| Equal-hover 2,000 steps | finite; 39.601 m payload drift | **fail** |

The load-share value is `0.75 * 9.81 / 3`; the equilibrium stretch is 1.22625
mm.  This confirms that MuJoCo represents the intended unilateral cable rather
than a rigid rod.

## Constraint allocation

At the nominal pose, MuJoCo reports `nefc = 79`.  Of these, 76 are existing
actuator-limit rows and the three tendon upper-limit rows bring the total to
**79**.  `mujoco-warp.put_data` rejects `njmax=64` and accepts the model only
at `njmax >= 79`; use **njmax=128** for headroom.  `nconmax=64` accepted the
non-contact nominal configuration.  The three tendons add only three rows, so
their nominal solver-row cost is small; they are not the dominant allocation.

## GPU throughput and per-environment randomization

No CUDA device is exposed in this execution environment.  Warp reports only a
CPU device and errors with `"cuda" device requested but CUDA is not supported
by the hardware or driver`.  Therefore there are **no defensible RTX A4000
throughput, peak-memory, or maximum-environment numbers** to report.  The
sweep is implemented in `validate_scaffold.py`; it expands `tendon_range`,
writes upper ranges of 0.55 m and 0.65 m to separate world batches, steps, and
catches `torch.cuda.OutOfMemoryError` per requested batch size.  It intentionally
prints `SKIPPED` rather than fabricating measurements when CUDA is unavailable.

## Design concern / next experiment

The failure does not disprove spatial tendons: their unilateral semantics and
static force split are correct.  It does show that equal motor thrust is not a
stability test for this coupled free-body system.  The cable hooks apply forces
below each quad's CoM and the payload is free in all six DoF; without an
attitude/position stabilizer the team departs rapidly.  Before any RL work,
repeat the 2,000-step test on the actual A4000 with a minimal deterministic
geometric attitude/position hold (or temporarily constrain the quad poses).
If that still diverges, tune tendon `solref`/damping and attachment geometry.
Only if the stabilized test fails should the cable representation be replaced
with a `connect` equality constraint or short body chain.

## Environment note

`pyproject.toml` declares the required editable path dependency and the
project `.venv` was created with CPython 3.12.11.  A full fresh `uv sync` could
not complete here because the sandbox has no package-network access; validation
used `uv run --no-sync` with the already-installed working project dependencies
on the Python path.  This does not alter `mj-adapt-drones`.
