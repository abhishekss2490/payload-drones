# Multi-Agent Policy Wiring in mjlab

How to run a **decentralized** N=3 policy inside mjlab's **single-agent** `ManagerBasedRlEnv` contract
without forking rsl-rl.

## The tension

mjlab/rsl-rl assume single-agent shapes:

| | shape |
|---|---|
| observation | `(num_envs, obs_dim)` |
| action | `(num_envs, action_dim)` |
| reward, dones | `(num_envs,)` |

We need N agents that each see **only their own** observation and emit **only their own** 4 motor
commands — decentralization is core to the CoRL comparison, so it cannot be faked with a centralized
network that happens to output 12 numbers.

## Rejected: centralized joint policy

One network, input = all 3 agents' states concatenated, output = 12 motors. Simple, but it *is*
centralized — every agent's action depends on every teammate's state. This forfeits the entire
decentralization claim and is not comparable to CoRL. **Rejected.**

## Chosen: agents folded into the batch dimension, inside the model

Keep mjlab's single-agent contract at the env boundary. Put all MARL structure inside the model.

```
env obs   (num_envs, N * per_agent_dim)      <- mjlab sees a normal single-agent env
   |
   |  reshape inside model.get_latent()
   v
          (num_envs * N, per_agent_dim)      <- N agents become extra batch rows
   |
   |  shared encoders + shared actor MLP  (parameter sharing)
   v
          (num_envs * N, 4)
   |
   |  reshape back
   v
env action (num_envs, N * 4)                 <- mjlab sees a normal single-agent env
```

**Decentralization holds by construction**: each row of the reshaped batch contains exactly one agent's
own observation. No teammate state is reachable from within a row. At deployment, run one row.

### Why this needs no rsl-rl changes

PPO sees a `(num_envs, 12)`-dim action space with a factorized, parameter-shared Gaussian policy. The
joint log-prob is the sum of per-agent log-probs (independence across the action dims already assumed by
the diagonal Gaussian), and the team reward is shared. That is precisely the cooperative
parameter-sharing MAPPO/IPPO objective — obtained for free from the existing single-agent PPO.

The base repo already uses a **scalar-std** `GaussianDistribution` (see `_squashed_forward` in
`rl/models.py`), so the std is naturally shared across agents rather than being 12 independent values.

### Critic: CTDE falls out of the existing structure

The base repo already runs an **asymmetric actor/critic** — the critic receives `dynamics_params`
(privileged) while the deployed student does not. Extending that to CTDE is the same pattern:

- **Actor**: per-agent local obs only → reshaped to `(num_envs * N, ·)`.
- **Critic**: full team state (all agents' dynamics + payload ground truth) → stays at `(num_envs, ·)`,
  emitting one value per env.

Since the team reward is shared, one value per env is correct — no need to replicate V(s) per agent.
This is cleaner than standard MAPPO, which replicates the centralized value across agents.

## Observation layout

Per agent, mirroring the base repo's group structure:

| group | contents | who sees it |
|---|---|---|
| `agent_state` | own pose/vel error rel. payload, own last motor command | actor + critic |
| `payload_state` | payload pose, twist, and goal error — **the only shared channel** | actor + critic |
| `agent_dynamics` | own `[L, m, J, ipos, iquat, km_kf, TWR, tau]` (15-dim, privileged) | **critic + Phase-1 teacher only** |
| `agent_history` | 50-step state-action history (per agent) | **student only** |
| `team_state` | all agents' dynamics + true payload mass | **critic only** |

`payload_state` is deliberately the sole inter-agent channel — it is the "communication through the
payload" that the core claim is about.

## Agent masking (needed for H2, built from day one)

A disabled agent must remain **dead weight hanging on its cable** — CoRL's failure test powers a MAV off
but it keeps hanging. Do not delete the body.

- `active_mask: (num_envs, N)` bool on the env.
- Masked agents: motor commands forced to 0 in the action term (after the `u²` conversion, so thrust is
  exactly zero).
- Masked agents' policy outputs are still computed but discarded — keeps tensor shapes static, which
  matters for a compiled warp model.
- Masked agents contribute no reward and are excluded from H1's tension-share statistics.
- The same mechanism gives a cheap variable-N ablation later: mask k agents from t=0.

## Open questions to resolve empirically

1. **Does the shared-reward credit assignment suffice?** With a purely team reward, a lazy agent is not
   penalized directly. If load sharing fails to emerge, the fix is *not* an allocation reward term
   (that would defeat H1) but likely a per-agent action/thrust-efficiency cost, which CoRL already has.
2. **Do we need CoRL's downwash-avoidance term?** They weight it heavily (λ₆=3.0). With heterogeneous
   sizes the downwash geometry differs per agent — may matter more for us, not less.
3. **Symmetry breaking at init.** Three identical-looking agents may collapse to identical behavior
   early in training. Heterogeneous dynamics should break this naturally — that is H4 — but confirm it
   rather than assume it.
