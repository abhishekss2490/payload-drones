"""mjlab task registration for the two payload RMA phases.

Observation groups are gated per phase for a reason that only shows up at
training scale. `RolloutStorage` allocates every observation group for
`num_steps_per_env * num_envs` transitions, so a group that no network reads
still costs full memory:

    agent_history: 6300 dims x 8192 envs x 256 steps x 4 B = 49.22 GiB

That is 80 % of the observation payload and an immediate OOM on a 15.6 GiB card.
Phase 1's teacher reads privileged `agent_dynamics`, never history; only Phase
2's student TCN needs it. `team_probe` is analysis-only and read by no network
at all, so it is off for both training phases and on for play/eval.
"""

from mjlab.tasks.registry import register_mjlab_task

from payload_drones.config.rl_config import payload_rma_phase1_cfg, payload_rma_phase2_cfg
from payload_drones.payload_env_cfg import make_payload_env_cfg
from payload_drones.rl.runner import MARLDistillationRunner, MARLPhase1Runner

register_mjlab_task(
    task_id="Mjlab-PayloadTransport-RMA-Phase1",
    env_cfg=make_payload_env_cfg(include_history=False, include_team_probe=False),
    play_env_cfg=make_payload_env_cfg(
        num_envs=1, include_history=False, include_team_probe=True, play=True
    ),
    rl_cfg=payload_rma_phase1_cfg(),
    runner_cls=MARLPhase1Runner,
)
register_mjlab_task(
    task_id="Mjlab-PayloadTransport-RMA-Phase2",
    env_cfg=make_payload_env_cfg(include_history=True, include_team_probe=False),
    play_env_cfg=make_payload_env_cfg(
        num_envs=1, include_history=True, include_team_probe=True, play=True
    ),
    rl_cfg=payload_rma_phase2_cfg(),
    runner_cls=MARLDistillationRunner,
)
