"""Curriculum hooks for progressively sharper payload tracking."""

from __future__ import annotations

import torch

from mjlab.envs.mdp.curriculums import reward_curriculum
from mjlab.managers.curriculum_manager import CurriculumTermCfg


def team_thrust_to_weight(env, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Collective thrust-to-weight used for payload trajectory curriculum bins.

    This intentionally differs from the base single-quad TWR: all three
    vehicles contribute their four-motor maximum thrust, while the suspended
    payload contributes to the mass that must be supported.
    """
    if not hasattr(env, "team_state"):
        raise RuntimeError("TeamState must exist before computing team TWR")
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    team = env.team_state
    max_thrust = sum(agent.max_thrust_per_motor[ids] * 4.0 for agent in team.agents)
    total_mass = sum(agent.mass[ids] for agent in team.agents) + team.payload_mass[ids]
    return max_thrust / (total_mass * 9.81).clamp_min(1.0e-6)


def team_twr_bin(
    env,
    bin_edges: torch.Tensor,
    num_bins: int,
    env_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return collective TWR and its curriculum bin for each requested team."""
    twr = team_thrust_to_weight(env, env_ids)
    bins = torch.bucketize(twr, bin_edges[1:-1]).clamp(0, num_bins - 1)
    return twr, bins

_ITER = 256

curriculum_terms = {
    "payload_position_tracking_std": CurriculumTermCfg(
        func=reward_curriculum,
        params={"reward_name": "payload_position_tracking", "stages": [{"step": 0, "params": {"std": 0.75}}, {"step": 200 * _ITER, "params": {"std": 0.5}}]},
    ),
    "payload_position_fine_std": CurriculumTermCfg(
        func=reward_curriculum,
        params={"reward_name": "payload_position_fine", "stages": [{"step": 0, "params": {"std": 0.20}}, {"step": 200 * _ITER, "params": {"std": 0.1}}]},
    ),
}
