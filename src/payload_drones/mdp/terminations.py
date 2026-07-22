"""Failure conditions for the cable-coupled team."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.terminations import nan_detection, time_out
from mjlab.managers.termination_manager import TerminationTermCfg

from payload_drones.mdp.observations import _payload, _ref_pos
from payload_drones.mdp.team_scaling import N_AGENTS


def payload_ground_contact(env: ManagerBasedRlEnv, threshold: float = 0.15) -> torch.Tensor:
    return _payload(env).data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] < threshold


def payload_far_from_goal(env: ManagerBasedRlEnv, command_name: str = "payload_pose", max_dist: float = 3.0) -> torch.Tensor:
    return (_payload(env).data.root_link_pos_w - _ref_pos(env, command_name)).norm(dim=-1) > max_dist


def any_quad_tumbling(env: ManagerBasedRlEnv, max_ang_vel: float = 50.0) -> torch.Tensor:
    rates = torch.stack([env.scene[f"quad_{i}"].data.root_link_ang_vel_b.norm(dim=-1) for i in range(N_AGENTS)], -1)
    active = env.team_state.active_mask if hasattr(env, "team_state") else torch.ones_like(rates, dtype=torch.bool)
    return ((rates > max_ang_vel) & active).any(-1)


def quad_collision(env: ManagerBasedRlEnv, clearance: float = 0.05) -> torch.Tensor:
    """Terminate when two rotor disks would physically intersect.

    The separation bound must depend on the pair's own arm lengths, which
    randomize over 0.05-0.16 m: two 0.16 m quads overlap below 0.32 m, while two
    0.05 m quads are still 0.20 m clear at that distance. A single fixed
    threshold is therefore either too lax for large quads or unsurvivable for
    small ones -- the previous fixed 0.25 m was the latter, sitting 1 cm below
    the 0.2598 m spawn separation so episodes died at t=0 regardless of policy.
    """
    pos = torch.stack([env.scene[f"quad_{i}"].data.root_link_pos_w for i in range(N_AGENTS)], 1)
    d = torch.cdist(pos, pos)
    pair = torch.triu_indices(N_AGENTS, N_AGENTS, offset=1, device=env.device)
    dist = d[:, pair[0], pair[1]]

    if not hasattr(env, "team_state"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    arms = torch.stack([s.arm_length for s in env.team_state.agents], dim=1)
    min_sep = arms[:, pair[0]] + arms[:, pair[1]] + clearance
    return (dist < min_sep).any(-1)


def cable_overextension(env: ManagerBasedRlEnv, factor: float = 1.5) -> torch.Tensor:
    if not hasattr(env, "team_state"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    p = _payload(env)
    lengths = []
    for i in range(N_AGENTS):
        q = env.scene[f"quad_{i}"]
        hook = q.data.site_pos_w[:, q.find_sites(("hook",))[0][0]]
        attach = p.data.site_pos_w[:, p.find_sites((f"attach_{i}",))[0][0]]
        lengths.append((attach - hook).norm(dim=-1))
    return (torch.stack(lengths, -1) > factor * env.team_state.cable_length).any(-1)


termination_terms = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "payload_ground_contact": TerminationTermCfg(func=payload_ground_contact),
    "payload_far_from_goal": TerminationTermCfg(func=payload_far_from_goal),
    "any_quad_tumbling": TerminationTermCfg(func=any_quad_tumbling),
    "quad_collision": TerminationTermCfg(func=quad_collision),
    "cable_overextension": TerminationTermCfg(func=cable_overextension),
    "nan_detection": TerminationTermCfg(func=nan_detection),
}
