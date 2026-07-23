"""Payload-tracking rewards (deliberately no load-allocation objective)."""

from __future__ import annotations

import math

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.rewards import is_alive, is_terminated
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.lab_api.math import quat_error_magnitude

from payload_drones.mdp.observations import _payload, _ref_pos
from payload_drones.mdp.team_scaling import N_AGENTS


_GAUSSIAN_SCALE = math.sqrt(-2 * math.log(0.1))


def _exp_kernel(error_sq: torch.Tensor, std: float) -> torch.Tensor:
    """Gaussian kernel that is 1 at zero error and about 0.1 at ``std``."""
    return torch.exp(-0.5 * (error_sq / (std**2)) * _GAUSSIAN_SCALE**2)


def payload_position_tracking(env: ManagerBasedRlEnv, command_name: str = "payload_pose", std: float = 0.5) -> torch.Tensor:
    error_sq = (_payload(env).data.root_link_pos_w - _ref_pos(env, command_name)).square().sum(-1)
    return _exp_kernel(error_sq, std)


def payload_orientation_tracking(env: ManagerBasedRlEnv, command_name: str = "payload_pose", std: float = 0.3) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(cmd.quat_command, _payload(env).data.root_link_quat_w)
    return _exp_kernel(error.square(), std)


def payload_velocity_tracking(env: ManagerBasedRlEnv, command_name: str = "payload_pose", std: float = 0.5) -> torch.Tensor:
    ref = env.command_manager.get_term(command_name).command[:, 7:10]
    error_sq = (_payload(env).data.root_link_lin_vel_w - ref).square().sum(-1)
    return _exp_kernel(error_sq, std)


def payload_ang_vel_penalty(env: ManagerBasedRlEnv, std: float = 1.0) -> torch.Tensor:
    return _exp_kernel(_payload(env).data.root_link_ang_vel_b.square().sum(-1), std)


def _masked_mean(env: ManagerBasedRlEnv, values: torch.Tensor) -> torch.Tensor:
    mask = env.team_state.active_mask.float() if hasattr(env, "team_state") else torch.ones_like(values)
    return (values * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
    delta = (env.action_manager.action - env.action_manager.prev_action).reshape(env.num_envs, N_AGENTS, 4)
    return _masked_mean(env, delta.square().mean(-1))


def body_rate_penalty(env: ManagerBasedRlEnv, max_cost: float = 25.0) -> torch.Tensor:
    """Bounded mean squared body rate across active vehicles.

    A catastrophic final integration step can overshoot the 50 rad/s tumbling
    threshold by an arbitrary amount.  Capping this shaping cost keeps that one
    step from overwhelming the explicit, fixed terminal penalty.
    """
    rates = torch.stack([env.scene[f"quad_{i}"].data.root_link_ang_vel_b.square().mean(-1) for i in range(N_AGENTS)], dim=-1)
    return _masked_mean(env, rates).clamp(max=max_cost)


def quad_proximity_penalty(env: ManagerBasedRlEnv, min_separation: float = 0.5) -> torch.Tensor:
    pos = torch.stack([env.scene[f"quad_{i}"].data.root_link_pos_w for i in range(N_AGENTS)], dim=1)
    distances = torch.cdist(pos, pos)
    pairs = torch.triu_indices(N_AGENTS, N_AGENTS, offset=1, device=env.device)
    return torch.relu(min_separation - distances[:, pairs[0], pairs[1]]).mean(-1)


def cable_slack_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    if not hasattr(env, "team_state"):
        return torch.zeros(env.num_envs, device=env.device)
    payload = _payload(env)
    values = []
    for i in range(N_AGENTS):
        q = env.scene[f"quad_{i}"]
        hook = q.data.site_pos_w[:, q.find_sites(("hook",))[0][0]]
        attach = payload.data.site_pos_w[:, payload.find_sites((f"attach_{i}",))[0][0]]
        values.append(((attach - hook).norm(dim=-1) < env.team_state.cable_length[:, i]).float())
    return _masked_mean(env, torch.stack(values, -1))


# Initial widths are the first stage of the live curriculum.  In particular,
# the coarse position std is intentionally kept in the reward config so Tyro's
# ``--env.rewards.payload-position-tracking.params.std`` flag can select the
# early capture basin; the curriculum only overwrites it at later stages.
reward_terms = {
    # Coarse position std stays WIDE at 0.75 for capture during learning, then
    # the curriculum bounds it down (0.60 -> 0.45) only after iteration 1000,
    # once the policy can already fly. Keeping it wide early preserves gradient
    # at the 0.7-1.8 m the aligned-spawn baseline actually drifted to; tightening
    # it from the start would starve the very gradient that closes the offset.
    # (broad-capture at 1.5 was falsified on the cluster: it drifted, not tracked.)
    "payload_position_tracking": RewardTermCfg(func=payload_position_tracking, weight=0.30, params={"std": 0.75}),
    "payload_position_fine": RewardTermCfg(func=payload_position_tracking, weight=0.20, params={"std": 0.20}),
    # Orientation std widened 0.75 -> 1.0 so there is gradient where the payload
    # actually sits: at 1.1 rad error std=0.75 gives ~0.007 (no signal) while
    # std=1.0 gives ~0.06. The orientation curriculum that tightened this toward
    # 0.30 is removed (see curriculum.py) -- tightening removed the little signal
    # the weakly-actuated attitude axis had before it was ever being held.
    "payload_orientation_tracking": RewardTermCfg(func=payload_orientation_tracking, weight=0.20, params={"std": 1.0}),
    "payload_velocity_tracking": RewardTermCfg(func=payload_velocity_tracking, weight=0.15, params={"std": 1.0}),
    "payload_ang_vel_penalty": RewardTermCfg(func=payload_ang_vel_penalty, weight=0.15, params={"std": 2.0}),
    # At the worst bounded shaping values, a live step still has positive
    # reward: 0.10 alive - 0.05 body-rate - 0.008 action-rate - 0.005 proximity
    # - 0.004 slack = +0.033 before non-negative tracking terms.  Since a real
    # termination additionally costs 2.0 after dt scaling, early termination
    # cannot improve return by escaping future shaping costs.
    "is_alive": RewardTermCfg(func=is_alive, weight=0.10),
    "action_rate": RewardTermCfg(func=action_rate, weight=0.0),
    "body_rate_penalty": RewardTermCfg(func=body_rate_penalty, weight=0.0, params={"max_cost": 25.0}),
    "quad_proximity_penalty": RewardTermCfg(func=quad_proximity_penalty, weight=-0.01),
    "cable_slack_penalty": RewardTermCfg(func=cable_slack_penalty, weight=-0.004),
    # Rewards are scaled by dt=0.01, so this is a -2.0 terminal event cost.
    "crash_penalty": RewardTermCfg(func=is_terminated, weight=-200.0),
}
