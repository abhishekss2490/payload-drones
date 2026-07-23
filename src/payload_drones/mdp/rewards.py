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
_LN10 = math.log(10.0)


def _exp_kernel(error_sq: torch.Tensor, std: float) -> torch.Tensor:
    """Gaussian kernel: 1 at zero error, ~0.1 at ``std``. Light-tailed (exp of the
    SQUARED error), so it goes flat quickly -- good for a sharp precision well near
    zero, poor for capturing a large error."""
    return torch.exp(-0.5 * (error_sq / (std**2)) * _GAUSSIAN_SCALE**2)


def _norm_kernel(error: torch.Tensor, scale: float) -> torch.Tensor:
    """Heavy-tailed kernel: 1 at zero error, ~0.1 at ``scale``, exp of the NORM.

    This is the CoRL (Zeng et al. 2025) tracking-reward shape, ``exp(-lambda*||e||)``
    with ``lambda = ln(10)/scale`` so the reward is 0.1 at ``error = scale`` (matching
    the Gaussian's convention). Unlike the Gaussian it keeps real gradient at large
    errors -- e.g. at scale=1.5 it is still 0.22 at 1 m and 0.07 at 1.8 m, where the
    Gaussian is ~0. A slow cable-suspended payload drifts to 0.7-1.8 m from its
    reference, exactly the range where the Gaussian went dead, so the coarse tracking
    terms use this shape and a Gaussian fine term supplies precision near zero.
    """
    return torch.exp(-_LN10 * error / scale)


def payload_position_tracking(env: ManagerBasedRlEnv, command_name: str = "payload_pose", scale: float = 1.5) -> torch.Tensor:
    """Coarse position capture: heavy-tailed, gradient across the whole drift range."""
    error = (_payload(env).data.root_link_pos_w - _ref_pos(env, command_name)).norm(dim=-1)
    return _norm_kernel(error, scale)


def payload_position_fine(env: ManagerBasedRlEnv, command_name: str = "payload_pose", std: float = 0.2) -> torch.Tensor:
    """Fine position precision: sharp Gaussian well near zero."""
    error_sq = (_payload(env).data.root_link_pos_w - _ref_pos(env, command_name)).square().sum(-1)
    return _exp_kernel(error_sq, std)


def payload_orientation_tracking(env: ManagerBasedRlEnv, command_name: str = "payload_pose", scale: float = 1.5) -> torch.Tensor:
    """Attitude tracking: heavy-tailed so weakly-actuated attitude has gradient at 1+ rad."""
    cmd = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(cmd.quat_command, _payload(env).data.root_link_quat_w)
    return _norm_kernel(error, scale)


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


reward_terms = {
    # Coarse position + orientation use the heavy-tailed CoRL kernel (exp of the
    # norm), scale 1.5. It keeps gradient across the 0.7-1.8 m / ~1 rad range the
    # payload actually drifts to, where the old Gaussian went dead -- this is the
    # structural fix for loose tracking, replacing the std-tuning workarounds.
    # No coarse curriculum: the kernel is not sparse, so it is held constant and
    # precision comes from the Gaussian fine term below. (`scale` is also the CLI
    # override point for a capture-width ablation.)
    "payload_position_tracking": RewardTermCfg(func=payload_position_tracking, weight=0.30, params={"scale": 1.5}),
    "payload_position_fine": RewardTermCfg(func=payload_position_fine, weight=0.20, params={"std": 0.20}),
    "payload_orientation_tracking": RewardTermCfg(func=payload_orientation_tracking, weight=0.20, params={"scale": 1.5}),
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
