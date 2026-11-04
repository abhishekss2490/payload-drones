"""Payload-tracking rewards (deliberately no load-allocation objective)."""

from __future__ import annotations

import math

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.rewards import is_terminated
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



def payload_tracking_reward(
    env: ManagerBasedRlEnv,
    command_name: str = "payload_pose",
    position_scale: float = 1.5,
    position_fine_std: float = 0.5,
    velocity_std: float = 0.5,
    orientation_scale: float = 1.5,
    angular_velocity_std: float = 2.0,
) -> torch.Tensor:
    """Position-gated team tracking score in [0, 1], before manager dt scaling.

    P = 0.625 * coarse_position + 0.375 * fine_position.
    Return P * (0.375 * velocity + 0.500 * orientation + 0.125 * angular_stability).
    Widths specify the error at which a component equals 0.1. Angular stability
    assumes the current task's static attitude reference (zero desired body rate).
    """
    payload = _payload(env)
    reference = env.command_manager.get_term(command_name).command
    position_error = payload.data.root_link_pos_w - (reference[:, :3] + env.scene.env_origins)
    position_error_sq = position_error.square().sum(-1)
    position = (
        0.625 * _norm_kernel(position_error.norm(dim=-1), position_scale)
        + 0.375 * _exp_kernel(position_error_sq, position_fine_std)
    )
    velocity_error_sq = (payload.data.root_link_lin_vel_w - reference[:, 7:10]).square().sum(-1)
    orientation_error = quat_error_magnitude(reference[:, 3:7], payload.data.root_link_quat_w)
    angular_velocity_sq = payload.data.root_link_ang_vel_b.square().sum(-1)
    motion = (
        0.375 * _exp_kernel(velocity_error_sq, velocity_std)
        + 0.500 * _norm_kernel(orientation_error, orientation_scale)
        + 0.125 * _exp_kernel(angular_velocity_sq, angular_velocity_std)
    )
    return position * motion


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
    return torch.relu(1.0 - distances[:, pairs[0], pairs[1]] / min_separation).mean(-1)


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


# One positive composite term; all shaping parameters remain fixed during training.
# RewardManager applies dt once to every configured term. At dt=0.01, the failure
# weight below gives a -2 event cost; a time limit alone is not a failure.
reward_terms = {
    "payload_tracking": RewardTermCfg(
        func=payload_tracking_reward,
        weight=1.0,
        params={
            "position_scale": 1.5,
            "position_fine_std": 0.5,
            "velocity_std": 0.5,
            "orientation_scale": 1.5,
            "angular_velocity_std": 2.0,
        },
    ),
    "action_rate": RewardTermCfg(func=action_rate, weight=0.0),
    "body_rate_penalty": RewardTermCfg(
        func=body_rate_penalty, weight=0.0, params={"max_cost": 25.0}
    ),
    # Normalized proximity cost in [0,1]; this preserves the old physical cost:
    # -0.005 * mean(relu(1-d/0.5)) == -0.01 * mean(relu(0.5-d)).
    "quad_proximity_penalty": RewardTermCfg(
        func=quad_proximity_penalty, weight=-0.005, params={"min_separation": 0.5}
    ),
    "cable_slack_penalty": RewardTermCfg(func=cable_slack_penalty, weight=-0.004),
    "crash_penalty": RewardTermCfg(func=is_terminated, weight=-200.0),
}
