"""Decentralized actor and privileged team observation terms."""

from __future__ import annotations

import torch
import warp as wp
import mujoco

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_conjugate, quat_mul

from payload_drones.mdp.team_scaling import N_AGENTS

POLICY_DIM = 38
DYNAMICS_DIM = 15
HISTORY_LENGTH = 50
TRAJECTORY_WAYPOINTS = 200
TRAJECTORY_FEATURES = 6


def _rot6(quat: torch.Tensor) -> torch.Tensor:
    """First two rotation-matrix columns, a continuous 6D orientation."""
    return matrix_from_quat(quat)[..., :, :2].transpose(-1, -2).reshape(*quat.shape[:-1], 6)


def _payload(env: ManagerBasedRlEnv):
    return env.scene["payload"]


def _ref_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).command[:, :3] + env.scene.env_origins


def _policy_stacked(env: ManagerBasedRlEnv, command_name: str = "payload_pose") -> torch.Tensor:
    payload = _payload(env)
    p_pos = payload.data.root_link_pos_w
    p_vel = payload.data.root_link_lin_vel_w
    p_ang = payload.data.root_link_ang_vel_b
    cmd = env.command_manager.get_term(command_name)
    ref_pos = _ref_pos(env, command_name)
    q_err = quat_mul(cmd.quat_command, quat_conjugate(payload.data.root_link_quat_w))
    q_err = torch.where(q_err[:, :1] < 0, -q_err, q_err)
    payload_goal = torch.cat([ref_pos - p_pos, _rot6(q_err)], dim=-1)
    payload_twist = torch.cat([p_vel, p_ang], dim=-1)

    last = torch.zeros(env.num_envs, N_AGENTS, 4, device=env.device)
    if hasattr(env, "action_manager") and "team_thrust" in env.action_manager.active_terms:
        last = env.action_manager.get_term("team_thrust").motor_command

    parts = []
    for i in range(N_AGENTS):
        quad = env.scene[f"quad_{i}"]
        hook = quad.data.site_pos_w[:, quad.find_sites(("hook",))[0][0]]
        attach = payload.data.site_pos_w[:, payload.find_sites((f"attach_{i}",))[0][0]]
        cable = attach - hook  # hook -> attach, exactly as specified
        cable_len = cable.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        # This is a binary cable-state sensor, not the cable length.  It reads
        # the physical tendon model rather than TeamState, so the actor cannot
        # access the privileged randomisation tensor itself.
        tendon_id = env.scene.spec.tendon(f"cable_{i}").id
        rest_length = env.sim.model.tendon_lengthspring[:, tendon_id, 1:2]
        slack = (cable_len < rest_length).float()
        parts.append(torch.cat([
            quad.data.root_link_pos_w - p_pos,
            quad.data.root_link_lin_vel_w - p_vel,
            _rot6(quad.data.root_link_quat_w),
            quad.data.root_link_ang_vel_b,
            last[:, i],
            cable / cable_len,
            slack,
            payload_goal,
            payload_twist,
        ], dim=-1))
    result = torch.stack(parts, dim=1)
    assert result.shape[-1] == POLICY_DIM
    return result


def policy(env: ManagerBasedRlEnv, command_name: str = "payload_pose") -> torch.Tensor:
    """Actor observations, flattened only at the mjlab manager boundary."""
    return _policy_stacked(env, command_name).reshape(env.num_envs, -1)


def agent_dynamics_stacked(env: ManagerBasedRlEnv) -> torch.Tensor:
    if not hasattr(env, "team_state"):
        return torch.zeros(env.num_envs, N_AGENTS, DYNAMICS_DIM, device=env.device)
    fields = []
    for qs in env.team_state.agents:
        fields.append(torch.cat([
            qs.arm_length[:, None], qs.mass[:, None], qs.inertia, qs.ipos, qs.iquat,
            qs.km_kf[:, None], qs.twr[:, None], qs.motor_tau[:, None],
        ], dim=-1))
    return torch.stack(fields, dim=1)


def agent_dynamics(env: ManagerBasedRlEnv) -> torch.Tensor:
    return agent_dynamics_stacked(env).reshape(env.num_envs, -1)


def agent_history(env: ManagerBasedRlEnv, command_name: str = "payload_pose") -> torch.Tensor:
    """Ego-only rolling ``(policy, action)`` histories for all three agents."""
    dim = POLICY_DIM + 4
    if not hasattr(env, "_team_history"):
        env._team_history = torch.zeros(env.num_envs, N_AGENTS, HISTORY_LENGTH, dim, device=env.device)
    buf: torch.Tensor = env._team_history
    reset = env.episode_length_buf == 0
    if reset.any():
        buf[reset] = 0
    action = torch.zeros(env.num_envs, N_AGENTS, 4, device=env.device)
    if "team_thrust" in env.action_manager.active_terms:
        action = env.action_manager.get_term("team_thrust").motor_command
    current = torch.cat([_policy_stacked(env, command_name), action], dim=-1)
    buf[:, :, :-1] = buf[:, :, 1:].clone()
    buf[:, :, -1] = current
    return buf.reshape(env.num_envs, -1)


def critic(env: ManagerBasedRlEnv, command_name: str = "payload_pose") -> torch.Tensor:
    """Centralized privileged state; never route this group to the actor."""
    if hasattr(env, "team_state"):
        t = env.team_state
        privileged = torch.cat([
            t.payload_mass[:, None], t.payload_inertia, t.cable_length,
            t.active_mask.float(), t.margin[:, None],
        ], dim=-1)
    else:
        # mass(1) + inertia(3) + cable lengths(3) + active mask(3) + margin(1)
        privileged = torch.zeros(env.num_envs, 11, device=env.device)
    return torch.cat([policy(env, command_name), agent_dynamics(env), privileged], dim=-1)


def team_probe(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Analysis channel: tension, thrust fraction, and payload wrench residual."""
    tension_parts = []
    # Cables are upper-limit constraints, so their physical force lives in the
    # solver EFC rows rather than stiffness * extension. Warp keeps capacity-
    # sized EFC buffers; only indices below per-world ``nefc`` are valid.
    efc = env.sim.wp_data.efc
    efc_force = wp.to_torch(efc.force)
    efc_type = wp.to_torch(efc.type)
    efc_id = wp.to_torch(efc.id)
    row = torch.arange(efc_force.shape[1], device=env.device)[None, :]
    valid = row < env.sim.data.nefc[:, None]
    tendon_limit_type = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)
    for i in range(N_AGENTS):
        tendon_id = env.scene.spec.tendon(f"cable_{i}").id
        is_this_tendon = valid & (efc_type == tendon_limit_type) & (efc_id == tendon_id)
        tension_parts.append(torch.where(is_this_tendon, efc_force, 0.0).sum(dim=1).clamp_min(0.0))
    tension = torch.stack(tension_parts, dim=-1)
    thrust = torch.zeros(env.num_envs, N_AGENTS, 4, device=env.device)
    if "team_thrust" in env.action_manager.active_terms:
        thrust = env.action_manager.get_term("team_thrust").motor_command.square()
    payload = _payload(env)
    # MuJoCo's external wrench is the residual after all applied forces.
    wrench = getattr(payload.data, "root_link_force_w", torch.zeros(env.num_envs, 6, device=env.device))
    return torch.cat([tension, thrust.reshape(env.num_envs, -1), wrench], dim=-1)


def payload_trajectory(
    env: ManagerBasedRlEnv,
    command_name: str = "payload_pose",
) -> torch.Tensor:
    """Shared 200-waypoint payload preview, with no per-agent axis.

    Reference positions are local to the current payload position; velocities
    remain the world-frame reference velocity.  The output is intentionally one
    tensor per environment so a later trajectory encoder can run once and
    broadcast its latent to the three agents.
    """
    cmd = env.command_manager.get_term(command_name)
    offsets = torch.arange(1, TRAJECTORY_WAYPOINTS + 1, device=env.device)
    future = (cmd._buf_idx()[:, None] + offsets[None, :]).clamp(max=cmd.N_buf - 1)
    index = future[:, :, None].expand(-1, -1, 3)
    position = cmd.position_ref.gather(1, index)
    velocity = cmd.velocity_ref.gather(1, index)
    payload_local_position = _payload(env).data.root_link_pos_w - env.scene.env_origins
    features = torch.cat([position - payload_local_position[:, None, :], velocity], dim=-1)
    return features.reshape(env.num_envs, TRAJECTORY_WAYPOINTS * TRAJECTORY_FEATURES)


policy_group = ObservationGroupCfg(terms={"policy": ObservationTermCfg(func=policy)}, enable_corruption=False)
agent_dynamics_group = ObservationGroupCfg(terms={"agent_dynamics": ObservationTermCfg(func=agent_dynamics)}, enable_corruption=False)
agent_history_group = ObservationGroupCfg(terms={"agent_history": ObservationTermCfg(func=agent_history)}, enable_corruption=False)
critic_group = ObservationGroupCfg(terms={"critic": ObservationTermCfg(func=critic)}, enable_corruption=False)
team_probe_group = ObservationGroupCfg(terms={"team_probe": ObservationTermCfg(func=team_probe)}, enable_corruption=False)
payload_trajectory_group = ObservationGroupCfg(
    terms={"payload_trajectory": ObservationTermCfg(func=payload_trajectory)}, enable_corruption=False
)
