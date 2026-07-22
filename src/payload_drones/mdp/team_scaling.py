"""Per-team dynamics state and reset-time randomisation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mj_adapt_drones.mdp.quad_scaling import (
    QuadState,
    ScaleCfg,
    initialize_quad_state,
    quadrotor_dynamics_randomization,
)

N_AGENTS = 3
GRAVITY = 9.81


@dataclass
class TeamState:
    """Randomised parameters which belong to one cooperative team."""

    agents: list[QuadState]
    payload_mass: torch.Tensor
    payload_inertia: torch.Tensor
    cable_length: torch.Tensor
    active_mask: torch.Tensor
    margin: torch.Tensor

    @classmethod
    def zeros(cls, num_envs: int, device: str | torch.device) -> "TeamState":
        return cls(
            agents=[QuadState.zeros(num_envs, device) for _ in range(N_AGENTS)],
            payload_mass=torch.zeros(num_envs, device=device),
            payload_inertia=torch.zeros(num_envs, 3, device=device),
            cable_length=torch.zeros(num_envs, N_AGENTS, device=device),
            active_mask=torch.ones(num_envs, N_AGENTS, dtype=torch.bool, device=device),
            margin=torch.zeros(num_envs, device=device),
        )


def initialize_team_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    """Startup event creating the persistent state buffers used by all terms."""
    del env_ids
    if not hasattr(env, "team_state"):
        env.team_state = TeamState.zeros(env.num_envs, env.device)


def _ids(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    return torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids


@requires_model_fields(
    # The base event is called from this wrapper, so its metadata is not visible
    # to EventManager. Repeat the complete contract here: EventManager expands
    # only fields advertised by *registered* event functions, before reset-time
    # writes begin.
    "body_mass",
    "body_ipos",
    "body_inertia",
    "body_iquat",
    "site_pos",
    "actuator_gear",
    "actuator_dynprm",
    "actuator_ctrlrange",
    "tendon_lengthspring",
    "tendon_range",
    recompute=RecomputeLevel.set_const,
)
def team_dynamics_randomization(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    scale_cfg: ScaleCfg,
    payload_asset_cfg: SceneEntityCfg = SceneEntityCfg("payload", body_names=("payload",)),
    max_payload_mass: float = 2.5,
    cable_length_range: tuple[float, float] = (0.5, 0.8),
    iquat_max_angle: float = 0.175,
) -> None:
    """Randomise every quad independently plus the privileged payload/cable state.

    The validated single-quad event writes ``env.quad_state``.  We intentionally
    point that temporary compatibility slot at a different TeamState member on
    each invocation, so no sampled agent can overwrite another agent's values.
    ``margin`` is sampled as the loaded team thrust-to-weight ratio.  The
    optional upper payload cap can only reduce the payload, so the value stored
    in TeamState is recomputed afterwards as the *realized* loaded ratio.
    """
    if max_payload_mass <= 0:
        raise ValueError("max_payload_mass must be positive")
    initialize_team_state(env, env_ids)
    ids = _ids(env, env_ids)
    team: TeamState = env.team_state

    for i in range(N_AGENTS):
        env.quad_state = team.agents[i]
        asset_cfg = SceneEntityCfg(f"quad_{i}", body_names=("quad",))
        site_cfg = SceneEntityCfg(
            f"quad_{i}",
            site_names=(
                "motor_front_left", "motor_front_right", "motor_back_right", "motor_back_left",
            ),
        )
        actuator_cfg = SceneEntityCfg(f"quad_{i}", actuator_names=("motor_.*",))
        # These cfgs are created inside this wrapper rather than supplied to an
        # EventManager, hence they must be resolved explicitly before reuse.
        asset_cfg.resolve(env.scene)
        site_cfg.resolve(env.scene)
        actuator_cfg.resolve(env.scene)
        quadrotor_dynamics_randomization(
            env,
            ids,
            scale_cfg=scale_cfg,
            asset_cfg=asset_cfg,
            site_cfg=site_cfg,
            actuator_cfg=actuator_cfg,
            iquat_max_angle=iquat_max_angle,
        )

    # Keep a backwards-compatible alias for tools expecting a quad_state.  Team
    # terms always use team_state and therefore never confuse it with agent zero.
    env.quad_state = team.agents[0]

    payload_cfg = payload_asset_cfg
    payload_cfg.resolve(env.scene)
    # A cfg selecting all bodies deliberately retains ``slice(None)`` after
    # resolve; the payload entity contains exactly its one root body.
    payload_body = env.scene[payload_cfg.name].indexing.body_ids.to(env.device).long()
    if payload_body.numel() != 1:
        raise ValueError("payload_asset_cfg must resolve exactly one body")

    # Choose a loaded, rather than unloaded, team TWR.  Let S be the total
    # liftable mass and M the total vehicle mass.  Sampling a ratio strictly
    # below S/M gives payload S/ratio-M > 0 for every sampled team.  Crucially,
    # there is no lower payload clamp: that was the source of prior infeasible
    # weak-team samples.
    team_mass = sum(agent.mass[ids] for agent in team.agents)
    liftable_mass = sum(agent.max_thrust_per_motor[ids] * 4.0 for agent in team.agents) / GRAVITY
    unloaded_twr = liftable_mass / team_mass.clamp_min(1.0e-8)
    # torch.uniform_ accepts scalar bounds, whereas the specified upper bound
    # is per-team.  Draw U(0, 1) and scale each team's interval instead.
    sampled_margin = 1.5 + torch.rand(len(ids), device=env.device) * (
        torch.minimum(torch.full_like(unloaded_twr, 3.0), 0.9 * unloaded_twr) - 1.5
    )
    mass = (liftable_mass / sampled_margin - team_mass).clamp_max(max_payload_mass)
    if not bool((mass > 0).all()):
        raise RuntimeError("conditional payload sampling produced a non-positive mass")
    realized_margin = liftable_mass / (team_mass + mass)
    # The XML inertial tensor is diagonal.  Uniform mass scaling preserves its
    # shape and is the physically consistent operation requested by the spec.
    nominal_mass = env.sim.model.body_mass[ids, payload_body[0]].clone().clamp_min(1e-8)
    nominal_inertia = env.sim.model.body_inertia[ids, payload_body[0]].clone()
    inertia = nominal_inertia * (mass / nominal_mass).unsqueeze(-1)
    env.sim.model.body_mass[ids, payload_body[0]] = mass
    env.sim.model.body_inertia[ids, payload_body[0]] = inertia
    team.payload_mass[ids] = mass
    team.payload_inertia[ids] = inertia
    team.margin[ids] = realized_margin

    cable_ids = torch.tensor(
        [env.scene.spec.tendon(f"cable_{i}").id for i in range(N_AGENTS)],
        device=env.device,
        dtype=torch.long,
    )
    length = torch.empty(len(ids), N_AGENTS, device=env.device).uniform_(*cable_length_range)
    e, t = torch.meshgrid(ids, cable_ids, indexing="ij")
    # MjSpec's springlength/range map to the batched tendon_lengthspring/range
    # fields in mujoco-warp.  The upper endpoint is the cable rest length.
    env.sim.model.tendon_lengthspring[e, t, 1] = length
    env.sim.model.tendon_range[e, t, 1] = length
    team.cable_length[ids] = length
    team.active_mask[ids] = True


def sample_agent_failure(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    p_failure: float = 0.0,
    min_active: int = 2,
) -> None:
    """Optionally disable one vehicle while leaving it physically attached."""
    if not 0 <= p_failure <= 1:
        raise ValueError("p_failure must lie in [0, 1]")
    if not 0 <= min_active <= N_AGENTS:
        raise ValueError("min_active must be between 0 and N_AGENTS")
    initialize_team_state(env, env_ids)
    ids = _ids(env, env_ids)
    mask = env.team_state.active_mask
    mask[ids] = True
    if p_failure == 0 or min_active >= N_AGENTS:
        return
    fail_envs = ids[torch.rand(len(ids), device=env.device) < p_failure]
    if len(fail_envs):
        chosen = torch.randint(N_AGENTS, (len(fail_envs),), device=env.device)
        mask[fail_envs, chosen] = False


def settle_quads_for_cable_slack(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    slack_ratio: float = 0.95,
) -> None:
    """Place each quad directly above its own attach point, cable slightly slack.

    This event deliberately runs after every root-reset event and **fully owns
    quad placement** -- horizontal and vertical.

    Horizontal: each quad is centred over its own attach site, so the spawn
    triangle always equals the payload's attachment triangle. That geometry was
    previously duplicated in three places -- ``payload.xml``'s attach sites,
    ``builder.py``'s ``_QUAD_POSITIONS``, and hard-coded per-quad pose ranges in
    the env config -- and they silently disagreed: enlarging the first two left
    the third at the old 0.15 m radius, so quads still spawned 0.2598 m apart and
    315 of 512 envs began already inside the collision bound. Deriving x/y from
    the live attach sites removes that class of bug entirely.

    Vertical: cable rest lengths are independent per agent, whereas payload
    translation gives all three tendons the same geometric length under a
    symmetric attachment layout. Moving each quad vertically preserves that DR
    axis and gives every cable the same relative slack.
    """
    if not 0.0 < slack_ratio <= 1.0:
        raise ValueError("slack_ratio must lie in (0, 1]")
    ids = _ids(env, env_ids)
    if not hasattr(env, "team_state"):
        raise RuntimeError("team_state must exist before settling cable geometry")

    # Root-reset events have written qpos/qvel but not necessarily refreshed
    # derived site poses. Refresh once, then use the actual hook/attach geometry
    # rather than hard-coding spawn heights or attachment offsets.
    env.sim.forward()
    payload = env.scene["payload"]
    target_lengths = slack_ratio * env.team_state.cable_length[ids]

    for i in range(N_AGENTS):
        quad = env.scene[f"quad_{i}"]
        hook_id = quad.find_sites(("hook",))[0][0]
        attach_id = payload.find_sites((f"attach_{i}",))[0][0]
        hook = quad.data.site_pos_w[ids, hook_id]
        attach = payload.data.site_pos_w[ids, attach_id]

        # Centre the hook over its attach point, so the cable is exactly vertical
        # and the spawn triangle equals the attachment triangle by construction.
        horizontal_offset = attach[:, :2] - hook[:, :2]
        # With the hook centred, the whole cable length is vertical.
        vertical = target_lengths[:, i]
        desired_hook_z = attach[:, 2] + vertical

        root_pose = quad.data.root_link_pose_w[ids].clone()
        root_pose[:, :2] += horizontal_offset
        root_pose[:, 2] += desired_hook_z - hook[:, 2]
        quad.write_root_link_pose_to_sim(root_pose, env_ids=ids)
        quad.write_root_link_velocity_to_sim(
            torch.zeros(len(ids), 6, device=env.device), env_ids=ids,
        )
