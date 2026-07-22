"""Configuration factory for ``PayloadTransport-v0``."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import reset_root_state_uniform, reset_scene_to_default
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mj_adapt_drones.mdp.quad_scaling import ScaleCfg

from payload_drones.mdp.actions import TeamThrustActionCfg
from payload_drones.mdp.curriculum import curriculum_terms
from payload_drones.mdp.observations import (
    agent_dynamics_group,
    agent_history_group,
    critic_group,
    payload_trajectory_group,
    policy_group,
    team_probe_group,
)
from payload_drones.mdp.payload_command import PayloadTrajectoryCommandCfg
from payload_drones.mdp.rewards import reward_terms
from payload_drones.mdp.team_scaling import (
    initialize_team_state,
    sample_agent_failure,
    settle_quads_for_cable_slack,
    team_dynamics_randomization,
)
from payload_drones.mdp.terminations import termination_terms
from payload_drones.scene.builder import make_payload_scene_cfg


def make_payload_env_cfg(
    num_envs: int = 4096,
    scale_cfg: ScaleCfg | None = None,
    episode_length_s: float = 10.0,
    p_failure: float = 0.0,
    trajectory_enabled: bool = True,
    max_payload_speed: float = 1.5,
    include_history: bool = False,
    include_team_probe: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the fixed-three-agent environment; default failure is evaluation-off.

    ``include_history`` gates the 6300-dim ``agent_history`` group, mirroring the
    base repo's ``make_quadrotor_env_cfg(include_history=...)``. It is the input
    to the Phase 2 adaptation module and is **unused by the Phase 1 teacher**,
    which reads privileged ``agent_dynamics`` instead. Leaving it on during
    Phase 1 puts it in the PPO rollout buffer, where it is 80 % of the stored
    observation payload: at 8192 envs x 256 steps that is 49.22 GiB of dead
    weight and an immediate CUDA OOM on a 15.6 GiB card.

    ``include_team_probe`` gates the analysis-only tension/thrust channel. It is
    never read by any network, so training configs should switch it off; keep it
    on for evaluation runs that measure H1/H2.
    """
    scale_cfg = ScaleCfg() if scale_cfg is None else scale_cfg
    observations = {
        "policy": policy_group,
        "agent_dynamics": agent_dynamics_group,
        "critic": critic_group,
        "payload_trajectory": payload_trajectory_group,
    }
    if include_history:
        observations["agent_history"] = agent_history_group
    if include_team_probe:
        observations["team_probe"] = team_probe_group
    if play:
        for group in observations.values():
            group.enable_corruption = False

    # Insertion order is reset semantics: model defaults, randomised constants,
    # optional failure, then state reset against the freshly randomised model.
    events = {
        "initialize_team_state": EventTermCfg(func=initialize_team_state, mode="startup"),
        "reset_scene_to_default": EventTermCfg(func=reset_scene_to_default, mode="reset"),
        "team_dynamics_randomization": EventTermCfg(
            func=team_dynamics_randomization,
            mode="reset",
            params={"scale_cfg": scale_cfg, "payload_asset_cfg": SceneEntityCfg("payload", body_names=("payload",))},
        ),
        "sample_agent_failure": EventTermCfg(
            func=sample_agent_failure, mode="reset", params={"p_failure": p_failure, "min_active": 2}
        ),
        "reset_payload_root": EventTermCfg(
            func=reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("payload"),
                # Spawn the load inside the goal altitude band (0.8-2.0 m), not
                # just clear of the floor. At the previous z=0.2 the payload sat
                # 5 cm above the 0.15 m ground-contact termination and below
                # every sampled goal, so episodes ended in ~11 of 1000 steps
                # before the policy could learn anything. Quads follow
                # automatically: the settle event places each one at its own
                # cable length above the payload.
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (1.0, 1.4), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {},
            },
        ),
        "reset_quad_0_root": EventTermCfg(func=reset_root_state_uniform, mode="reset", params={"asset_cfg": SceneEntityCfg("quad_0"), "pose_range": {"x": (0.15, 0.15), "y": (0.0, 0.0), "z": (0.96, 0.96), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}, "velocity_range": {}}),
        "reset_quad_1_root": EventTermCfg(func=reset_root_state_uniform, mode="reset", params={"asset_cfg": SceneEntityCfg("quad_1"), "pose_range": {"x": (-0.075, -0.075), "y": (0.129904, 0.129904), "z": (0.96, 0.96), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}, "velocity_range": {}}),
        "reset_quad_2_root": EventTermCfg(func=reset_root_state_uniform, mode="reset", params={"asset_cfg": SceneEntityCfg("quad_2"), "pose_range": {"x": (-0.075, -0.075), "y": (-0.129904, -0.129904), "z": (0.96, 0.96), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}, "velocity_range": {}}),
        # Must follow all root-state resets: independent rest lengths require
        # independent quad heights to begin with equal relative cable slack.
        "settle_quads_for_cable_slack": EventTermCfg(
            func=settle_quads_for_cable_slack, mode="reset", params={"slack_ratio": 0.95},
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=make_payload_scene_cfg(num_envs=num_envs),
        observations=observations,
        actions={"team_thrust": TeamThrustActionCfg(entity_name="quad_0")},
        commands={"payload_pose": PayloadTrajectoryCommandCfg(
            episode_length_s=episode_length_s,
            trajectory_enabled=trajectory_enabled,
            max_payload_speed=max_payload_speed,
        )},
        events=events,
        rewards=reward_terms,
        terminations=termination_terms,
        curriculum=curriculum_terms,
        sim=SimulationCfg(nconmax=64, njmax=128, mujoco=MujocoCfg(integrator="euler", timestep=0.005)),
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="payload", body_name="payload", distance=3.0),
        decimation=2,
        episode_length_s=episode_length_s,
    )
