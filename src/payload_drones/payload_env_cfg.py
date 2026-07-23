"""Configuration factory for ``PayloadTransport-v0``."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from payload_drones.mdp.quad_scaling import ScaleCfg

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
    # Pin the randomization ranges explicitly rather than inheriting whatever
    # the installed base repo happens to default to. The KIAC checkout is 26
    # commits ahead of the development checkout and defaults `twr_range` to
    # (2.0, 6.0) where ours is (2.0, 4.0) -- so an unpinned ScaleCfg() would
    # silently give clusters and laptops different team capability, different
    # payload masses via the capacity-conditioned sampling of section 2b, and
    # different curriculum TWR bins. Every feasibility number in env_spec.md was
    # measured at (2.0, 4.0).
    scale_cfg = ScaleCfg(twr_range=(2.0, 4.0)) if scale_cfg is None else scale_cfg
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
    # then optional failure.  PayloadTrajectoryCommand owns root-state spawn
    # after reset events have completed, exactly as the validated base command
    # does; this prevents the physical state and reference being sampled apart.
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
        # The contact-free cable scaffold needed only 79 EFC rows.  Once the
        # intended plane and real crash contacts are present, one 64-env sanity
        # rollout measured a requirement of 165 rows.  Use enough capacity for
        # all 64 configured contacts (four pyramidal rows each), tendons, and
        # actuator constraints without overflow during crash transients.
        sim=SimulationCfg(nconmax=64, njmax=512, mujoco=MujocoCfg(integrator="euler", timestep=0.005)),
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="payload", body_name="payload", distance=3.0),
        decimation=2,
        episode_length_s=episode_length_s,
    )
