"""Acceptance tests from env_spec section 9."""

from __future__ import annotations

import math

import pytest
import torch
import warp as wp

# The sandbox's home directory is read-only.  This does not affect simulation;
# it simply gives Warp a writable kernel cache for the test process.
wp.config.kernel_cache_dir = "/tmp/payload-drones-warp-cache"

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from payload_drones.mdp.observations import agent_dynamics, payload_trajectory, policy, team_probe
from payload_drones.mdp.rewards import reward_terms
from payload_drones.mdp.terminations import (
    any_quad_ground_contact,
    payload_ground_contact,
    quad_collision,
)
from payload_drones.config.rl_config import payload_rma_phase1_cfg
from payload_drones.mdp.quad_scaling import ScaleCfg
from payload_drones.mdp.team_scaling import team_dynamics_randomization
from payload_drones.payload_env_cfg import make_payload_env_cfg


def _env(
    num_envs: int,
    device: str = "cpu",
    arm_sampling_mode: str = "correlated",
) -> ManagerBasedRlEnv:
    # These environment-contract tests exercise every observation group. The
    # training configs deliberately gate history off for Phase 1 to avoid
    # RolloutStorage allocating an unused 6300-dim tensor.
    cfg = make_payload_env_cfg(
        num_envs=num_envs,
        include_history=True,
        arm_sampling_mode=arm_sampling_mode,
    )
    return ManagerBasedRlEnv(cfg, device=device)


def test_1_construct_reset_and_random_steps_are_finite():
    env = _env(16)
    obs, _ = env.reset()
    for _ in range(100):
        obs, reward, *_ = env.step(2 * torch.rand(env.num_envs, 12, device=env.device) - 1)
        assert torch.isfinite(reward).all()
        assert all(torch.isfinite(value).all() for value in obs.values())


def test_2_observation_group_shapes():
    env = _env(4)
    obs, _ = env.reset()
    expected = {
        "policy": (4, 3 * 38),
        "agent_dynamics": (4, 3 * 15),
        "agent_history": (4, 3 * 50 * (38 + 4)),
        # policy(all agents), dynamics(all agents), payload mass/inertia,
        # cable lengths, active mask = 114 + 45 + 1 + 3 + 3 + 3.
        "critic": (4, 170),
        "team_probe": (4, 3 + 3 * 4 + 6),
        # This reference is shared by the team: no three-agent axis/copy.
        "payload_trajectory": (4, 200 * 6),
    }
    assert {name: tuple(value.shape) for name, value in obs.items()} == expected


@pytest.mark.parametrize("arm_sampling_mode", ["correlated", "independent"])
def test_3_team_arm_sampling_modes_preserve_physics_dr(arm_sampling_mode: str):
    env = _env(512, arm_sampling_mode=arm_sampling_mode)
    env.reset()
    arms = torch.stack([state.arm_length for state in env.team_state.agents], dim=1)
    mean_within_team_std = arms.std(dim=1).mean()
    pairwise_correlation = torch.corrcoef(arms.T)[
        torch.triu_indices(3, 3, offset=1).unbind()
    ].mean()

    if arm_sampling_mode == "correlated":
        # B~U(.060,.145) with independent +/-10% offsets has about 5.4 mm
        # within-team std and strong across-env pairwise correlation.
        assert 0.003 < mean_within_team_std < 0.008
        assert pairwise_correlation > 0.85
    else:
        # The control remains exactly the original iid U(.050,.160) draw.
        assert mean_within_team_std > 0.02
        assert abs(pairwise_correlation) < 0.12

    # Correlating only arms must not turn the underlying DR into a shared
    # model buffer.  Check elementwise TeamState/model agreement for both
    # modes and independently sampled body-mass variation across envs.
    for agent in range(3):
        body = env.scene.spec.body(f"quad_{agent}/quad").id
        mass = env.sim.model.body_mass[:, body]
        assert torch.allclose(mass, env.team_state.agents[agent].mass)
        assert mass.std() > 0


def test_3a_correlated_arm_configuration_rejects_endpoint_clipping():
    with pytest.raises(ValueError, match="hard arm range"):
        make_payload_env_cfg(
            base_arm_range=(0.060, 0.150), arm_relative_spread=0.10
        )


def test_3b_team_partial_reset_leaves_unselected_arm_samples_unchanged():
    env = _env(16)
    env.reset(seed=314159)
    previous_arms = [agent.arm_length.clone() for agent in env.team_state.agents]
    previous_masses = [agent.mass.clone() for agent in env.team_state.agents]
    ids = torch.tensor([1, 5, 9, 13], device=env.device)
    team_dynamics_randomization(
        env,
        ids,
        scale_cfg=ScaleCfg(twr_range=(2.0, 4.0)),
        payload_asset_cfg=SceneEntityCfg("payload", body_names=("payload",)),
    )
    selected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    selected[ids] = True
    for agent, old_arm, old_mass in zip(env.team_state.agents, previous_arms, previous_masses):
        assert torch.equal(agent.arm_length[~selected], old_arm[~selected])
        assert torch.equal(agent.mass[~selected], old_mass[~selected])


def test_4_domain_randomization_reaches_every_model_field():
    """Direct-physics guard against TeamState/model divergence.

    The registered team wrapper must advertise the base event's model fields.
    Otherwise mujoco-warp keeps stride-zero shared arrays: TeamState varies,
    but the last write is broadcast to every simulated world. This test runs
    after the *complete* reset sequence, so it also catches a later reset term
    restoring model defaults.
    """
    env = _env(512)
    env.reset(seed=314159)
    model = env.sim.model
    spec = env.scene.spec

    def body_id(name: str) -> int:
        return spec.body(name).id

    def actuator_ids(agent: int) -> list[int]:
        return [spec.actuator(f"quad_{agent}/motor_{name}").id for name in (
            "front_left", "front_right", "back_right", "back_left",
        )]

    def site_ids(agent: int) -> list[int]:
        return [spec.site(f"quad_{agent}/motor_{name}").id for name in (
            "front_left", "front_right", "back_right", "back_left",
        )]

    def has_cross_env_variation(value: torch.Tensor, field: str) -> None:
        variation = value.reshape(env.num_envs, -1).std(dim=0).amax()
        assert variation > 1.0e-7, f"{field} is shared across environments: std={variation.item():.3e}"

    # Every base-DR field has an independently checkable TeamState relation.
    signs = torch.tensor([-1.0, 1.0, -1.0, 1.0], device=env.device)
    for agent in range(3):
        state = env.team_state.agents[agent]
        body = body_id(f"quad_{agent}/quad")
        actuators = actuator_ids(agent)
        sites = site_ids(agent)

        assert torch.allclose(model.body_mass[:, body], state.mass)
        assert torch.allclose(model.body_inertia[:, body], state.inertia)
        assert torch.allclose(model.body_ipos[:, body], state.ipos)
        assert torch.allclose(model.body_iquat[:, body], state.iquat)
        assert torch.allclose(
            model.actuator_ctrlrange[:, actuators, 1],
            state.max_thrust_per_motor[:, None].expand(-1, 4),
        )
        assert torch.allclose(
            model.actuator_gear[:, actuators, 5], state.km_kf[:, None] * signs,
        )
        assert torch.allclose(
            model.actuator_dynprm[:, actuators, 0],
            state.motor_tau[:, None].expand(-1, 4),
        )
        # Motor-position noise is intentionally sampled inside the base event
        # and not retained in TeamState. Its radius must still vary with the
        # stored arm length (the base event's own physical-consistency contract).
        site_radius = model.site_pos[:, sites, :2].norm(dim=-1)
        ratio = site_radius / state.arm_length[:, None]
        assert ratio.min() > 0.85 and ratio.max() < 1.15

        for field, value in {
            f"quad_{agent}.body_mass": model.body_mass[:, body],
            f"quad_{agent}.body_ipos": model.body_ipos[:, body],
            f"quad_{agent}.body_inertia": model.body_inertia[:, body],
            f"quad_{agent}.body_iquat": model.body_iquat[:, body],
            f"quad_{agent}.site_pos": model.site_pos[:, sites],
            f"quad_{agent}.actuator_gear": model.actuator_gear[:, actuators],
            f"quad_{agent}.actuator_dynprm": model.actuator_dynprm[:, actuators],
            f"quad_{agent}.actuator_ctrlrange": model.actuator_ctrlrange[:, actuators],
        }.items():
            has_cross_env_variation(value, field)

    payload_body = body_id("payload/payload")
    assert torch.allclose(model.body_mass[:, payload_body], env.team_state.payload_mass)
    assert torch.allclose(model.body_inertia[:, payload_body], env.team_state.payload_inertia)
    has_cross_env_variation(model.body_mass[:, payload_body], "payload.body_mass")
    has_cross_env_variation(model.body_inertia[:, payload_body], "payload.body_inertia")

    for cable in range(3):
        tendon = spec.tendon(f"cable_{cable}").id
        assert torch.allclose(model.tendon_lengthspring[:, tendon, 1], env.team_state.cable_length[:, cable])
        assert torch.allclose(model.tendon_range[:, tendon, 1], env.team_state.cable_length[:, cable])
        assert torch.count_nonzero(model.tendon_stiffness[:, tendon]) == 0
        assert torch.count_nonzero(model.tendon_damping[:, tendon]) == 0
        for field, value in {
            f"cable_{cable}.tendon_lengthspring": model.tendon_lengthspring[:, tendon],
            f"cable_{cable}.tendon_range": model.tendon_range[:, tendon],
        }.items():
            has_cross_env_variation(value, field)


def test_5_reset_geometry_gives_each_cable_small_relative_slack():
    """Independent cable lengths require independently placed quad roots.

    This reads the hook/attach geometry and model spring constants directly;
    TeamState alone cannot establish that the simulated reset is safe.
    """
    env = _env(512)
    env.reset(seed=424242)
    env.sim.forward()
    payload = env.scene["payload"]

    lengths = []
    rest_lengths = []
    quad_root_z = []
    for agent in range(3):
        quad = env.scene[f"quad_{agent}"]
        hook_id = quad.find_sites(("hook",))[0][0]
        attach_id = payload.find_sites((f"attach_{agent}",))[0][0]
        geometric_length = (quad.data.site_pos_w[:, hook_id] - payload.data.site_pos_w[:, attach_id]).norm(dim=-1)
        tendon_id = env.scene.spec.tendon(f"cable_{agent}").id
        actual_tendon_length = env.sim.data.ten_length[:, tendon_id]
        rest_length = env.sim.model.tendon_lengthspring[:, tendon_id, 1]

        # The explicit site distance is a geometry cross-check; ten_length is
        # the simulator quantity whose range/spring law is actually evaluated.
        assert torch.allclose(actual_tendon_length, geometric_length, atol=1.0e-6)
        lengths.append(actual_tendon_length)
        rest_lengths.append(rest_length)
        quad_root_z.append(quad.data.root_link_pos_w[:, 2])

    length = torch.stack(lengths, dim=1)
    rest = torch.stack(rest_lengths, dim=1)
    root_z = torch.stack(quad_root_z, dim=1)
    relative_length = length / rest

    # Check every cable individually: an aggregate can hide a taut cable behind
    # two slack ones.  The configured target is exactly 0.95, with deliberately
    # loose bounds for different tensor/device precisions.
    assert torch.all(length <= rest + 1.0e-6)
    assert torch.all((relative_length >= 0.93) & (relative_length <= 0.97))
    # Strict slack means the upper unilateral limit cannot exert an initial
    # constraint force. Verify the solver-row-backed analysis probe is zero;
    # ``nefc`` alone also counts unrelated actuator rows.
    assert torch.count_nonzero(team_probe(env)[:, :3]) == 0
    # The ground plane is z=0; root height leaves a substantial margin even for
    # the shortest sampled cable, while the event itself derives it from the
    # live attachment positions rather than this test's spawn constants.
    assert root_z.amin() > 0.25


def test_reset_spawns_complete_team_at_reference_with_identity_attitude():
    env = _env(512)
    env.reset(seed=20260722)
    command = env.command_manager.get_term("payload_pose")
    payload = env.scene["payload"]

    position_error = (
        payload.data.root_link_pos_w
        - (command.command[:, :3] + env.scene.env_origins)
    ).norm(dim=-1)
    velocity_error = (
        payload.data.root_link_lin_vel_w - command.command[:, 7:10]
    ).norm(dim=-1)
    max_position_error = (3.0**0.5) * command.cfg.spawn_position_perturbation
    max_velocity_error = (3.0**0.5) * command.cfg.spawn_velocity_perturbation
    identity = torch.zeros(env.num_envs, 4, device=env.device)
    identity[:, 0] = 1.0

    assert position_error.max() <= max_position_error + 1.0e-5
    assert velocity_error.max() <= max_velocity_error + 1.0e-5
    assert torch.equal(command.quat_command, identity)
    assert torch.equal(payload.data.root_link_quat_w, identity)
    # No hover warm-up in the baseline: every episode is a moving trajectory.
    hover_fraction = command.is_hover_episode.float().mean()
    assert hover_fraction == 0.0
    assert not payload_ground_contact(env).any()
    assert not any_quad_ground_contact(env).any()
    assert not quad_collision(env).any()


def test_4_policy_has_no_payload_mass_or_cable_length_leak():
    env = _env(8)
    env.reset()
    before = policy(env).clone()
    env.team_state.payload_mass.add_(100.0)
    env.team_state.cable_length.add_(100.0)
    # The policy tensor uses physical cable geometry only; changing privileged
    # metadata alone must not enter the actor channel.
    assert torch.equal(before, policy(env))


def test_5_inactive_agent_motor_force_is_exactly_zero():
    env = _env(4)
    env.reset()
    env.team_state.active_mask[:, 1] = False
    env.action_manager.process_action(torch.ones(env.num_envs, 12, device=env.device))
    force = env.action_manager.get_term("team_thrust").processed_actions
    assert torch.count_nonzero(force[:, 1]) == 0
    assert torch.count_nonzero(force[:, 0]) > 0


def test_6_composite_reward_matches_hand_calculated_cases():
    """Check position gating, moving references, attitude, rates and quaternion sign."""
    env = _env(7)
    try:
        env.reset(seed=73)
        payload = env.scene["payload"]
        command = env.command_manager.get_term("payload_pose")
        # A nonzero reference velocity catches accidental rewards for absolute stillness.
        command.velocity_ref.zero_()
        command.velocity_ref[:, :, 0] = 0.5
        reference = command.command.clone()
        state = payload.data.default_root_state.clone()
        state[:, :3] = reference[:, :3] + env.scene.env_origins
        state[:, 3:7] = reference[:, 3:7]
        state[:, 7:10] = reference[:, 7:10]
        state[:, 10:13] = 0.0
        state[1:3, 0] += 0.5
        state[2, 7:10] = 0.0
        state[3, 0] += 3.0
        state[4, 3:7] = torch.tensor([math.cos(0.75), 0.0, 0.0, math.sin(0.75)])
        state[5, 10] = 2.0
        command._quat_ref[6] *= -1.0  # q and -q represent the same orientation.
        payload.write_root_state_to_sim(state)
        env.sim.forward()

        cfg = env.reward_manager.get_term_cfg("payload_tracking")
        actual = cfg.func(env, **cfg.params)
        # Independent arithmetic from the approved formulation, before dt scaling.
        expected = torch.tensor([1.0, 0.3275993021, 0.2170345376, 0.00625, 0.55, 0.8875, 1.0])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        assert actual.shape == (env.num_envs,)
        assert torch.isfinite(actual).all()
        assert ((actual >= 0.0) & (actual <= 1.0)).all()
        assert [name for name, term in reward_terms.items() if term.weight > 0] == [
            "payload_tracking"
        ]
        assert "is_alive" not in reward_terms
    finally:
        env.close()


def test_reward_safety_costs_are_bounded_and_scaled_once():
    env = _env(3)
    try:
        env.reset(seed=73)
        # Pair distances produce normalized separation costs 0, 1/3 and 1.
        offsets = torch.tensor([
            [[0.0, 0.0, 3.0], [1.0, 0.0, 3.0], [2.0, 0.0, 3.0]],
            [[0.0, 0.0, 3.0], [0.25, 0.0, 3.0], [0.5, 0.0, 3.0]],
            [[0.0, 0.0, 3.0], [0.0, 0.0, 3.0], [0.0, 0.0, 3.0]],
        ])
        for i in range(3):
            quad = env.scene[f"quad_{i}"]
            pose = quad.data.root_link_pose_w.clone()
            pose[:, :3] = env.scene.env_origins + offsets[:, i]
            quad.write_root_link_pose_to_sim(pose)
        env.sim.forward()

        # Set the geometric slack proxy to zero, one and three slack cables.
        payload = env.scene["payload"]
        for i in range(3):
            quad = env.scene[f"quad_{i}"]
            hook = quad.data.site_pos_w[:, quad.find_sites(("hook",))[0][0]]
            attach = payload.data.site_pos_w[:, payload.find_sites((f"attach_{i}",))[0][0]]
            offsets_i = torch.tensor([-0.1, 0.1 if i == 0 else -0.1, 0.1])
            env.team_state.cable_length[:, i] = (attach - hook).norm(dim=-1) + offsets_i

        expected_costs = torch.tensor([0.0, 1.0 / 3.0, 1.0])
        for name in ("quad_proximity_penalty", "cable_slack_penalty"):
            cfg = env.reward_manager.get_term_cfg(name)
            torch.testing.assert_close(cfg.func(env, **cfg.params), expected_costs)
        task_cfg = env.reward_manager.get_term_cfg("payload_tracking")
        task = task_cfg.func(env, **task_cfg.params)
        env.termination_manager.terminated.zero_()
        env.termination_manager.time_outs.zero_()
        assert env.step_dt == pytest.approx(0.01)
        live = env.reward_manager.compute(env.step_dt).clone()
        torch.testing.assert_close(live, 0.01 * (task - 0.009 * expected_costs))

        # Pure timeout has no failure cost; real failures cost -2, including at timeout.
        env.termination_manager.time_outs[:] = torch.tensor([True, False, True])
        env.termination_manager.terminated[:] = torch.tensor([False, True, True])
        finished = env.reward_manager.compute(env.step_dt).clone()
        torch.testing.assert_close(finished - live, torch.tensor([0.0, -2.0, -2.0]))
        gamma = payload_rma_phase1_cfg().algorithm.gamma
        assert 0.01 * 0.009 / (1.0 - gamma) == pytest.approx(0.018)
        assert reward_terms["action_rate"].weight == 0.0
        assert reward_terms["body_rate_penalty"].weight == 0.0
    finally:
        env.close()


def test_contact_terminations_match_simulated_contacts():
    env = _env(3)
    env.reset(seed=73)

    payload = env.scene["payload"]
    pose = payload.data.root_link_pose_w.clone()
    pose[0, 2] = 0.0
    payload.write_root_link_pose_to_sim(pose)
    env.sim.forward()
    assert payload_ground_contact(env).tolist() == [True, False, False]

    env.reset(seed=73)
    quad_0 = env.scene["quad_0"]
    pose = quad_0.data.root_link_pose_w.clone()
    pose[1, 2] = 0.0
    quad_0.write_root_link_pose_to_sim(pose)
    env.sim.forward()
    assert any_quad_ground_contact(env).tolist() == [False, True, False]

    env.reset(seed=73)
    quad_0 = env.scene["quad_0"]
    quad_1 = env.scene["quad_1"]
    pose = quad_1.data.root_link_pose_w.clone()
    pose[2] = quad_0.data.root_link_pose_w[2]
    quad_1.write_root_link_pose_to_sim(pose)
    env.sim.forward()
    assert quad_collision(env).tolist() == [False, False, True]


def test_reward_parameters_remain_fixed_after_former_curriculum_milestones():
    env = _env(2)
    try:
        env.reset()
        assert env.cfg.curriculum == {}

        def snapshot():
            return {
                name: (cfg.weight, dict(cfg.params))
                for name in env.reward_manager.active_terms
                for cfg in [env.reward_manager.get_term_cfg(name)]
            }

        initial = snapshot()
        for iteration in (500, 1000, 1500, 2000, 10000):
            env.common_step_counter = iteration * 256
            env.curriculum_manager.compute()
            assert snapshot() == initial
    finally:
        env.close()


def test_sweep_parameters_are_direct_config_overrides():
    env_cfg = make_payload_env_cfg(num_envs=2)
    command_cfg = env_cfg.commands["payload_pose"]
    assert env_cfg.rewards["payload_tracking"].params == {
        "position_scale": 1.5,
        "position_fine_std": 0.5,
        "velocity_std": 0.5,
        "orientation_scale": 1.5,
        "angular_velocity_std": 2.0,
    }
    assert env_cfg.curriculum == {}
    # No hover warm-up in the baseline (stationary episodes are unlike the
    # rho=0.25 fast-tracking target); every episode is a moving trajectory.
    assert command_cfg.hover_fraction_initial == 0.0
    assert command_cfg.hover_fraction_final == 0.0
    # Trajectory difficulty is pinned at rho = 0.25 (not performance-gated).
    assert command_cfg.rho_min == 0.25 and command_cfg.rho_max == 0.25

    rl_cfg = payload_rma_phase1_cfg()
    assert rl_cfg.max_iterations == 6000
    assert rl_cfg.algorithm.learning_rate == 1.0e-3
    assert rl_cfg.algorithm.num_learning_epochs == 5
    assert rl_cfg.algorithm.num_mini_batches == 16
    assert rl_cfg.algorithm.schedule == "adaptive"
    assert rl_cfg.algorithm.desired_kl == 0.015


def test_7_gpu_4096_construct_and_step_without_oom():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this execution environment")
    env = ManagerBasedRlEnv(
        make_payload_env_cfg(num_envs=4096, include_history=False, include_team_probe=False),
        device="cuda",
    )
    obs, _ = env.reset()
    env.step(torch.zeros(env.num_envs, 12, device="cuda"))
    assert all(torch.isfinite(value).all() for value in obs.values())


def test_8_payload_trajectory_is_shared_1200_vector():
    env = _env(4)
    obs, _ = env.reset()
    assert tuple(obs["payload_trajectory"].shape) == (env.num_envs, 1200)
    assert tuple(payload_trajectory(env).shape) == (env.num_envs, 1200)


def test_9_moving_trajectory_populates_velocity_reference():
    """Guards the former bug where the velocity slots were permanently zero.

    Two earlier versions of this test were wrong, both for timing reasons:

    1. It stepped 20 times and asserted speed > 1e-5. Trajectories ramp in with
       a quintic smoothstep whose speed goes as 30*u^2*(1-u)^2, so at u=0.02 the
       reference is legitimately ~1% of peak. Unseeded, it failed 3 runs in 5.
    2. It then stepped to mid-episode with zero motor commands. That cannot
       work: zero thrust means the team falls, terminations fire, and the env
       resets — `episode_step` never leaves ramp-in, so it still measured ~1e-7.

    Flight competence is not what this test is about, so do not depend on it.
    Index the buffer directly at a mid-episode step, as test 10 does.
    """
    env = _env(4)
    env.reset(seed=0)
    command = env.command_manager.get_term("payload_pose")

    # The generated buffer must contain real motion somewhere, unlike the
    # former static command whose velocity slots were zero everywhere.
    moving = ~command.is_hover_episode
    assert moving.any(), "seeded batch unexpectedly contains no moving episodes"
    peak_speed = command.velocity_ref[moving].norm(dim=-1).max()
    assert peak_speed > 0.1, f"trajectory buffer is essentially static: peak {peak_speed:.2e} m/s"

    # Read each trajectory at its own non-zero reference-speed sample without
    # relying on an arbitrary common episode time (short paths may finish
    # before the midpoint).
    command.episode_step[:] = command.velocity_ref.norm(dim=-1).argmax(dim=1) * command.cfg.trajectory_decimation
    speed = command.command[moving, 7:10].norm(dim=-1)
    assert torch.all(speed > 1.0e-3), (
        f"velocity reference is ~zero mid-trajectory: {speed}"
    )


def test_10_payload_preview_is_safe_at_episode_end():
    env = _env(2)
    env.reset()
    command = env.command_manager.get_term("payload_pose")
    # Force the final valid current index.  The 200th future waypoint must land
    # on the allocated final buffer entry, never read beyond it.
    command.episode_step[:] = command.N_episode * command.cfg.trajectory_decimation
    preview = payload_trajectory(env).reshape(env.num_envs, 200, 6)
    assert torch.equal(command._buf_idx(), torch.full_like(command._buf_idx(), command.N_episode - 1))
    assert torch.allclose(preview[:, -1, 3:], command.velocity_ref[:, command.N_buf - 1])


def test_11_static_setpoint_fallback_keeps_zero_velocity_reference():
    env = ManagerBasedRlEnv(make_payload_env_cfg(num_envs=3, trajectory_enabled=False), device="cpu")
    obs, _ = env.reset()
    command = env.command_manager.get_term("payload_pose")
    assert not command.cfg.trajectory_enabled
    assert torch.count_nonzero(command.velocity_ref) == 0
    assert torch.count_nonzero(obs["payload_trajectory"].reshape(env.num_envs, 200, 6)[..., 3:]) == 0


def test_12_conditional_payload_sampling_has_a_feasibility_preserving_soft_floor():
    """Every actual reset sample has positive payload and meets its soft floor."""
    env = _env(4096)
    env.reset(seed=1729)
    team = env.team_state
    total_thrust = sum(agent.max_thrust_per_motor * 4.0 for agent in team.agents)
    total_quad_mass = sum(agent.mass for agent in team.agents)
    unloaded_twr = total_thrust / (total_quad_mass * 9.81)
    # Loaded-TWR floor is 2.0 (raised from the old 1.5 to lighten payloads and
    # drop marginal teams, but kept below 2.5 to keep payloads heavy enough that
    # capability-driven load sharing is still exercised -- see the H1 rationale).
    soft_floor = torch.minimum(torch.full_like(unloaded_twr, 2.0), 0.9 * unloaded_twr)
    assert torch.all(team.payload_mass > 0)
    assert torch.all(team.margin >= soft_floor - 1.0e-6)
    assert torch.allclose(team.margin, total_thrust / ((total_quad_mass + team.payload_mass) * 9.81))


def test_13_payload_trajectory_speed_cap_is_respected():
    """The generated reference, not vehicle motion, must honor the speed cap."""
    env = _env(512)
    env.reset(seed=2718)
    command = env.command_manager.get_term("payload_pose")
    peak_speed = command.velocity_ref.norm(dim=-1).amax()
    assert command._current_speed_cap() == 0.5
    assert peak_speed <= 1.05 * command._current_speed_cap()


def test_14_gpu_cable_reset_stays_finite_for_400_random_steps():
    """Full-scale regression for the former step-zero cable launch.

    This is intentionally GPU-only: 4096 x 400 is the training-relevant
    operating point and is prohibitively slow on the CPU fallback.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this execution environment")
    torch.manual_seed(98765)
    env = ManagerBasedRlEnv(
        make_payload_env_cfg(num_envs=4096, include_history=False, include_team_probe=False),
        device="cuda",
    )
    obs, _ = env.reset(seed=98765)
    max_separation = torch.zeros((), device=env.device)
    payload = env.scene["payload"]
    for _ in range(400):
        obs, reward, _, _, _ = env.step(2.0 * torch.rand(env.num_envs, 12, device=env.device) - 1.0)
        assert torch.isfinite(reward).all()
        assert all(torch.isfinite(value).all() for value in obs.values())
        for agent in range(3):
            separation = (env.scene[f"quad_{agent}"].data.root_link_pos_w - payload.data.root_link_pos_w).norm(dim=-1)
            max_separation = torch.maximum(max_separation, separation.amax())
    assert max_separation.item() < 5.0
