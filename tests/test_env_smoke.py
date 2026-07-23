"""Acceptance tests from env_spec section 9."""

from __future__ import annotations

import pytest
import torch
import warp as wp

# The sandbox's home directory is read-only.  This does not affect simulation;
# it simply gives Warp a writable kernel cache for the test process.
wp.config.kernel_cache_dir = "/tmp/payload-drones-warp-cache"

from mjlab.envs import ManagerBasedRlEnv
from payload_drones.mdp.observations import agent_dynamics, payload_trajectory, policy, team_probe
from payload_drones.mdp.rewards import body_rate_penalty, reward_terms
from payload_drones.mdp.terminations import (
    any_quad_ground_contact,
    payload_ground_contact,
    quad_collision,
)
from payload_drones.config.rl_config import payload_rma_phase1_cfg
from payload_drones.payload_env_cfg import make_payload_env_cfg


def _env(num_envs: int, device: str = "cpu") -> ManagerBasedRlEnv:
    # These environment-contract tests exercise every observation group. The
    # training configs deliberately gate history off for Phase 1 to avoid
    # RolloutStorage allocating an unused 6300-dim tensor.
    return ManagerBasedRlEnv(make_payload_env_cfg(num_envs=num_envs, include_history=True), device=device)


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


def test_3_per_team_arm_lengths_are_independent():
    env = _env(512)
    env.reset()
    arms = torch.stack([state.arm_length for state in env.team_state.agents], dim=1)
    # Independent uniform samples occasionally land close together.  Requiring
    # every one of 512 teams to exceed a threshold would reject correct IID
    # sampling; test the population mean instead.
    assert arms.std(dim=1).mean() > 0.02


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
    # The probabilistic check is deliberately broad; it catches disabled or
    # all-hover behavior without making one seeded draw brittle.
    hover_fraction = command.is_hover_episode.float().mean()
    assert 0.30 < hover_fraction < 0.50
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


def test_6_reward_sanity_includes_bounded_survival_economics():
    """The positional kernels have the stated near/far behaviour.

    Section 9's stronger claim that *total* reward at 3 m position error is
    near zero cannot follow from section 6: orientation, velocity, and angular
    velocity terms contribute 0.50 at that state.  This test pins the specified
    kernels and documents the numerical contradiction rather than hiding it.
    """
    env = _env(2)
    env.reset()
    payload = env.scene["payload"]
    command = env.command_manager.get_term("payload_pose")
    state = payload.data.default_root_state.clone()
    state[:, :3] = command.command[:, :3] + env.scene.env_origins
    state[:, 3:7] = command.quat_command
    state[:, 7:] = 0
    payload.write_root_state_to_sim(state)
    env.sim.forward()
    near = sum(term.weight * term.func(env, **term.params) for term in reward_terms.values() if term.weight > 0)
    state[:, 0] += 3.0
    payload.write_root_state_to_sim(state)
    env.sim.forward()
    far_pos = reward_terms["payload_position_tracking"].func(env, std=0.5)
    far_fine = reward_terms["payload_position_fine"].func(env, std=0.1)
    far = sum(term.weight * term.func(env, **term.params) for term in reward_terms.values() if term.weight > 0)
    # Tracking terms sum to 1.0 and a non-terminated live state adds 0.1.
    assert torch.allclose(near, torch.full_like(near, 1.10), atol=1e-4)
    assert (far_pos < 1e-12).all() and (far_fine < 1e-12).all()
    assert torch.allclose(far, torch.full_like(far, 0.60), atol=1e-4)


def test_reward_economics_make_every_bounded_live_step_nonnegative():
    env = _env(2)
    env.reset()
    for i in range(3):
        quad = env.scene[f"quad_{i}"]
        velocity = torch.zeros(env.num_envs, 6, device=env.device)
        velocity[:, 3:] = 1.0e6
        quad.write_root_link_velocity_to_sim(velocity)
    env.sim.forward()
    assert torch.equal(body_rate_penalty(env), torch.full((env.num_envs,), 25.0))

    # Worst values under clipped actions and bounded shaping.  Positive
    # tracking terms are omitted, making this a conservative lower bound.
    lower_bound = 0.10 - 0.002 * 25.0 - 0.002 * 4.0 - 0.01 * 0.5 - 0.004
    assert lower_bound > 0.0
    assert reward_terms["action_rate"].weight == 0.0
    assert reward_terms["body_rate_penalty"].weight == 0.0
    assert reward_terms["crash_penalty"].weight * 0.01 == -2.0


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


def test_reward_curriculum_holds_wide_then_tightens_after_1000_iterations():
    env = _env(2)
    env.reset()

    def snapshot() -> dict[str, float]:
        return {
            name: env.reward_manager.get_term_cfg(name).params["std"]
            for name in (
                "payload_position_tracking",
                "payload_position_fine",
                "payload_orientation_tracking",
                "payload_velocity_tracking",
                "payload_ang_vel_penalty",
            )
        }

    initial = {
        "payload_position_tracking": 0.75,
        "payload_position_fine": 0.20,
        "payload_orientation_tracking": 1.0,   # held constant: no attitude curriculum
        "payload_velocity_tracking": 1.0,
        "payload_ang_vel_penalty": 2.0,
    }
    assert snapshot() == initial

    # Curriculum does NOT start until 1000 iterations: at 500 nothing has moved.
    env.common_step_counter = 500 * 256
    env.curriculum_manager.compute()
    assert snapshot() == initial
    assert env.reward_manager.get_term_cfg("action_rate").weight == 0.0
    assert env.reward_manager.get_term_cfg("body_rate_penalty").weight == 0.0

    # First tightening at 1000. Orientation stays wide at 1.0.
    env.common_step_counter = 1000 * 256
    env.curriculum_manager.compute()
    assert snapshot() == {
        "payload_position_tracking": 0.60,
        "payload_position_fine": 0.15,
        "payload_orientation_tracking": 1.0,
        "payload_velocity_tracking": 0.75,
        "payload_ang_vel_penalty": 1.50,
    }
    assert env.reward_manager.get_term_cfg("action_rate").weight == -0.001
    assert env.reward_manager.get_term_cfg("body_rate_penalty").weight == -0.001

    # Final tightening at 1500, held for the last 500 of the 2000-iter run.
    env.common_step_counter = 1500 * 256
    env.curriculum_manager.compute()
    assert snapshot() == {
        "payload_position_tracking": 0.45,
        "payload_position_fine": 0.10,
        "payload_orientation_tracking": 1.0,
        "payload_velocity_tracking": 0.50,
        "payload_ang_vel_penalty": 1.00,
    }
    assert env.reward_manager.get_term_cfg("action_rate").weight == -0.002
    assert env.reward_manager.get_term_cfg("body_rate_penalty").weight == -0.002


def test_sweep_parameters_are_direct_config_overrides():
    env_cfg = make_payload_env_cfg(num_envs=2)
    command_cfg = env_cfg.commands["payload_pose"]
    assert env_cfg.rewards["payload_position_tracking"].params["std"] == 0.75
    assert env_cfg.rewards["payload_orientation_tracking"].params["std"] == 1.0
    # Attitude is held wide with no tightening curriculum (weak cable authority).
    assert "payload_orientation_tracking_std" not in env_cfg.curriculum
    assert command_cfg.hover_fraction_initial == 0.40
    assert command_cfg.hover_fraction_final == 0.15

    rl_cfg = payload_rma_phase1_cfg()
    assert rl_cfg.max_iterations == 2000
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


def test_12_conditional_payload_sampling_is_margin_feasible():
    """The corrected loaded-TWR sampling must keep every team controllable."""
    env = _env(4096)
    env.reset(seed=1729)
    team = env.team_state
    total_thrust = sum(agent.max_thrust_per_motor * 4.0 for agent in team.agents)
    total_quad_mass = sum(agent.mass for agent in team.agents)
    assert torch.all(total_thrust > 1.5 * (total_quad_mass + team.payload_mass) * 9.81)
    assert torch.all(team.payload_mass > 0)
    assert torch.allclose(team.margin, total_thrust / ((total_quad_mass + team.payload_mass) * 9.81))


def test_13_payload_trajectory_speed_cap_is_respected():
    """The generated reference, not vehicle motion, must honor the speed cap."""
    env = _env(512)
    env.reset(seed=2718)
    command = env.command_manager.get_term("payload_pose")
    peak_speed = command.velocity_ref.norm(dim=-1).amax()
    assert peak_speed <= 1.05 * command.cfg.max_payload_speed


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
