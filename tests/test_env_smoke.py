"""Acceptance tests from env_spec section 9."""

from __future__ import annotations

import pytest
import torch
import warp as wp

# The sandbox's home directory is read-only.  This does not affect simulation;
# it simply gives Warp a writable kernel cache for the test process.
wp.config.kernel_cache_dir = "/tmp/payload-drones-warp-cache"

from mjlab.envs import ManagerBasedRlEnv
from payload_drones.mdp.observations import agent_dynamics, payload_trajectory, policy
from payload_drones.mdp.rewards import reward_terms
from payload_drones.mdp.team_scaling import (
    CABLE_DAMPING_SAFETY,
    CABLE_STIFFNESS_SAFETY,
    MAX_CABLE_DAMPING,
    MAX_CABLE_STIFFNESS,
)
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

    min_mass = torch.stack([state.mass for state in env.team_state.agents], dim=1).amin(dim=1)
    expected_k = torch.minimum(
        torch.full_like(min_mass, MAX_CABLE_STIFFNESS),
        CABLE_STIFFNESS_SAFETY * min_mass / env.physics_dt**2,
    )
    expected_c = torch.minimum(
        torch.full_like(min_mass, MAX_CABLE_DAMPING),
        CABLE_DAMPING_SAFETY * min_mass / env.physics_dt,
    )
    for cable in range(3):
        tendon = spec.tendon(f"cable_{cable}").id
        assert torch.allclose(model.tendon_lengthspring[:, tendon, 1], env.team_state.cable_length[:, cable])
        assert torch.allclose(model.tendon_range[:, tendon, 1], env.team_state.cable_length[:, cable])
        assert torch.allclose(model.tendon_stiffness[:, tendon], expected_k)
        assert torch.allclose(model.tendon_damping[:, tendon], expected_c)
        for field, value in {
            f"cable_{cable}.tendon_lengthspring": model.tendon_lengthspring[:, tendon],
            f"cable_{cable}.tendon_range": model.tendon_range[:, tendon],
            f"cable_{cable}.tendon_stiffness": model.tendon_stiffness[:, tendon],
            f"cable_{cable}.tendon_damping": model.tendon_damping[:, tendon],
        }.items():
            has_cross_env_variation(value, field)


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


def test_6_reward_sanity_exposes_spec_inconsistency():
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
    assert torch.allclose(near, torch.ones_like(near), atol=1e-4)
    assert (far_pos < 1e-12).all() and (far_fine < 1e-12).all()
    assert torch.allclose(far, torch.full_like(far, 0.50), atol=1e-4)


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
    peak_speed = command.velocity_ref.norm(dim=-1).max()
    assert peak_speed > 0.1, f"trajectory buffer is essentially static: peak {peak_speed:.2e} m/s"

    # Read each trajectory at its own non-zero reference-speed sample without
    # relying on an arbitrary common episode time (short paths may finish
    # before the midpoint).
    command.episode_step[:] = command.velocity_ref.norm(dim=-1).argmax(dim=1) * command.cfg.trajectory_decimation
    speed = command.command[:, 7:10].norm(dim=-1)
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
