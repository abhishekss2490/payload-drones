"""Speed-curriculum checks, runnable individually without a training run."""

import math
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import warp as wp

wp.config.kernel_cache_dir = "/tmp/payload-drones-warp-cache"

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

from payload_drones.config.rl_config import payload_rma_phase1_cfg
from payload_drones.mdp.payload_command import (
    PayloadTrajectoryCommand,
    PayloadTrajectoryCommandCfg,
)
from payload_drones.payload_env_cfg import make_payload_env_cfg
from payload_drones.rl.runner import MARLPhase1Runner


def _schedule_command(**overrides):
    command = object.__new__(PayloadTrajectoryCommand)
    command.cfg = PayloadTrajectoryCommandCfg(**overrides)
    command._env = SimpleNamespace(common_step_counter=0)
    return command


def _circular_env_cfg(num_envs=4):
    cfg = make_payload_env_cfg(num_envs=num_envs, include_team_probe=False)
    command = cfg.commands["payload_pose"]
    command.amplitude_range_xy = (0.75, 0.75)
    command.amplitude_range_z = (0.1, 0.1)
    command.freq_list = [1]
    command.delta_range = (math.pi / 2, math.pi / 2)
    return cfg


def test_01_speed_schedule_endpoints_midpoint_and_disabled():
    command = _schedule_command()
    assert command.cfg.steps_per_iteration == payload_rma_phase1_cfg().num_steps_per_env
    for iteration, expected in [
        (0, 0.5),
        (1, 0.5013333333333333),
        (375, 1.0),
        (749, 1.4986666666666666),
        (750, 1.5),
        (2000, 1.5),
    ]:
        command._env.common_step_counter = iteration * 256
        assert command._current_speed_cap() == pytest.approx(expected)
    command._env.common_step_counter = 96_128  # Halfway plus half an iteration.
    assert command._current_speed_cap() == pytest.approx(1.0006666666666666)
    command.cfg.speed_anneal_iterations = 0
    command._env.common_step_counter = 0
    assert command._current_speed_cap() == 1.5


def test_02_training_play_and_phase2_configs():
    phase1 = load_env_cfg("Mjlab-PayloadTransport-RMA-Phase1")
    command = phase1.commands["payload_pose"]
    assert (command.initial_payload_speed, command.max_payload_speed) == (0.5, 1.5)
    assert command.speed_anneal_iterations == 750
    assert (command.rho_min, command.rho_max) == (0.25, 0.25)
    assert (command.hover_fraction_initial, command.hover_fraction_final) == (0.0, 0.0)
    assert phase1.curriculum == {}
    rewards = {name: (term.weight, term.params) for name, term in phase1.rewards.items()}
    assert "payload_position_far" not in rewards
    for task, play in [
        ("Mjlab-PayloadTransport-RMA-Phase1", True),
        ("Mjlab-PayloadTransport-RMA-Phase2", False),
        ("Mjlab-PayloadTransport-RMA-Phase2", True),
    ]:
        cfg = load_env_cfg(task, play=play)
        term = cfg.commands["payload_pose"]
        assert term.speed_anneal_iterations == 0
        assert term.max_payload_speed == 1.5
        assert (term.rho_min, term.rho_max) == (0.25, 0.25)
        assert {name: (r.weight, r.params) for name, r in cfg.rewards.items()} == rewards
    overridden = make_payload_env_cfg(
        num_envs=1,
        initial_payload_speed=0.4,
        max_payload_speed=1.2,
        speed_anneal_iterations=600,
    ).commands["payload_pose"]
    assert (overridden.initial_payload_speed, overridden.max_payload_speed) == (0.4, 1.2)
    assert overridden.speed_anneal_iterations == 600


def test_03_generated_references_follow_ramp_without_changing_rho_or_rewards():
    env = ManagerBasedRlEnv(_circular_env_cfg(), device="cpu")
    try:
        command = env.command_manager.get_term("payload_pose")
        initial_rewards = {
            name: (term.weight, dict(term.params)) for name, term in env.cfg.rewards.items()
        }
        for iteration, cap, lower_bound in [
            (0, 0.5, 0.45),
            (375, 1.0, 0.9),
            (750, 1.5, 1.1),
            (1000, 1.5, 1.1),
        ]:
            env.common_step_counter = iteration * 256
            obs, _ = env.reset(seed=4551)
            peak = command.velocity_ref.norm(dim=-1).amax(dim=1)
            assert torch.all(peak <= cap + 1e-3), (iteration, peak)
            assert torch.all(peak > lower_bound), (iteration, peak)
            assert all(torch.isfinite(value).all() for value in obs.values())
            assert torch.all(command.rho_actual == 0.25)
            assert torch.all(command.bin_rho_ceiling == 0.25)
            assert not command.is_hover_episode.any()
            assert env.common_step_counter == iteration * 256
            assert torch.allclose(command.metrics["speed_cap"], torch.full_like(peak, cap))
            assert {
                name: (term.weight, dict(term.params)) for name, term in env.cfg.rewards.items()
            } == initial_rewards

        # Changing the global cap does not rewrite another team's active reference.
        env.common_step_counter = 0
        env.reset(seed=4551)
        old_position = command.position_ref[1:].clone()
        old_velocity = command.velocity_ref[1:].clone()
        env.common_step_counter = 375 * 256
        env.reset(env_ids=torch.tensor([0]))
        torch.testing.assert_close(command.position_ref[1:], old_position, rtol=0, atol=0)
        torch.testing.assert_close(command.velocity_ref[1:], old_velocity, rtol=0, atol=0)
        assert command.velocity_ref[0].norm(dim=-1).max() > 0.9
    finally:
        env.close()


def test_04_runner_rollout_override_and_checkpoint_resume(tmp_path):
    env = ManagerBasedRlEnv(_circular_env_cfg(num_envs=2), device="cpu")
    try:
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        cfg = asdict(payload_rma_phase1_cfg())
        cfg.update(num_steps_per_env=8, upload_model=False)
        cfg["algorithm"].update(num_mini_batches=1, num_learning_epochs=1)
        runner = MARLPhase1Runner(wrapper, deepcopy(cfg), log_dir=None, device="cpu")
        command = env.command_manager.get_term("payload_pose")
        assert command.cfg.steps_per_iteration == 8
        env.common_step_counter = 375 * 8
        env.reset(seed=4551)
        assert command._current_speed_cap() == 1.0
        runner.current_learning_iteration = 374
        checkpoint = tmp_path / "speed_midpoint.pt"
        runner.save(str(checkpoint))

        # Start again at 0.5; load must regenerate the initial buffers at 1.0.
        env.common_step_counter = 0
        env.reset(seed=4551)
        assert command.velocity_ref.norm(dim=-1).max() <= 0.501
        restored = MARLPhase1Runner(wrapper, deepcopy(cfg), log_dir=None, device="cpu")
        restored.load(str(checkpoint), map_location="cpu")
        assert restored.current_learning_iteration == 375
        assert env.common_step_counter == 375 * 8
        assert command._current_speed_cap() == 1.0
        peak = command.velocity_ref.norm(dim=-1).amax(dim=1)
        assert torch.all(peak <= 1.001)
        assert torch.all(peak > 0.9)
        with torch.inference_mode():
            wrapper.step(torch.zeros(2, 12))
        assert env.common_step_counter == 375 * 8 + 1
        assert command._current_speed_cap() == pytest.approx(1.0001666666666666)
        env.common_step_counter = 750 * 8
        assert command._current_speed_cap() == 1.5
    finally:
        env.close()


def test_05_invalid_speed_configuration_is_rejected():
    env = SimpleNamespace(num_envs=1, device="cpu")
    for overrides, message in [
        ({"max_payload_speed": 0.0}, "max_payload_speed"),
        ({"max_payload_speed": float("nan")}, "max_payload_speed"),
        ({"initial_payload_speed": 0.0}, "initial_payload_speed"),
        ({"initial_payload_speed": float("inf")}, "initial_payload_speed"),
        ({"initial_payload_speed": 2.0}, "initial_payload_speed"),
        ({"speed_anneal_iterations": -1}, "speed_anneal_iterations"),
        ({"steps_per_iteration": 0}, "steps_per_iteration"),
    ]:
        with pytest.raises(ValueError, match=message):
            PayloadTrajectoryCommand(PayloadTrajectoryCommandCfg(**overrides), env)
