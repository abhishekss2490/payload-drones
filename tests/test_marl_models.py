"""Acceptance tests for folded payload-transport RMA models."""

from __future__ import annotations

from unittest import mock

import pytest
import torch
import warp as wp
from tensordict import TensorDict

wp.config.kernel_cache_dir = "/tmp/payload-drones-warp-cache"

from mjlab.envs import ManagerBasedRlEnv
from robotics_common.learning import DynamicsDistillation
from rsl_rl.storage import RolloutStorage

from payload_drones.mdp.observations import HISTORY_LENGTH, POLICY_DIM
from payload_drones.payload_env_cfg import make_payload_env_cfg
from payload_drones.rl.models import (
    ACTION_DIM_PER_AGENT,
    DYNAMICS_DIM,
    N_AGENTS,
    MARLStudentModel,
    MARLTeacherModel,
    SharedAgentGaussianDistribution,
    fold_agents,
    unfold_agents,
)


def _groups() -> dict[str, tuple[str, ...]]:
    return {
        "actor": ("policy", "payload_trajectory", "agent_dynamics"),
        "critic": ("critic", "payload_trajectory"),
        "student": ("policy", "payload_trajectory", "agent_history"),
        "teacher": ("policy", "payload_trajectory", "agent_dynamics"),
    }


def _dist(std_type: str = "log") -> dict[str, object]:
    return {
        "class_name": "payload_drones.rl.models.SharedAgentGaussianDistribution",
        "init_std": 0.15,
        "std_type": std_type,
    }


def _teacher(obs: TensorDict, std_type: str = "log") -> MARLTeacherModel:
    return MARLTeacherModel(obs, _groups(), "actor", 12, distribution_cfg=_dist(std_type))


def _student(obs: TensorDict, std_type: str = "log") -> MARLStudentModel:
    return MARLStudentModel(obs, _groups(), "student", 12, distribution_cfg=_dist(std_type))


@pytest.fixture(scope="module")
def env_obs() -> TensorDict:
    env = ManagerBasedRlEnv(make_payload_env_cfg(num_envs=4, include_history=True), device="cpu")
    obs, _ = env.reset(seed=0)
    return obs


def _clone_obs(obs: TensorDict | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in obs.items()}


def test_1_teacher_student_and_critic_shapes(env_obs: TensorDict):
    teacher = _teacher(env_obs)
    student = _student(env_obs)
    critic = MARLTeacherModel(env_obs, _groups(), "critic", 1)
    critic.bind_shared_trajectory_encoder(teacher.trajectory_encoder)
    assert tuple(teacher(env_obs).shape) == (4, 12)
    assert tuple(student(env_obs).shape) == (4, 12)
    assert tuple(critic(env_obs).shape) == (4, 1)


def test_2_trajectory_encoder_runs_once_and_broadcasts(env_obs: TensorDict):
    teacher = _teacher(env_obs)
    with mock.patch.object(
        teacher.trajectory_encoder,
        "forward",
        wraps=teacher.trajectory_encoder.forward,
    ) as forward:
        latent = teacher.get_latent(env_obs)
    assert forward.call_count == 1
    z_traj = latent.view(-1, N_AGENTS, 62)[:, :, POLICY_DIM : POLICY_DIM + 16]
    assert torch.equal(z_traj[:, 0], z_traj[:, 1])
    assert torch.equal(z_traj[:, 1], z_traj[:, 2])


def test_3_agent_ordering_round_trips_and_isolates_actions(env_obs: TensorDict):
    known = torch.arange(4 * N_AGENTS * 2, dtype=torch.float32).view(4, -1)
    assert torch.equal(unfold_agents(fold_agents(known, 2), 2), known)

    teacher = _teacher(env_obs)
    baseline = teacher(env_obs)
    changed = _clone_obs(env_obs)
    changed["policy"][:, POLICY_DIM : 2 * POLICY_DIM] += 1.0
    after = teacher(changed)
    baseline_blocks = baseline.view(-1, N_AGENTS, ACTION_DIM_PER_AGENT)
    after_blocks = after.view(-1, N_AGENTS, ACTION_DIM_PER_AGENT)
    assert torch.equal(baseline_blocks[:, 0], after_blocks[:, 0])
    assert not torch.equal(baseline_blocks[:, 1], after_blocks[:, 1])
    assert torch.equal(baseline_blocks[:, 2], after_blocks[:, 2])


def test_4_ego_history_and_dynamics_are_isolated(env_obs: TensorDict):
    teacher = _teacher(env_obs)
    student = _student(env_obs)
    baseline_action = student(env_obs)
    baseline_latent = student.estimated_dynamics_latent(env_obs)

    changed_history = _clone_obs(env_obs)
    per_agent_history = HISTORY_LENGTH * (POLICY_DIM + ACTION_DIM_PER_AGENT)
    changed_history["agent_history"][:, per_agent_history : 2 * per_agent_history] += 2.0
    changed_action = student(changed_history)
    changed_latent = student.estimated_dynamics_latent(changed_history)
    assert torch.equal(baseline_action.view(-1, 3, 4)[:, 0], changed_action.view(-1, 3, 4)[:, 0])
    assert torch.equal(baseline_action.view(-1, 3, 4)[:, 2], changed_action.view(-1, 3, 4)[:, 2])
    assert torch.equal(baseline_latent.view(-1, 3, 8)[:, 0], changed_latent.view(-1, 3, 8)[:, 0])
    assert torch.equal(baseline_latent.view(-1, 3, 8)[:, 2], changed_latent.view(-1, 3, 8)[:, 2])

    changed_dynamics = _clone_obs(env_obs)
    changed_dynamics["agent_dynamics"][:, DYNAMICS_DIM : 2 * DYNAMICS_DIM] += 2.0
    # Student is entirely blind to privileged dynamics; its adaptation remains unchanged.
    assert torch.equal(baseline_action, student(changed_dynamics))
    assert torch.equal(baseline_latent, student.estimated_dynamics_latent(changed_dynamics))
    teacher_before = teacher(env_obs).view(-1, 3, 4)
    teacher_after = teacher(changed_dynamics).view(-1, 3, 4)
    assert torch.equal(teacher_before[:, 0], teacher_after[:, 0])
    assert torch.equal(teacher_before[:, 2], teacher_after[:, 2])


def test_5_actor_has_no_centralized_privileged_leak(env_obs: TensorDict):
    teacher = _teacher(env_obs)
    baseline = teacher(env_obs)
    changed = _clone_obs(env_obs)
    # critic = policy(114), all ego dynamics(45), then mass/inertia/cables/mask/margin.
    changed["critic"][:, 114 + DYNAMICS_DIM : 114 + 2 * DYNAMICS_DIM] += 3.0
    changed["critic"][:, 159:166] += 7.0  # payload mass/inertia and cable lengths
    assert torch.equal(baseline, teacher(changed))


def test_6_distillation_targets_are_env_major_ego_rows(env_obs: TensorDict):
    # A mistaken (N, B, D) transpose must fail deterministically, independent of
    # random initialization: env-major agent order is the pairing contract.
    student = _student(env_obs)
    teacher = _teacher(env_obs)
    prediction = student.estimated_dynamics_latent(env_obs)
    target = teacher.dynamics_latent(env_obs)

    assert prediction.shape == (next(iter(env_obs.values())).shape[0], N_AGENTS, 8)
    assert target.shape == prediction.shape
    assert torch.equal(prediction.reshape(-1, 8), student._agent_estimated_latent(env_obs))
    assert torch.equal(target.reshape(-1, 8), teacher._agent_dynamics_latent(env_obs))


@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_7_exploration_std_is_tied_after_optimizer_step(env_obs: TensorDict, std_type: str):
    actor = _teacher(env_obs, std_type)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    actions = actor(env_obs, stochastic_output=True)
    std_before = actor.output_std.detach().view(-1, N_AGENTS, ACTION_DIM_PER_AGENT)
    name = "std_param" if std_type == "scalar" else "log_std_param"
    parameter = getattr(actor.distribution, name)
    assert parameter.shape == (4,)
    torch.testing.assert_close(std_before, torch.full_like(std_before, 0.15))
    parameter_before = parameter.detach().clone()
    assert torch.equal(std_before[:, 0], std_before[:, 1])
    assert torch.equal(std_before[:, 1], std_before[:, 2])
    loss = -actor.get_output_log_prob(actions).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    actor(env_obs, stochastic_output=True)
    std_after = actor.output_std.detach().view(-1, N_AGENTS, ACTION_DIM_PER_AGENT)
    assert torch.isfinite(parameter.grad).all()
    assert not torch.equal(parameter_before, parameter.detach())
    assert (std_after > 0).all()
    assert torch.equal(std_after[:, 0], std_after[:, 1])
    assert torch.equal(std_after[:, 1], std_after[:, 2])


@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_8_phase2_load_freezes_everything_except_tcn(env_obs: TensorDict, std_type: str):
    teacher = _teacher(env_obs, std_type)
    student = _student(env_obs, std_type)
    name = "std_param" if std_type == "scalar" else "log_std_param"
    with torch.no_grad():
        values = torch.tensor([0.1, 0.15, 0.2, 0.25])
        getattr(teacher.distribution, name).copy_(values.log() if std_type == "log" else values)
    storage = RolloutStorage("distillation", 4, 1, env_obs, [12], "cpu")
    algorithm = DynamicsDistillation(student, teacher, storage, learning_rate=1e-3)
    algorithm.load({"actor_state_dict": teacher.state_dict()}, load_cfg=None, strict=True)
    torch.testing.assert_close(
        getattr(student.distribution, name), getattr(teacher.distribution, name), rtol=0, atol=0
    )
    assert not getattr(student.distribution, name).requires_grad
    assert all(not p.requires_grad for p in student.trajectory_encoder.parameters())
    assert all(not p.requires_grad for p in student.mlp.parameters())
    assert all(p.requires_grad for p in student.adaptation_module.parameters())

    before = {name: p.detach().clone() for name, p in student.named_parameters()}
    loss = student.estimated_dynamics_latent(env_obs).square().mean()
    algorithm.optimizer.zero_grad()
    loss.backward()
    algorithm.optimizer.step()
    for name, parameter in student.named_parameters():
        changed = not torch.equal(before[name], parameter.detach())
        assert changed == name.startswith("adaptation_module.")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable in this execution environment"
)
def test_9_gpu_8192_forward_and_backward_without_oom():
    device = torch.device("cuda")
    batch = 8192
    obs = TensorDict(
        {
            "policy": torch.randn(batch, 114, device=device),
            "payload_trajectory": torch.randn(batch, 1200, device=device),
            "agent_dynamics": torch.randn(batch, 45, device=device),
            "agent_history": torch.randn(batch, 6300, device=device),
            "critic": torch.randn(batch, 170, device=device),
        },
        batch_size=[batch],
        device=device,
    )
    teacher, student = _teacher(obs).to(device), _student(obs).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    (teacher(obs).square().mean() + student(obs).square().mean()).backward()
    torch.cuda.synchronize(device)
    assert torch.cuda.max_memory_allocated(device) > 0


@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_10_distribution_checkpoint_round_trip(tmp_path, std_type: str):
    distribution = SharedAgentGaussianDistribution(12, init_std=0.15, std_type=std_type)
    name = "std_param" if std_type == "scalar" else "log_std_param"
    values = torch.tensor([0.02, 0.1, 0.4, 0.8])
    with torch.no_grad():
        getattr(distribution, name).copy_(values.log() if std_type == "log" else values)
    mean = torch.zeros(2, 12)
    distribution.update(mean)
    torch.testing.assert_close(distribution.std, values.repeat(3).expand_as(mean))
    assert set(distribution.state_dict()) == {name}
    checkpoint = tmp_path / "distribution.pt"
    torch.save(distribution.state_dict(), checkpoint)
    restored = SharedAgentGaussianDistribution(12, init_std=0.15, std_type=std_type)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    restored.update(mean)
    torch.testing.assert_close(restored.std, distribution.std, rtol=0, atol=0)
    torch.testing.assert_close(restored.entropy, distribution.entropy, rtol=0, atol=0)


@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_11_fixed_std_remains_nonlearnable(std_type: str):
    distribution = SharedAgentGaussianDistribution(
        12, init_std=0.15, std_type=std_type, learn_std=False
    )
    assert all(not p.requires_grad for p in distribution.parameters())
