"""Acceptance tests for folded payload-transport RMA models."""

from __future__ import annotations

from unittest import mock

import pytest
import torch
import warp as wp
from tensordict import TensorDict

wp.config.kernel_cache_dir = "/tmp/payload-drones-warp-cache"

from mjlab.envs import ManagerBasedRlEnv
from rsl_rl.storage import RolloutStorage

from payload_drones.rl.modules import TrajectoryEncoder
from payload_drones.mdp.observations import HISTORY_LENGTH, POLICY_DIM
from payload_drones.payload_env_cfg import make_payload_env_cfg
from payload_drones.rl.algorithms import TeamLatentDistillation
from payload_drones.rl.models import (
    ACTION_DIM_PER_AGENT,
    DYNAMICS_DIM,
    N_AGENTS,
    MARLStudentModel,
    MARLTeacherModel,
    fold_agents,
    unfold_agents,
)


def _groups() -> dict[str, tuple[str, ...]]:
    return {
        "actor": ("policy", "payload_trajectory", "agent_dynamics"),
        "critic": ("critic",),
        "student": ("policy", "payload_trajectory", "agent_history"),
        "teacher": ("policy", "payload_trajectory", "agent_dynamics"),
    }


def _dist() -> dict[str, object]:
    return {
        "class_name": "payload_drones.rl.models.SharedAgentGaussianDistribution",
        "init_std": 0.3,
        "std_type": "scalar",
    }


def _teacher(obs: TensorDict) -> MARLTeacherModel:
    return MARLTeacherModel(obs, _groups(), "actor", 12, distribution_cfg=_dist())


def _student(obs: TensorDict) -> MARLStudentModel:
    return MARLStudentModel(obs, _groups(), "student", 12, distribution_cfg=_dist())


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
    baseline_latent = student.get_adaptation_latent(env_obs)

    changed_history = _clone_obs(env_obs)
    per_agent_history = HISTORY_LENGTH * (POLICY_DIM + ACTION_DIM_PER_AGENT)
    changed_history["agent_history"][:, per_agent_history : 2 * per_agent_history] += 2.0
    changed_action = student(changed_history)
    changed_latent = student.get_adaptation_latent(changed_history)
    assert torch.equal(baseline_action.view(-1, 3, 4)[:, 0], changed_action.view(-1, 3, 4)[:, 0])
    assert torch.equal(baseline_action.view(-1, 3, 4)[:, 2], changed_action.view(-1, 3, 4)[:, 2])
    assert torch.equal(baseline_latent.view(-1, 3, 8)[:, 0], changed_latent.view(-1, 3, 8)[:, 0])
    assert torch.equal(baseline_latent.view(-1, 3, 8)[:, 2], changed_latent.view(-1, 3, 8)[:, 2])

    changed_dynamics = _clone_obs(env_obs)
    changed_dynamics["agent_dynamics"][:, DYNAMICS_DIM : 2 * DYNAMICS_DIM] += 2.0
    # Student is entirely blind to privileged dynamics; its adaptation remains unchanged.
    assert torch.equal(baseline_action, student(changed_dynamics))
    assert torch.equal(baseline_latent, student.get_adaptation_latent(changed_dynamics))
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


def test_6_distillation_uses_ego_paired_rows(env_obs: TensorDict):
    # Distinct known ego rows make a mistaken (N, B, D) target transpose fail
    # deterministically, independent of random neural-network initialization.
    predicted = torch.arange(4 * N_AGENTS * 8, dtype=torch.float32).view(-1, 8)
    target = predicted.clone()
    assert TeamLatentDistillation.ego_paired_loss(predicted, target) == 0
    permuted = target.view(-1, N_AGENTS, 8)[:, [1, 0, 2]].reshape(-1, 8)
    assert TeamLatentDistillation.ego_paired_loss(predicted, permuted) > 0


def test_7_exploration_std_is_tied_after_optimizer_step(env_obs: TensorDict):
    actor = _teacher(env_obs)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    actions = actor(env_obs, stochastic_output=True)
    std_before = actor.output_std.detach().view(-1, N_AGENTS, ACTION_DIM_PER_AGENT)
    assert actor.distribution.std_param.shape == (4,)
    assert torch.equal(std_before[:, 0], std_before[:, 1])
    assert torch.equal(std_before[:, 1], std_before[:, 2])
    loss = -actor.get_output_log_prob(actions).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    actor(env_obs, stochastic_output=True)
    std_after = actor.output_std.detach().view(-1, N_AGENTS, ACTION_DIM_PER_AGENT)
    assert torch.equal(std_after[:, 0], std_after[:, 1])
    assert torch.equal(std_after[:, 1], std_after[:, 2])


def test_8_phase2_load_freezes_everything_except_tcn(env_obs: TensorDict):
    teacher = _teacher(env_obs)
    student = _student(env_obs)
    storage = RolloutStorage("distillation", 4, 1, env_obs, [12], "cpu")
    algorithm = TeamLatentDistillation(student, teacher, storage, learning_rate=1e-3)
    algorithm.load({"actor_state_dict": teacher.state_dict()}, load_cfg=None, strict=True)
    assert all(not p.requires_grad for p in student.trajectory_encoder.parameters())
    assert all(not p.requires_grad for p in student.mlp.parameters())
    assert all(p.requires_grad for p in student.adaptation_module.parameters())

    before = {name: p.detach().clone() for name, p in student.named_parameters()}
    loss = student.get_adaptation_latent(env_obs).square().mean()
    algorithm.optimizer.zero_grad()
    loss.backward()
    algorithm.optimizer.step()
    for name, parameter in student.named_parameters():
        changed = not torch.equal(before[name], parameter.detach())
        assert changed == name.startswith("adaptation_module.")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable in this execution environment")
def test_9_gpu_8192_forward_and_backward_without_oom():
    device = torch.device("cuda")
    batch = 8192
    obs = TensorDict({
        "policy": torch.randn(batch, 114, device=device),
        "payload_trajectory": torch.randn(batch, 1200, device=device),
        "agent_dynamics": torch.randn(batch, 45, device=device),
        "agent_history": torch.randn(batch, 6300, device=device),
        "critic": torch.randn(batch, 170, device=device),
    }, batch_size=[batch], device=device)
    teacher, student = _teacher(obs).to(device), _student(obs).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    (teacher(obs).square().mean() + student(obs).square().mean()).backward()
    torch.cuda.synchronize(device)
    assert torch.cuda.max_memory_allocated(device) > 0
