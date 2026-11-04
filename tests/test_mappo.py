"""Sequential correctness checks for the payload per-drone PPO implementation."""

import math

import pytest
import torch

from payload_drones.rl.algorithms import MAPPO


def test_01_four_motor_probability_grouping():
    mean = torch.zeros(1, 12, dtype=torch.float64)
    std = torch.ones_like(mean)
    actions = torch.zeros_like(mean)
    before = MAPPO._per_agent_log_prob(mean, std, actions)
    expected = torch.full((1, 3), -2 * math.log(2 * math.pi), dtype=torch.float64)
    torch.testing.assert_close(before, expected, rtol=0, atol=1e-12)
    actions[0, 4] = 1.0
    after = MAPPO._per_agent_log_prob(mean, std, actions)
    torch.testing.assert_close(
        after - before,
        torch.tensor([[0.0, -0.5, 0.0]], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )


from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest import mock

from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from payload_drones.config.rl_config import payload_rma_phase1_cfg
from payload_drones.rl.models import MARLTeacherModel


def _observations(envs=4, step=0, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(4551 + step)
    return TensorDict(
        {
            "policy": torch.randn(envs, 114, generator=generator, device=device),
            "payload_trajectory": torch.randn(envs, 1200, generator=generator, device=device),
            "agent_dynamics": torch.randn(envs, 45, generator=generator, device=device),
            "critic": torch.randn(envs, 170, generator=generator, device=device),
            "sample_id": (torch.arange(envs, device=device) + step * envs).unsqueeze(1).float(),
        },
        batch_size=[envs],
        device=device,
    )


def _components(envs=4, steps=4):
    torch.manual_seed(4551)
    obs = _observations(envs)
    groups = payload_rma_phase1_cfg().obs_groups
    distribution = {
        "class_name": "payload_drones.rl.models.SharedAgentGaussianDistribution",
        "init_std": 0.15,
        "std_type": "log",
    }
    actor = MARLTeacherModel(obs, groups, "actor", 12, distribution_cfg=distribution)
    critic = MARLTeacherModel(obs, groups, "critic", 1)
    storage = RolloutStorage("rl", envs, steps, obs, [12], "cpu")
    return actor, critic, storage


def _algorithm(envs=4, steps=4, **overrides):
    options = {
        "num_learning_epochs": 2,
        "num_mini_batches": 2,
        "learning_rate": 1e-3,
        "gamma": 0.995,
        "lam": 0.95,
        "clip_param": 0.2,
        "desired_kl": 0.015,
        "entropy_coef": 0.0,
        "normalize_advantage_per_mini_batch": False,
    }
    options.update(overrides)
    return MAPPO(*_components(envs, steps), **options)


def _collect(alg):
    envs = alg.storage.num_envs
    steps = alg.storage.num_transitions_per_env
    with torch.inference_mode():
        for step in range(steps):
            alg.act(_observations(envs, step))
            rewards = torch.arange(envs).float() * 0.17 + step * 0.31
            dones = torch.zeros(envs, dtype=torch.long)
            timeouts = torch.zeros(envs)
            if step == 1:
                dones[0] = 1
                if envs > 1:
                    dones[1] = 1
                    timeouts[1] = 1
            alg.process_env_step(
                _observations(envs, step + 1), rewards, dones, {"time_outs": timeouts}
            )
        alg.compute_returns(_observations(envs, steps))
    return alg


def _unchanged_parameters(model, before):
    return all(torch.equal(p.detach(), before[name]) for name, p in model.named_parameters())


def _state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _state_equal(a, b)
    else:
        assert left == right


def test_02_stored_probabilities_and_minibatch_alignment():
    alg = _collect(_algorithm())
    st = alg.storage
    expected_actions = st.actions.flatten(0, 1).clone()
    expected_params = tuple(p.flatten(0, 1).clone() for p in st.distribution_params)
    seen = []
    for batch in st.mini_batch_generator(2, 1):
        ids = batch.observations["sample_id"].squeeze(-1).long()
        seen.extend(ids.tolist())
        torch.testing.assert_close(batch.actions, expected_actions[ids], rtol=0, atol=0)
        for actual, expected in zip(batch.old_distribution_params, expected_params):
            torch.testing.assert_close(actual, expected[ids], rtol=0, atol=0)
            assert not actual.requires_grad
        lp = MAPPO._per_agent_log_prob(*batch.old_distribution_params, batch.actions)
        assert lp.shape == (len(ids), 3)
        torch.testing.assert_close(
            lp.sum(-1), batch.old_actions_log_prob.squeeze(-1), rtol=0, atol=1e-5
        )
    assert sorted(seen) == list(range(16))


def test_03_initial_ratios():
    alg = _collect(_algorithm())
    for batch in alg.storage.mini_batch_generator(2, 1):
        alg.actor(batch.observations, stochastic_output=True)
        new = MAPPO._per_agent_log_prob(*alg.actor.output_distribution_params, batch.actions)
        with torch.no_grad():
            old = MAPPO._per_agent_log_prob(*batch.old_distribution_params, batch.actions)
        assert new.requires_grad
        assert not old.requires_grad
        torch.testing.assert_close(torch.exp(new - old), torch.ones_like(new), rtol=0, atol=1e-5)


def test_04_positive_advantage_clipping():
    ratio = torch.tensor([[1.5, 0.9, 0.5]], dtype=torch.float64)
    loss = MAPPO._surrogate(ratio, torch.tensor([[0.8]], dtype=torch.float64), 0.2)
    assert loss.item() == pytest.approx(-2.08 / 3, abs=1e-12)


def test_05_negative_advantage_clipping():
    ratio = torch.tensor([[1.5, 0.9, 0.5]], dtype=torch.float64)
    loss = MAPPO._surrogate(ratio, torch.tensor([[-0.8]], dtype=torch.float64), 0.2)
    assert loss.item() == pytest.approx(2.56 / 3, abs=1e-12)


def test_06_mean_aggregation_and_gradient_scale():
    ratio = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    adv = torch.tensor([[0.6], [-0.3]], dtype=torch.float64)
    loss = MAPPO._surrogate(ratio, adv, 0.2)
    assert loss.item() == pytest.approx(-0.15, abs=1e-12)
    loss.backward()
    torch.testing.assert_close(ratio.grad, (-adv / 6).expand(2, 3), rtol=0, atol=1e-12)


@pytest.mark.parametrize("batch_size", [1, 3, 64])
def test_07_broadcasting(batch_size):
    generator = torch.Generator().manual_seed(71)
    ratio = 0.3 + torch.rand(batch_size, 3, generator=generator, dtype=torch.float64)
    adv = torch.randn(batch_size, 1, generator=generator, dtype=torch.float64)
    expected = 0.0
    for team in range(batch_size):
        for drone in range(3):
            r, a = ratio[team, drone].item(), adv[team, 0].item()
            expected -= min(r * a, max(0.8, min(1.2, r)) * a)
    expected /= batch_size * 3
    assert MAPPO._surrogate(ratio, adv, 0.2).item() == pytest.approx(expected, abs=1e-12)


def test_08_clipping_gradient_selectivity():
    actions = torch.ones(1, 12, dtype=torch.float64)
    mean = torch.zeros_like(actions, requires_grad=True)
    new = MAPPO._per_agent_log_prob(mean, torch.ones_like(mean), actions)
    old = new.detach() - torch.log(torch.tensor([[1.5, 1.0, 1.0]], dtype=torch.float64))
    MAPPO._surrogate(torch.exp(new - old), torch.ones(1, 1, dtype=torch.float64), 0.2).backward()
    torch.testing.assert_close(mean.grad[:, :4], torch.zeros(1, 4, dtype=torch.float64))
    torch.testing.assert_close(
        mean.grad[:, 4:], torch.full((1, 8), -1 / 3, dtype=torch.float64), rtol=0, atol=1e-12
    )


def test_09_team_gae_timeouts_and_rollout_normalization():
    alg = _algorithm(envs=2, steps=3, num_mini_batches=1)
    values = torch.tensor([[[0.2], [0.4]], [[0.3], [0.5]], [[0.7], [0.8]]])
    last = torch.tensor([[0.9], [1.0]])
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    dones = torch.tensor([[0, 0], [1, 1], [0, 0]])
    timeouts = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    with mock.patch.object(alg.critic, "forward", side_effect=[*values, last]):  # noqa: SIM117
        with torch.inference_mode():
            for step in range(3):
                alg.act(_observations(2, step))
                alg.process_env_step(
                    _observations(2, step + 1),
                    rewards[step],
                    dones[step],
                    {"time_outs": timeouts[step]},
                )
            alg.compute_returns(_observations(2, 3))
    expected_returns = torch.tensor(
        [
            [[3.850675], [6.276136875]],
            [[3.0], [4.4975]],
            [[5.8955], [6.995]],
        ]
    )
    torch.testing.assert_close(alg.storage.returns, expected_returns, rtol=0, atol=2e-6)
    assert alg.storage.rewards[1, 0, 0].item() == 3.0
    assert alg.storage.rewards[1, 1, 0].item() == pytest.approx(4.4975, abs=1e-6)
    assert rewards[1, 1].item() == 4.0
    raw = [3.650675, 5.876136875, 2.7, 3.9975, 5.1955, 6.195]
    center = sum(raw) / len(raw)
    scale = math.sqrt(sum((x - center) ** 2 for x in raw) / (len(raw) - 1))
    expected = torch.tensor([(x - center) / (scale + 1e-8) for x in raw]).reshape(3, 2, 1)
    torch.testing.assert_close(alg.storage.advantages, expected, rtol=0, atol=2e-6)
    assert alg.storage.advantages.shape == (3, 2, 1)
    assert alg.normalize_advantage_per_mini_batch is False


def test_10_rollout_data_fixed_across_optimization():
    alg = _collect(_algorithm(num_learning_epochs=3))
    st = alg.storage
    fixed = {
        key: getattr(st, key).clone()
        for key in (
            "actions",
            "actions_log_prob",
            "advantages",
            "returns",
            "values",
        )
    }
    old_params = tuple(p.clone() for p in st.distribution_params)
    actor_before = {n: p.detach().clone() for n, p in alg.actor.named_parameters()}
    critic_before = {n: p.detach().clone() for n, p in alg.critic.named_parameters()}
    original_generator = st.mini_batch_generator
    original_step = alg.optimizer.step
    expected_advantages = []
    consumed = []
    steps = []

    def generator(*args):
        for batch in original_generator(*args):
            ids = batch.observations["sample_id"].squeeze(-1).long()
            expected_advantages.append(fixed["advantages"].flatten(0, 1)[ids])
            yield batch

    def surrogate(ratio, adv, clip):
        torch.testing.assert_close(adv, expected_advantages[len(consumed)], rtol=0, atol=0)
        consumed.append(True)
        return MAPPO._surrogate(ratio, adv, clip)

    def optimizer_step(*args, **kwargs):
        for key, expected in fixed.items():
            assert torch.equal(getattr(st, key), expected)
        for current, expected in zip(st.distribution_params, old_params):
            assert torch.equal(current, expected)
            assert not current.requires_grad
        steps.append(True)
        return original_step(*args, **kwargs)

    with (
        mock.patch.object(st, "mini_batch_generator", side_effect=generator),
        mock.patch.object(alg, "_surrogate", side_effect=surrogate),
        mock.patch.object(alg.optimizer, "step", side_effect=optimizer_step),
    ):
        metrics = alg.update()
    assert len(steps) == len(consumed) == 6
    assert not _unchanged_parameters(alg.actor, actor_before)
    assert not _unchanged_parameters(alg.critic, critic_before)
    assert all(math.isfinite(value) for value in metrics.values())
    assert st.step == 0
    for key, expected in fixed.items():
        assert torch.equal(getattr(st, key), expected)
    for current, expected in zip(st.distribution_params, old_params):
        assert torch.equal(current, expected)


def test_11_stored_samples_survive_environment_clipping():
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

    alg = _algorithm(envs=2, steps=2, num_mini_batches=1, num_learning_epochs=1)
    forced = torch.linspace(-1.8, 1.8, 24).reshape(2, 12)
    received = []
    wrapper = object.__new__(RslRlVecEnvWrapper)
    wrapper.clip_actions = 1.0
    wrapper.num_envs = 2

    def environment_step(actions):
        received.append(actions.clone())
        return (
            _observations(2),
            torch.tensor([0.2, 1.1]),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            {},
        )

    wrapper.env = SimpleNamespace(
        step=environment_step,
        unwrapped=SimpleNamespace(cfg=SimpleNamespace(is_finite_horizon=False)),
    )
    with (
        torch.inference_mode(),
        mock.patch.object(alg.actor.distribution, "sample", return_value=forced.reshape(6, 4)),
    ):
        for step in range(2):
            actions = alg.act(_observations(2, step))
            obs, rewards, dones, extras = wrapper.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
        alg.compute_returns(_observations(2, 2))
    assert (forced.abs() > 1).any()
    for sent in received:
        torch.testing.assert_close(sent, forced.clamp(-1, 1), rtol=0, atol=0)
    for stored in alg.storage.actions:
        torch.testing.assert_close(stored, forced, rtol=0, atol=0)
    visited = []

    def check_log_prob(mean, std, actions):
        alg.storage.observations["sample_id"].flatten().long()
        assert actions.shape == (4, 12)
        assert (actions.abs() > 1).any()
        assert all(any(torch.equal(row, expected) for expected in forced) for row in actions)
        visited.append(True)
        return MAPPO._per_agent_log_prob(mean, std, actions)

    with (
        mock.patch.object(alg, "_per_agent_log_prob", side_effect=check_log_prob),
        mock.patch.object(alg.optimizer, "step", return_value=None),
    ):
        alg.update()
    assert len(visited) == 2


def test_12_preserved_critic_entropy_and_joint_kl():
    alg = _collect(_algorithm())
    with torch.no_grad():
        alg.actor.distribution.log_std_param.add_(math.log(1.15))
        [m for m in alg.actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1].bias.add_(0.02)
        [m for m in alg.critic.mlp.modules() if isinstance(m, torch.nn.Linear)][-1].bias.add_(0.3)
    stock = PPO(
        deepcopy(alg.actor),
        deepcopy(alg.critic),
        deepcopy(alg.storage),
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-3,
        gamma=0.995,
        lam=0.95,
        clip_param=0.2,
        desired_kl=0.015,
        entropy_coef=0.0,
        normalize_advantage_per_mini_batch=False,
    )
    stock.critic.bind_shared_trajectory_encoder(stock.actor.trajectory_encoder)
    batch = next(alg.storage.mini_batch_generator(1, 1))
    alg.actor(batch.observations, stochastic_output=True)
    mu, sd = alg.actor.output_distribution_params
    old_mu, old_sd = batch.old_distribution_params
    drone_ratios = torch.exp(
        MAPPO._per_agent_log_prob(mu, sd, batch.actions)
        - MAPPO._per_agent_log_prob(old_mu, old_sd, batch.actions)
    )
    joint_ratio = torch.exp(
        alg.actor.get_output_log_prob(batch.actions) - batch.old_actions_log_prob.squeeze(-1)
    )
    torch.testing.assert_close(drone_ratios.prod(-1), joint_ratio, rtol=1e-5, atol=1e-5)
    expected_kl = (
        torch.log(sd / old_sd)
        + (old_sd.square() + (old_mu - mu).square()) / (2 * sd.square())
        - 0.5
    ).sum(-1)
    torch.testing.assert_close(
        alg.actor.get_kl_divergence((old_mu, old_sd), (mu, sd)),
        expected_kl,
        rtol=1e-5,
        atol=1e-6,
    )
    expected_entropy = (torch.log(sd) + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)
    torch.testing.assert_close(alg.actor.output_entropy, expected_entropy, rtol=1e-6, atol=1e-6)
    torch.manual_seed(123)
    with mock.patch.object(alg.optimizer, "step", return_value=None):
        actual = alg.update()
    torch.manual_seed(123)
    with mock.patch.object(stock.optimizer, "step", return_value=None):
        expected = stock.update()
    for key in ("value", "Policy/kl", "Critic/explained_variance", "Critic/grad_norm"):
        assert actual[key] == pytest.approx(expected[key], rel=0, abs=1e-6), key
    # Entropy is the per-drone mean: exactly one third of stock PPO's joint sum.
    assert actual["entropy"] == pytest.approx(expected["entropy"] / 3, rel=0, abs=1e-6)
    assert alg.learning_rate == stock.learning_rate
    for key in ("Policy/clip_fraction", "Policy/clip_fraction_joint"):
        assert 0.0 <= actual[key] <= 1.0


@pytest.mark.parametrize(
    "kl,initial_lr,expected_lr",
    [
        (0.0, 1e-3, 1e-3),
        (0.005, 1e-3, 1.5e-3),
        (0.02, 1e-3, 1e-3),
        (0.04, 1e-3, 1e-3 / 1.5),
        (0.005, 0.009, 0.01),
        (0.04, 1.1e-5, 1e-5),
        (0.0075, 1e-3, 1e-3),
        (0.03, 1e-3, 1e-3),
    ],
    ids=[
        "zero",
        "increase",
        "hold",
        "decrease",
        "upper-bound",
        "lower-bound",
        "lower-threshold",
        "upper-threshold",
    ],
)
def test_13_joint_kl_controller(kl, initial_lr, expected_lr):
    alg = _collect(_algorithm(num_learning_epochs=1, num_mini_batches=1, learning_rate=initial_lr))
    with (
        mock.patch.object(
            alg.actor, "get_kl_divergence", return_value=torch.full((16,), kl, dtype=torch.float64)
        ),
        mock.patch.object(alg.optimizer, "step", return_value=None),
    ):
        metrics = alg.update()
    assert metrics["Policy/kl"] == pytest.approx(kl, abs=1e-12)
    # One minibatch: the last-minibatch KL and the mean are the same number.
    assert metrics["Policy/kl_last_minibatch"] == pytest.approx(kl, abs=1e-12)
    assert alg.learning_rate == pytest.approx(expected_lr, abs=1e-12)
    assert all(group["lr"] == alg.learning_rate for group in alg.optimizer.param_groups)


def test_14_config_and_tensor_guards():
    from rsl_rl.utils import resolve_callable

    cfg = payload_rma_phase1_cfg()
    assert cfg.algorithm.class_name == "payload_drones.rl.algorithms.MAPPO"
    assert cfg.algorithm.normalize_advantage_per_mini_batch is False
    assert cfg.algorithm.entropy_coef == 0.002
    assert resolve_callable(cfg.algorithm.class_name) is MAPPO
    with pytest.raises(ValueError, match="actions"):
        MAPPO._per_agent_log_prob(torch.zeros(12), torch.ones(12), torch.zeros(12))
    with pytest.raises(ValueError, match="mean, std"):
        MAPPO._per_agent_log_prob(torch.zeros(2, 12), torch.ones(1, 12), torch.zeros(2, 12))
    with pytest.raises(ValueError, match="ratios"):
        MAPPO._surrogate(torch.ones(3), torch.ones(1, 1), 0.2)
    with pytest.raises(ValueError, match="advantages"):
        MAPPO._surrogate(torch.ones(3, 3), torch.ones(3), 0.2)


@pytest.mark.parametrize(
    "mode",
    [
        "actor-recurrent",
        "critic-recurrent",
        "rnd",
        "symmetry",
        "multi-gpu",
        "minibatch-normalization",
        "distribution",
        "storage",
    ],
)
def test_14_unsupported_modes(mode):
    from contextlib import ExitStack

    from rsl_rl.modules import GaussianDistribution

    actor, critic, storage = _components()
    kwargs = {}
    with ExitStack() as stack:
        if mode == "actor-recurrent":
            actor.is_recurrent = True
        elif mode == "critic-recurrent":
            critic.is_recurrent = True
        elif mode == "rnd":
            stack.enter_context(
                mock.patch("rsl_rl.algorithms.ppo.RandomNetworkDistillation", return_value=object())
            )
            kwargs["rnd_cfg"] = {"dummy": True}
        elif mode == "symmetry":
            stack.enter_context(mock.patch("rsl_rl.algorithms.ppo.Symmetry", return_value=object()))
            kwargs["symmetry_cfg"] = {"dummy": True}
        elif mode == "multi-gpu":
            kwargs["multi_gpu_cfg"] = {"global_rank": 0, "world_size": 2}
        elif mode == "minibatch-normalization":
            kwargs["normalize_advantage_per_mini_batch"] = True
        elif mode == "distribution":
            actor.distribution = GaussianDistribution(4)
        elif mode == "storage":
            storage.actions_shape = (4,)
        with pytest.raises(ValueError):
            MAPPO(actor, critic, storage, **kwargs)


def test_16_entropy_bonus_is_per_drone_mean():
    """With a non-zero coefficient the entropy gradient must be 1/3 of the joint-sum version."""
    entropy_coef = payload_rma_phase1_cfg().algorithm.entropy_coef
    alg = _collect(_algorithm(num_learning_epochs=1, num_mini_batches=1, entropy_coef=entropy_coef))
    batch = next(alg.storage.mini_batch_generator(1, 1))
    alg.actor(batch.observations, stochastic_output=True)
    joint = alg.actor.output_entropy
    assert joint.shape == (16,)
    # Reference: entropy of one drone's four motors, closed form for the tiled std.
    per_drone = (torch.log(alg.actor.output_std[:, :4]) + 0.5 * math.log(2 * math.pi * math.e)).sum(
        -1
    )
    torch.testing.assert_close(joint / 3, per_drone, rtol=1e-6, atol=1e-6)
    # The term that reaches the loss: patch the surrogate to zero so only the entropy
    # bonus drives the std gradient, then compare against the closed-form gradient of
    # -coef * mean(joint / 3) with respect to log std (d entropy / d log std = 1 per motor).
    with (
        mock.patch.object(alg, "_surrogate", return_value=torch.zeros(())),
        mock.patch.object(alg.optimizer, "step", return_value=None),
    ):
        alg.critic.requires_grad_(False)
        metrics = alg.update()
    std_grad = alg.actor.distribution.log_std_param.grad
    # d/d log_std_m of -coef * mean_B(sum_{4 motors} log_std_m) = -coef.
    assert alg.entropy_coef == entropy_coef
    expected = torch.full_like(alg.actor.distribution.log_std_param, -entropy_coef)
    torch.testing.assert_close(std_grad, expected, rtol=1e-5, atol=1e-7)
    assert metrics["entropy"] == pytest.approx(per_drone.mean().item(), abs=1e-6)


def test_15_checkpoint_restores_models_optimizer_and_lr(tmp_path):
    alg = _collect(_algorithm())
    alg.update()
    checkpoint = tmp_path / "mappo.pt"
    torch.save(alg.save(), checkpoint)
    restored = _algorithm()
    assert restored.load(torch.load(checkpoint, weights_only=False), None, strict=True)
    obs = _observations()
    with torch.no_grad():
        torch.testing.assert_close(alg.actor(obs), restored.actor(obs), rtol=0, atol=0)
        torch.testing.assert_close(alg.critic(obs), restored.critic(obs), rtol=0, atol=0)
    _state_equal(alg.actor.state_dict(), restored.actor.state_dict())
    _state_equal(alg.critic.state_dict(), restored.critic.state_dict())
    _state_equal(alg.optimizer.state_dict(), restored.optimizer.state_dict())
    assert restored.learning_rate == alg.learning_rate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_17_gpu_two_iteration_training_and_resume(tmp_path):
    import json
    from pathlib import Path

    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

    from payload_drones.payload_env_cfg import make_payload_env_cfg
    from payload_drones.rl.runner import MARLPhase1Runner

    wp.config.kernel_cache_dir = str(Path.home() / ".cache/warp")
    torch.manual_seed(4551)
    torch.cuda.reset_peak_memory_stats()
    env_cfg = make_payload_env_cfg(num_envs=8, include_history=False, include_team_probe=False)
    env_cfg.seed = 4551
    env = ManagerBasedRlEnv(env_cfg, device="cuda")
    wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
    cfg = payload_rma_phase1_cfg()
    cfg.num_steps_per_env = 16
    cfg.algorithm.num_mini_batches = 4
    cfg.max_iterations = 2
    cfg.logger = "tensorboard"
    cfg.upload_model = False
    runner = MARLPhase1Runner(wrapper, asdict(cfg), str(tmp_path / "training"), device="cuda")
    assert isinstance(runner.alg, MAPPO)
    alg = runner.alg
    actor_before = {n: p.detach().clone() for n, p in alg.actor.named_parameters()}
    critic_before = {n: p.detach().clone() for n, p in alg.critic.named_parameters()}
    original_update = alg.update
    original_step = alg.optimizer.step
    updates = []
    optimizer_steps = []
    probability_errors = []
    initial_ratio_errors = []
    first_minibatch_kls = []

    def counted_step(*args, **kwargs):
        optimizer_steps.append(True)
        return original_step(*args, **kwargs)

    def audited_update():
        st = alg.storage
        frozen_advantages = st.advantages.clone()
        frozen_joint_lp = st.actions_log_prob.clone()
        frozen_params = tuple(p.clone() for p in st.distribution_params)
        with torch.no_grad():
            for index, batch in enumerate(st.mini_batch_generator(4, 1)):
                old = MAPPO._per_agent_log_prob(*batch.old_distribution_params, batch.actions)
                probability_errors.append(
                    (old.sum(-1) - batch.old_actions_log_prob.squeeze(-1)).abs().max().item()
                )
                torch.testing.assert_close(
                    old.sum(-1), batch.old_actions_log_prob.squeeze(-1), rtol=0, atol=1e-5
                )
                alg.actor(batch.observations, stochastic_output=True)
                params = alg.actor.output_distribution_params
                new = MAPPO._per_agent_log_prob(*params, batch.actions)
                ratios = torch.exp(new - old)
                initial_ratio_errors.append((ratios - 1).abs().max().item())
                torch.testing.assert_close(ratios, torch.ones_like(ratios), rtol=0, atol=1e-5)
                if index == 0:
                    first_minibatch_kls.append(
                        alg.actor.get_kl_divergence(batch.old_distribution_params, params)
                        .mean()
                        .item()
                    )
        metrics = original_update()
        assert all(math.isfinite(value) for value in metrics.values())
        assert torch.equal(st.advantages, frozen_advantages)
        assert torch.equal(st.actions_log_prob, frozen_joint_lp)
        for actual, before in zip(st.distribution_params, frozen_params):
            assert torch.equal(actual, before)
        updates.append(metrics)
        return metrics

    try:
        with (
            mock.patch.object(alg, "update", side_effect=audited_update),
            mock.patch.object(alg.optimizer, "step", side_effect=counted_step),
        ):
            runner.learn(2)
        assert len(updates) == 2
        assert len(optimizer_steps) == 2 * 5 * 4
        assert not _unchanged_parameters(alg.actor, actor_before)
        assert not _unchanged_parameters(alg.critic, critic_before)
        obs = wrapper.get_observations()
        with torch.no_grad():
            expected_actor = alg.actor(obs).clone()
            expected_critic = alg.critic(obs).clone()
        checkpoint = tmp_path / "verified_checkpoint.pt"
        runner.save(str(checkpoint))
        assert checkpoint.is_file()
        expected_counter = env.common_step_counter
        env.common_step_counter += 7
        restored = MARLPhase1Runner(wrapper, asdict(cfg), None, device="cuda")
        restored.load(str(checkpoint), map_location="cuda")
        assert restored.current_learning_iteration == 2
        assert env.common_step_counter == expected_counter
        with torch.no_grad():
            torch.testing.assert_close(restored.alg.actor(obs), expected_actor, rtol=0, atol=1e-6)
            torch.testing.assert_close(restored.alg.critic(obs), expected_critic, rtol=0, atol=1e-6)

        def check_optimizer_state():
            expected = alg.optimizer.state_dict()
            actual = restored.alg.optimizer.state_dict()
            _state_equal(expected["param_groups"], actual["param_groups"])
            assert expected["state"].keys() == actual["state"].keys()
            for parameter_id, expected_state in expected["state"].items():
                actual_state = actual["state"][parameter_id]
                assert expected_state.keys() == actual_state.keys()
                for name, value in expected_state.items():
                    other = actual_state[name]
                    if name == "step":
                        # map_location="cuda" also moves Adam's scalar counters.
                        # Compare their exact values; moment buffers must keep devices.
                        assert value.numel() == other.numel() == 1
                        torch.testing.assert_close(value.cpu(), other.cpu(), rtol=0, atol=0)
                    else:
                        _state_equal(value, other)

        check_optimizer_state()
        assert alg.learning_rate == restored.alg.learning_rate
        _state_equal(alg.actor.state_dict(), restored.alg.actor.state_dict())
        _state_equal(alg.critic.state_dict(), restored.alg.critic.state_dict())
        # Isolate optimizer restoration: compute one gradient set and copy it.
        # Independent CUDA convolution backward passes need not be bit-identical.
        # Clone cached inference-mode observations as rollout storage normally does.
        resume_obs = obs.clone()
        assert all(not value.is_inference() for value in resume_obs.values())
        alg.optimizer.zero_grad()
        restored.alg.optimizer.zero_grad()
        loss = alg.actor(resume_obs).square().mean() + alg.critic(resume_obs).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(alg.actor.parameters(), alg.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(alg.critic.parameters(), alg.max_grad_norm)
        for model, other in ((alg.actor, restored.alg.actor), (alg.critic, restored.alg.critic)):
            for parameter, restored_parameter in zip(
                model.parameters(), other.parameters(), strict=True
            ):
                restored_parameter.grad = None if parameter.grad is None else parameter.grad.clone()
        alg.optimizer.step()
        restored.alg.optimizer.step()
        continuation_errors = []
        for model, other in ((alg.actor, restored.alg.actor), (alg.critic, restored.alg.critic)):
            for name, value in model.state_dict().items():
                actual = other.state_dict()[name]
                continuation_errors.append((value - actual).abs().max().item())
                # CPU versus CUDA scalar step counters can change bias-correction rounding.
                torch.testing.assert_close(value, actual, rtol=1e-6, atol=1e-7)
        # Saved moments must reload exactly and still agree for identical gradients.
        check_optimizer_state()
        torch.cuda.synchronize()
        report = {
            "teams": 8,
            "rollout_steps": 16,
            "iterations": 2,
            "optimizer_steps": len(optimizer_steps),
            "max_old_probability_error": max(probability_errors),
            "max_initial_ratio_error": max(initial_ratio_errors),
            "initial_batch_joint_kl": first_minibatch_kls,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "metrics": updates,
            "checkpoint": str(checkpoint),
            "checkpoint_restored": True,
            "optimizer_continuation_matches": True,
            "max_continuation_parameter_error": max(continuation_errors),
        }
        output = Path(__file__).resolve().parents[1] / "outputs/mappo-verification"
        output.mkdir(parents=True, exist_ok=True)
        (output / "gpu_result.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if runner.logger.writer is not None:
            runner.logger.writer.close()
        wrapper.close()
