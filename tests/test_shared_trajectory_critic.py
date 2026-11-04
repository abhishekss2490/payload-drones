"""Actor-owned trajectory representation with a detached team-value path."""

import math
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from unittest import mock

import pytest
import torch
import warp as wp
from tensordict import TensorDict

wp.config.kernel_cache_dir = "/tmp/payload-drones-warp-cache"

from mjlab.envs import ManagerBasedRlEnv
from rsl_rl.storage import RolloutStorage

from payload_drones.config.rl_config import payload_rma_phase1_cfg
from payload_drones.payload_env_cfg import make_payload_env_cfg
from payload_drones.rl.algorithms import MAPPO
from payload_drones.rl.models import N_AGENTS, POLICY_DIM, MARLTeacherModel
from payload_drones.rl.modules import TrajectoryEncoder


def _observations(step=0, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(4551 + step)
    return TensorDict(
        {
            name: torch.randn(4, width, generator=generator, device=device)
            for name, width in {
                "policy": 114,
                "critic": 170,
                "payload_trajectory": 1200,
                "agent_dynamics": 45,
            }.items()
        },
        batch_size=[4],
        device=device,
    )


def _models(obs, latent_dim=16, normalization=False):
    cfg = payload_rma_phase1_cfg()
    actor_cfg, critic_cfg = asdict(cfg.actor), asdict(cfg.critic)
    for options in (actor_cfg, critic_cfg):
        options.pop("class_name")
        options.update(traj_latent_dim=latent_dim, obs_normalization=normalization)
    actor = MARLTeacherModel(obs, cfg.obs_groups, "actor", 12, **actor_cfg)
    critic = MARLTeacherModel(obs, cfg.obs_groups, "critic", 1, **critic_cfg)
    return actor, critic


def _algorithm(obs=None, device="cpu", latent_dim=16, normalization=False):
    torch.manual_seed(17)
    obs = _observations(device=device) if obs is None else obs
    actor, critic = _models(obs, latent_dim, normalization)
    storage = RolloutStorage("rl", obs["critic"].shape[0], 4, obs, [12], device)
    return MAPPO(
        actor,
        critic,
        storage,
        device=device,
        num_learning_epochs=2,
        num_mini_batches=2,
        entropy_coef=0.002,
        gamma=0.995,
        lam=0.95,
    )


def _collect(alg, device="cpu"):
    with torch.inference_mode():
        for step in range(4):
            alg.act(_observations(step, device))
            rewards = torch.arange(4, device=device).float() * 0.2 + step * 0.3
            alg.process_env_step(
                _observations(step + 1, device),
                rewards,
                torch.zeros(4, dtype=torch.long, device=device),
                {},
            )
        # Bootstrapping must use the final preview even without an actor forward call.
        with mock.patch.object(alg.actor, "forward", side_effect=AssertionError("actor called")):
            alg.compute_returns(_observations(9, device))
    return alg


def test_01_real_preview_changes_value_and_latents_match():
    env = ManagerBasedRlEnv(
        make_payload_env_cfg(num_envs=8, include_history=False, include_team_probe=False),
        device="cpu",
    )
    try:
        obs, _ = env.reset(seed=0)
        alg = _algorithm(obs)
        actor, critic = alg.actor, alg.critic
        assert critic.mlp[0].in_features == 186
        assert critic(obs).shape == (8, 1)
        actor_z = actor.get_latent(obs).view(8, N_AGENTS, -1)[
            :, :, POLICY_DIM : POLICY_DIM + actor.traj_latent_dim
        ]
        critic_z = critic.get_latent(obs)[:, 170:]
        for agent in range(N_AGENTS):
            torch.testing.assert_close(actor_z[:, agent], critic_z, rtol=0, atol=0)
        changed = {name: value.clone() for name, value in obs.items()}
        changed["payload_trajectory"] += 0.25 * torch.randn_like(changed["payload_trajectory"])
        with torch.no_grad():
            difference = (critic(obs) - critic(changed)).abs().max().item()
        assert difference > 1e-6, difference
        print(f"preview-only value change: {difference:.8g}")
    finally:
        env.close()


def test_02_value_loss_cannot_train_encoder_but_actor_loss_can():
    alg = _algorithm()
    obs = _observations()
    obs["payload_trajectory"].requires_grad_(True)
    obs["critic"].requires_grad_(True)
    encoder = alg.actor.trajectory_encoder
    before = {name: p.detach().clone() for name, p in encoder.named_parameters()}
    critic_before = {name: p.detach().clone() for name, p in alg.critic.named_parameters()}
    alg.optimizer.zero_grad(set_to_none=True)
    alg.critic(obs).square().mean().backward()
    assert all(p.grad is None for p in encoder.parameters())
    assert obs["payload_trajectory"].grad is None
    assert obs["critic"].grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in alg.critic.parameters())
    alg.optimizer.step()
    assert all(torch.equal(before[name], p) for name, p in encoder.named_parameters())
    assert any(not torch.equal(critic_before[name], p) for name, p in alg.critic.named_parameters())

    alg.optimizer.zero_grad(set_to_none=True)
    (alg.actor(obs) - 1.0).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    alg.optimizer.step()
    assert any(not torch.equal(before[name], p) for name, p in encoder.named_parameters())
    # The critic sees the updated actor representation, not a detached stale cache.
    with torch.no_grad():
        torch.testing.assert_close(
            alg.critic.get_latent(obs)[:, 170:], encoder(obs["payload_trajectory"]), rtol=0, atol=0
        )


def test_03_single_parameter_owner_and_binding_guards():
    obs = _observations()
    actor, critic = _models(obs)
    with pytest.raises(RuntimeError, match="bind the actor"):
        critic(obs)
    wrong_encoder = TrajectoryEncoder(input_dim=1200, output_dim=8)
    with pytest.raises(ValueError, match="dimensions must match"):
        critic.bind_shared_trajectory_encoder(wrong_encoder)
    with pytest.raises(ValueError, match="only the centralized critic"):
        actor.bind_shared_trajectory_encoder(actor.trajectory_encoder)
    storage = RolloutStorage("rl", 4, 4, obs, [12], "cpu")
    alg = MAPPO(actor, critic, storage)
    encoder_ids = {id(p) for p in actor.trajectory_encoder.parameters()}
    assert encoder_ids.isdisjoint({id(p) for p in critic.parameters()})
    counts = Counter(id(p) for group in alg.optimizer.param_groups for p in group["params"])
    assert all(count == 1 for count in counts.values())
    assert all(counts[parameter_id] == 1 for parameter_id in encoder_ids)
    assert critic._trajectory_encoder_ref() is actor.trajectory_encoder
    assert not any(isinstance(module, TrajectoryEncoder) for module in critic.modules())
    assert any(name.startswith("trajectory_encoder.") for name in actor.state_dict())
    assert all(name.startswith("mlp.") for name in critic.state_dict())
    assert len(critic.state_dict()) == 6
    bad_groups = dict(payload_rma_phase1_cfg().obs_groups, critic=("critic",))
    with pytest.raises(ValueError, match="observation groups"):
        MARLTeacherModel(obs, bad_groups, "critic", 1)


def test_04_bootstrap_and_full_ppo_update_keep_advantages_fixed():
    alg = _collect(_algorithm())
    fixed = {
        name: getattr(alg.storage, name).clone()
        for name in ("advantages", "returns", "values", "actions_log_prob")
    }
    assert alg.storage.values.shape == (4, 4, 1)
    assert abs(alg.storage.advantages.mean().item()) < 1e-6
    before = {
        name: p.detach().clone() for name, p in alg.actor.trajectory_encoder.named_parameters()
    }
    metrics = alg.update()
    assert all(math.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(before[name], p)
        for name, p in alg.actor.trajectory_encoder.named_parameters()
    )
    for name, expected in fixed.items():
        torch.testing.assert_close(getattr(alg.storage, name), expected, rtol=0, atol=0)
    assert alg.storage.step == 0


def test_05_checkpoint_round_trip_and_old_critic_rejection(tmp_path):
    alg = _collect(_algorithm())
    alg.update()
    checkpoint = tmp_path / "shared_critic.pt"
    torch.save(alg.save(), checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    restored = _algorithm()
    assert restored.load(saved, None, strict=True)
    obs = _observations(12)
    with torch.no_grad():
        torch.testing.assert_close(restored.actor(obs), alg.actor(obs), rtol=0, atol=0)
        torch.testing.assert_close(restored.critic(obs), alg.critic(obs), rtol=0, atol=0)
    assert restored.critic._trajectory_encoder_ref() is restored.actor.trajectory_encoder
    assert restored.critic._trajectory_encoder_ref() is not alg.actor.trajectory_encoder
    original_optim, restored_optim = alg.optimizer.state_dict(), restored.optimizer.state_dict()
    assert original_optim["param_groups"] == restored_optim["param_groups"]
    for index, state in original_optim["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(restored_optim["state"][index][name], value, rtol=0, atol=0)

    # The schema is unchanged, but the previous 170-input critic cannot be resumed.
    legacy = deepcopy(saved)
    legacy["critic_state_dict"]["mlp.0.weight"] = legacy["critic_state_dict"]["mlp.0.weight"][
        :, :170
    ]
    with pytest.raises(RuntimeError, match="size mismatch"):
        _algorithm().load(legacy, None, strict=True)
    # The actor's shape and state keys are unchanged; actor-only play loading works.
    restored.load(legacy, {"actor": True}, strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored.actor(obs), alg.actor(obs), rtol=0, atol=0)


def test_06_configured_latent_width_and_optional_normalization():
    obs = _observations()
    alg = _algorithm(obs, latent_dim=12, normalization=True)
    assert alg.critic.mlp[0].in_features == 182
    alg.actor.update_normalization(obs)
    alg.critic.update_normalization(obs)
    assert alg.critic.get_latent(obs).shape == (4, 182)
    assert torch.isfinite(alg.critic(obs)).all()
    alg.optimizer.zero_grad(set_to_none=True)
    alg.critic(obs).square().mean().backward()
    assert all(p.grad is None for p in alg.actor.trajectory_encoder.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_07_small_cuda_rollout_update_and_gradient_isolation():
    alg = _algorithm(device="cuda")
    obs = _observations(device="cuda")
    alg.optimizer.zero_grad(set_to_none=True)
    alg.critic(obs).square().mean().backward()
    assert all(p.grad is None for p in alg.actor.trajectory_encoder.parameters())
    with torch.no_grad():
        torch.testing.assert_close(
            alg.critic.get_latent(obs)[:, 170:],
            alg.actor.trajectory_encoder(obs["payload_trajectory"]),
        )
    metrics = _collect(alg, device="cuda").update()
    assert all(math.isfinite(value) for value in metrics.values())
    torch.cuda.synchronize()
