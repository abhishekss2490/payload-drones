# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# PPO.update adapted from rsl_rl at d6791cd22882580b5f5acd2f830ef1d79e916a4a.

"""Per-drone PPO clipping with a centralized team critic."""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO
from torch import nn
from torch.distributions import Normal

from payload_drones.rl.models import (
    ACTION_DIM_PER_AGENT,
    N_AGENTS,
    SharedAgentGaussianDistribution,
)


class MAPPO(PPO):
    """Average individually clipped drone surrogates using one team advantage.

    Rollout collection, team GAE, rollout-level advantage normalization, storage,
    and checkpoint handling are inherited from the pinned PPO implementation.
    Old per-drone probabilities are reconstructed from stored Gaussian parameters
    and sampled actions; the model's public log-probability API stays joint.

    The entropy term is the per-drone mean (joint entropy / N_AGENTS), on the same
    scale as the per-drone-mean surrogate (Yu et al., 2022). The joint sum next to
    a per-drone-mean surrogate triples the bonus; with ``entropy_coef = 0`` the term
    is zero either way. The logged ``entropy`` is this per-drone mean.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if (
            self.rnd is not None
            or self.symmetry is not None
            or self.is_multi_gpu
            or self.actor.is_recurrent
            or self.critic.is_recurrent
        ):
            raise ValueError(
                "MAPPO requires feed-forward actor and critic, one GPU or CPU, "
                "and no RND or symmetry extensions."
            )
        if self.normalize_advantage_per_mini_batch:
            raise ValueError(
                "MAPPO requires normalize_advantage_per_mini_batch=False: "
                "normalize team advantages once over the rollout."
            )
        if not isinstance(self.actor.distribution, SharedAgentGaussianDistribution):
            # ValueError like the other configuration guards above (the test suite
            # checks every unsupported mode with one exception type).
            raise ValueError("MAPPO requires SharedAgentGaussianDistribution.")  # noqa: TRY004
        action_dim = N_AGENTS * ACTION_DIM_PER_AGENT
        if self.storage.training_type != "rl" or tuple(self.storage.actions_shape) != (action_dim,):
            raise ValueError(f"MAPPO requires RL storage with {action_dim} actions per team.")
        # The optimizer already owns this encoder through the actor exactly once.
        # Binding before any forward call also covers rollout values and bootstrapping.
        self.critic.bind_shared_trajectory_encoder(self.actor.trajectory_encoder)

    @staticmethod
    def _per_agent_log_prob(
        mean: torch.Tensor, std: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Return [B, 3] log probabilities, summing only each drone's four motors."""
        action_dim = N_AGENTS * ACTION_DIM_PER_AGENT
        if actions.ndim != 2 or actions.shape[1] != action_dim:
            raise ValueError(f"Expected actions [B, {action_dim}], got {tuple(actions.shape)}.")
        if mean.shape != actions.shape or std.shape != actions.shape:
            raise ValueError("Gaussian mean, std, and sampled actions must all have shape [B, 12].")
        log_prob = Normal(mean, std).log_prob(actions)
        return log_prob.reshape(actions.shape[0], N_AGENTS, ACTION_DIM_PER_AGENT).sum(-1)

    @staticmethod
    def _surrogate(ratio: torch.Tensor, adv: torch.Tensor, clip: float) -> torch.Tensor:
        """Clip each [B, 3] ratio against its [B, 1] team advantage, then average."""
        if ratio.ndim != 2 or ratio.shape[1] != N_AGENTS:
            raise ValueError(f"Expected ratios [B, {N_AGENTS}], got {tuple(ratio.shape)}.")
        if adv.shape != (ratio.shape[0], 1):
            raise ValueError(f"Expected team advantages [B, 1], got {tuple(adv.shape)}.")
        surrogate = -adv * ratio
        surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
        return torch.max(surrogate, surrogate_clipped).mean()

    def update(self) -> dict[str, float]:
        """Use per-drone surrogate clipping and report per-drone/joint clip fractions.

        Preserve the pinned single-device PPO value loss, entropy, summed-KL
        controller, optimizer, gradient clipping, and update count.
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_kl = 0
        # KL of the final minibatch: measured against the rollout policy after every
        # step of this iteration, so it is the policy movement the controller reacts
        # to. The mean below includes the near-zero first-epoch minibatches.
        last_kl = 0.0
        mean_clip_fraction = 0
        mean_joint_clip_fraction = 0
        mean_actor_grad_norm = 0
        mean_critic_grad_norm = 0
        explained_variance = (
            1 - (self.storage.returns - self.storage.values).var() / self.storage.returns.var()
        )

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for batch in generator:
            # Preserve the stock actor forward call, including its sampling behavior.
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            distribution_params = self.actor.output_distribution_params
            mean, std = distribution_params
            # Evaluate the stored sampled actions, before environment-side clipping.
            actions_log_prob = self._per_agent_log_prob(mean, std, batch.actions)
            values = self.critic(
                batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
            )
            team_shape = (batch.actions.shape[0], 1)
            if (
                values.shape != team_shape
                or batch.values.shape != team_shape
                or batch.returns.shape != team_shape
            ):
                raise ValueError("Critic values and returns must have shape [B, 1].")
            # Per-drone mean, matching the surrogate's mean over drones. The joint
            # sum would weight the bonus N_AGENTS times relative to the surrogate.
            entropy = self.actor.output_entropy / N_AGENTS

            # Stored distribution parameters and actions stay fixed across all epochs.
            with torch.no_grad():
                old_mean, old_std = batch.old_distribution_params
                old_actions_log_prob = self._per_agent_log_prob(old_mean, old_std, batch.actions)

            # Keep the pinned joint-KL controller and its original bounds.
            with torch.inference_mode():
                kl = self.actor.get_kl_divergence(
                    batch.old_distribution_params, distribution_params
                )
                kl_mean = torch.mean(kl)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            log_ratio = actions_log_prob - old_actions_log_prob
            ratio = torch.exp(log_ratio)
            # compute_returns already normalized [T, E, 1] team advantages once.
            # Broadcast [B, 1] to [B, 3] here without changing the stored advantages.
            surrogate_loss = self._surrogate(ratio, batch.advantages, self.clip_param)

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
            )
            self.optimizer.zero_grad()
            loss.backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_kl += kl_mean.item()
            last_kl = kl_mean.item()
            mean_clip_fraction += ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
            with torch.no_grad():
                joint_ratio = torch.exp(log_ratio.sum(-1))
                mean_joint_clip_fraction += (
                    ((joint_ratio - 1.0).abs() > self.clip_param).float().mean().item()
                )
            mean_actor_grad_norm += actor_grad_norm.item()
            mean_critic_grad_norm += critic_grad_norm.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "Policy/kl": mean_kl / num_updates,
            "Policy/kl_last_minibatch": last_kl,
            "Policy/clip_fraction": mean_clip_fraction / num_updates,
            "Policy/clip_fraction_joint": mean_joint_clip_fraction / num_updates,
            "Policy/grad_norm": mean_actor_grad_norm / num_updates,
            "Critic/explained_variance": explained_variance.item(),
            "Critic/grad_norm": mean_critic_grad_norm / num_updates,
        }
        self.storage.clear()
        return loss_dict
