"""Latent-distillation algorithms for payload-drone RMA training."""
from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import Distillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from payload_drones.rl.models import MARLStudentModel, MARLTeacherModel


class LatentDistillation(Distillation):
    """Distill dynamics latent z_dyn from teacher encoder to student TCN."""

    student: MARLStudentModel
    teacher: MARLTeacherModel

    def act(self, obs: TensorDict) -> torch.Tensor:
        # Student produces actions for stepping the env
        self.transition.actions = self.student(obs, stochastic_output=True).detach()
        # Teacher's dynamics encoder produces z_dyn (distillation TARGET)
        self.transition.privileged_actions = self.teacher.get_dynamics_latent(obs).detach()
        self.transition.observations = obs
        return self.transition.actions

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        mean_latent_loss = 0.0
        loss = torch.tensor(0.0, device=self.device)
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.student.reset(hidden_state=self.last_hidden_states[0])
            self.teacher.reset(hidden_state=self.last_hidden_states[1])
            self.student.detach_hidden_state()
            for batch in self.storage.generator():
                # Student's adaptation module predicts z_dyn
                z_dyn_hat = self.student.get_adaptation_latent(batch.observations)
                # Teacher's z_dyn was stored as privileged_actions
                z_dyn = batch.privileged_actions

                latent_loss = self.loss_fn(z_dyn_hat, z_dyn)
                loss = loss + latent_loss
                mean_latent_loss += latent_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(
                            self.student.parameters(), self.max_grad_norm,
                        )
                    self.optimizer.step()
                    self.student.detach_hidden_state()
                    loss = torch.tensor(0.0, device=self.device)

                self.student.reset(batch.dones.view(-1))
                self.teacher.reset(batch.dones.view(-1))
                self.student.detach_hidden_state(batch.dones.view(-1))

        mean_latent_loss /= max(cnt, 1)
        self.storage.clear()
        self.last_hidden_states = (
            self.student.get_hidden_state(),
            self.teacher.get_hidden_state(),
        )
        self.student.detach_hidden_state()
        return {"latent_distillation": mean_latent_loss}

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        # Loading from Phase 1 PPO checkpoint
        if load_cfg is None and any(
            "actor_state_dict" in key for key in loaded_dict
        ):
            actor_sd = loaded_dict["actor_state_dict"]

            # Load teacher fully
            self.teacher.load_state_dict(actor_sd, strict=strict)
            self.teacher_loaded = True

            # Load student partially: trajectory_encoder + mlp + normalizer + distribution
            student_sd = self.student.state_dict()
            for key in actor_sd:
                if key.startswith((
                    "trajectory_encoder.",
                    "mlp.",
                    "obs_normalizer.",
                    "distribution.",
                )):
                    if key in student_sd:
                        student_sd[key] = actor_sd[key]
            self.student.load_state_dict(student_sd, strict=False)

            # Freeze everything except adaptation_module
            for name, param in self.student.named_parameters():
                if not name.startswith("adaptation_module."):
                    param.requires_grad = False

            # Rebuild optimizer with only trainable params
            trainable = [p for p in self.student.parameters() if p.requires_grad]
            self.optimizer = resolve_optimizer("adam")(trainable, lr=self.learning_rate)

            return False  # don't restore iteration

        return super().load(loaded_dict, load_cfg, strict)


class TeamLatentDistillation(LatentDistillation):
    """Latent distillation with env-major, ego-paired ``(B * 3, 8)`` targets."""

    @staticmethod
    def ego_paired_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(f"ego latent shape mismatch: {prediction.shape} vs {target.shape}")
        return torch.nn.functional.mse_loss(prediction, target)

    def update(self) -> dict[str, float]:
        # Stored targets and predictions retain their folded env-major ordering.
        # Replacing the base loss function keeps that pairing explicit.
        original_loss_fn = self.loss_fn
        self.loss_fn = self.ego_paired_loss
        try:
            return super().update()
        finally:
            self.loss_fn = original_loss_fn
