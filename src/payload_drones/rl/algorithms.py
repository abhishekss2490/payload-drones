"""Team-specific two-phase RMA distillation."""

from __future__ import annotations

import torch

from mj_adapt_drones.rl.algorithms import LatentDistillation

from payload_drones.rl.models import MARLStudentModel, MARLTeacherModel


class TeamLatentDistillation(LatentDistillation):
    """Latent distillation with env-major, ego-paired ``(B * 3, 8)`` targets."""

    student: MARLStudentModel
    teacher: MARLTeacherModel

    @staticmethod
    def ego_paired_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(f"ego latent shape mismatch: {prediction.shape} vs {target.shape}")
        return torch.nn.functional.mse_loss(prediction, target)

    def update(self) -> dict[str, float]:
        # The base update's stored target and prediction retain their folded
        # env-major ordering.  Replacing its loss function keeps that pairing
        # explicit and protects it from future refactors.
        original_loss_fn = self.loss_fn
        self.loss_fn = self.ego_paired_loss
        try:
            return super().update()
        finally:
            self.loss_fn = original_loss_fn
