"""Vendored from mj-adapt-drones at commit 1d52e89 on 2026-07-23.

This is a vendored copy for payload-drones independence; do not edit casually.
Only the reachable private MLP helpers are retained.
"""

import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState, MLP
from rsl_rl.utils import unpad_trajectories


def _rebuild_mlp_with_output_activation(
    model: MLPModel,
    latent_dim: int,
    hidden_dims: tuple[int, ...] | list[int],
    activation: str,
    output_activation: str,
) -> None:
    """Replace model.mlp with one that has a final activation."""
    if model.distribution is None:
        return
    model.mlp = MLP(
        latent_dim, model.distribution.input_dim, hidden_dims, activation,
        last_activation=output_activation,
    )
    model.distribution.init_mlp_weights(model.mlp)


def _squashed_forward(
    model: MLPModel,
    obs: TensorDict,
    masks: torch.Tensor | None,
    hidden_state: HiddenState,
    stochastic_output: bool,
) -> torch.Tensor:
    """MLPModel.forward with a tanh-squashed actor mean."""
    obs = unpad_trajectories(obs, masks) if masks is not None and not model.is_recurrent else obs
    latent = model.get_latent(obs, masks, hidden_state)
    mlp_output = model.mlp(latent)
    if model.distribution is not None:
        mlp_output = torch.tanh(mlp_output)
        if stochastic_output:
            model.distribution.update(mlp_output)
            return model.distribution.sample()
        return model.distribution.deterministic_output(mlp_output)
    return mlp_output
