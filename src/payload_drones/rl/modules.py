"""Temporal and trajectory encoders used by payload-drone RMA models."""
from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class CausalConv1d(nn.Module):
    """Conv1d with left-padding so output[t] only depends on input[:t+1]."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalTCN(nn.Module):
    """Stack of causal 1D convolutions for temporal encoding.

    Takes a state-action history sequence and produces a dynamics latent vector
    by reading the last timestep's output after the causal conv stack.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: tuple[int, ...],
        kernel_size: int,
        output_dim: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        act = resolve_nn_activation(activation)

        layers: list[nn.Module] = []
        in_ch = input_channels
        for out_ch in hidden_channels:
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size))
            layers.append(act)
            in_ch = out_ch
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Linear(in_ch, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, C_in) — history sequence, channels-last.

        Returns:
            (B, output_dim) — dynamics latent prediction.
        """
        x = x.permute(0, 2, 1)  # (B, C_in, T)
        x = self.convs(x)  # (B, C_out, T)
        x = x[:, :, -1]  # last timestep: (B, C_out)
        return self.proj(x)


class TrajectoryEncoder(nn.Module):
    """Encode a future trajectory window into a fixed-size latent vector.

    Reshapes the flat trajectory observation into (B, features_per_wp, num_wp),
    applies a small Conv1d stack over the waypoint sequence, pools via learned
    attention weights, and projects to the output latent dimension.
    """

    def __init__(
        self,
        input_dim: int,
        num_waypoints: int = 200,
        features_per_waypoint: int = 6,
        conv_channels: tuple[int, ...] = (16, 8),
        kernel_sizes: tuple[int, ...] = (5, 3),
        strides: tuple[int, ...] = (2, 2),
        output_dim: int = 16,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        assert input_dim == num_waypoints * features_per_waypoint
        self.num_waypoints = num_waypoints
        self.features_per_waypoint = features_per_waypoint

        act = resolve_nn_activation(activation)

        conv_layers: list[nn.Module] = []
        in_ch = features_per_waypoint
        for out_ch, k, s in zip(conv_channels, kernel_sizes, strides):
            conv_layers.append(nn.Conv1d(in_ch, out_ch, k, stride=s))
            conv_layers.append(act)
            in_ch = out_ch
        self.convs = nn.Sequential(*conv_layers)

        self.attn = nn.Linear(in_ch, 1)
        self.out_proj = nn.Linear(in_ch, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, input_dim) — flattened trajectory window.

        Returns:
            (B, output_dim) — trajectory latent.
        """
        # (B, num_wp, feat) → (B, feat, num_wp)
        x = x.view(-1, self.num_waypoints, self.features_per_waypoint)
        x = x.permute(0, 2, 1)
        x = self.convs(x)  # (B, C_out, seq_len)
        x = x.permute(0, 2, 1)  # (B, seq_len, C_out)

        attn_weights = torch.softmax(self.attn(x), dim=1)  # (B, seq_len, 1)
        x = (x * attn_weights).sum(dim=1)  # (B, C_out)
        return self.out_proj(x)
