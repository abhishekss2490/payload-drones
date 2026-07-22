"""Vendored from mj-adapt-drones at commit 1d52e89 on 2026-07-23.

This is a vendored copy for payload-drones independence; do not edit casually.
Only the reachable private Gaussian reward helper is retained.
"""

import math

import torch

_GAUSSIAN_SCALE = math.sqrt(-2 * math.log(0.1))


def _exp_kernel(error_sq: torch.Tensor, std: float) -> torch.Tensor:
    """Gaussian exp kernel: 1 at error=0, ~0.1 at error=std."""
    return torch.exp(-0.5 * (error_sq / (std**2)) * _GAUSSIAN_SCALE**2)
