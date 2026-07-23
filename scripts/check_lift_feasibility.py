"""Measure conditional payload sampling through the ScaleCfg distribution.

Payload mass is derived from a sampled loaded team thrust-to-weight ratio, so
each randomized team has a non-zero payload and enough collective lift for the
requested 1.5x margin.  This standalone check deliberately follows the same
mass/thrust distribution as the reset event and reports the resulting payload
distribution to catch accidental collapse toward zero.

Run: ``uv run --no-sync python scripts/check_lift_feasibility.py``
"""

from __future__ import annotations

import torch

from payload_drones.mdp.quad_scaling import ScaleCfg, torch_polyval

GRAVITY = 9.81
N_AGENTS = 3
N_SAMPLES = 200_000
MAX_PAYLOAD_MASS = 2.5


def sample_quads(n: int, cfg: ScaleCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mass, max_total_thrust) for n independently sampled quadrotors."""
    arm = cfg.sample_arm_length(n, device)

    avg = torch_polyval(cfg._to_coeffs(cfg.avg_mass_fit, device), arm)
    std = torch_polyval(cfg._to_coeffs(cfg.std_mass_fit, device), arm).clamp(min=0)
    mass = (torch.empty_like(avg).uniform_(-1, 1) * std + avg).clamp(min=0.01)

    twr = torch.empty(n, device=device).uniform_(*cfg.twr_range)
    max_thrust = twr * mass * GRAVITY
    return mass, max_thrust


def sample_payload(
    team_mass: torch.Tensor,
    team_thrust: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (payload_mass, realized_loaded_team_twr)."""
    liftable_mass = team_thrust / GRAVITY
    unloaded_twr = liftable_mass / team_mass
    sampled_margin = 1.5 + torch.rand_like(unloaded_twr) * (
        torch.minimum(torch.full_like(unloaded_twr, 3.0), 0.9 * unloaded_twr) - 1.5
    )
    payload = (liftable_mass / sampled_margin - team_mass).clamp_max(MAX_PAYLOAD_MASS)
    realized_margin = liftable_mass / (team_mass + payload)
    return payload, realized_margin


def main() -> None:
    device = "cpu"
    cfg = ScaleCfg()
    torch.manual_seed(0)

    print(f"arm_length range : {cfg.arm_length_range}")
    print(f"twr range        : {cfg.twr_range}")
    print("loaded TWR       : U(1.5, min(3.0, 0.9 * unloaded_team_twr))")
    print(f"payload cap      : {MAX_PAYLOAD_MASS} kg (upper clamp only)")
    print()

    mass = torch.empty(N_SAMPLES, N_AGENTS)
    thrust = torch.empty(N_SAMPLES, N_AGENTS)
    for i in range(N_AGENTS):
        m, t = sample_quads(N_SAMPLES, cfg, device)
        mass[:, i], thrust[:, i] = m, t

    team_mass = mass.sum(dim=1)
    team_thrust = thrust.sum(dim=1)

    print(f"per-quad mass     : {mass.min():.4f} .. {mass.max():.4f} kg "
          f"(median {mass.median():.4f})")
    print(f"team mass         : {team_mass.min():.4f} .. {team_mass.max():.4f} kg "
          f"(median {team_mass.median():.4f})")
    print(f"team max thrust   : {team_thrust.min():.3f} .. {team_thrust.max():.3f} N "
          f"(median {team_thrust.median():.3f})")
    print()

    payload, realized_margin = sample_payload(team_mass, team_thrust)
    margin_feasible = team_thrust > 1.5 * (team_mass + payload) * GRAVITY
    print("=== conditional-sampling feasibility ===")
    print(f"teams feasible at 1.5x margin : {margin_feasible.float().mean() * 100:.2f} %")
    print(f"teams with positive payload    : {(payload > 0).float().mean() * 100:.2f} %")
    print(f"realized loaded team TWR       : {realized_margin.min():.4f} .. {realized_margin.max():.4f} "
          f"(median {realized_margin.median():.4f})")
    print()
    q = torch.tensor([0.00, 0.05, 0.50, 0.95, 1.00])
    values = payload.quantile(q)
    print("payload mass (kg)")
    print(f"  min / 5th / median / 95th / max: "
          f"{values[0]:.4f} / {values[1]:.4f} / {values[2]:.4f} / {values[3]:.4f} / {values[4]:.4f}")


if __name__ == "__main__":
    main()
