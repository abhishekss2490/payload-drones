"""Quadrotor dynamics scaling and domain randomization for payload transport.

The quadrotor dynamics and scaling model shares heritage with prior single-quad
RMA work.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.dr._core import _get_entity_indices
from mjlab.envs.mdp.dr.body import (
    _build_perturbation_U,
    _cholesky_4x4,
    _decompose_pseudo_inertia_J,
    _reconstruct_pseudo_inertia_J,
)
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg


from mjlab.envs import ManagerBasedRlEnv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def torch_polyval(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Batched GPU polynomial evaluation (highest-degree-first, like np.polyval)."""
    result = torch.zeros_like(x)
    for c in coeffs:
        result = result * x + c
    return result


def _project_inertia_to_pseudo_inertia_cone(
    inertia: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Project principal moments so the pseudo-inertia matrix is SPD."""
    ixx, iyy, izz = inertia.unbind(dim=-1)
    sigma = torch.stack(
        [
            0.5 * (iyy + izz - ixx),
            0.5 * (ixx + izz - iyy),
            0.5 * (ixx + iyy - izz),
        ],
        dim=-1,
    ).clamp_min(eps)
    return torch.stack(
        [
            sigma[..., 1] + sigma[..., 2],
            sigma[..., 0] + sigma[..., 2],
            sigma[..., 0] + sigma[..., 1],
        ],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# QuadState — per-env dynamic parameters
# ---------------------------------------------------------------------------


@dataclass
class QuadState:
    """Per-environment quadrotor state populated by domain randomisation."""

    arm_length: torch.Tensor
    mass: torch.Tensor
    inertia: torch.Tensor  # (num_envs, 3) — Jxx, Jyy, Jzz after pseudo-inertia decomposition
    ipos: torch.Tensor  # (num_envs, 3) — COM offset in body frame
    iquat: torch.Tensor  # (num_envs, 4) — principal inertia axis orientation (wxyz)
    km_kf: torch.Tensor
    motor_tau: torch.Tensor
    max_thrust_per_motor: torch.Tensor
    hover_thrust: torch.Tensor
    twr: torch.Tensor

    @staticmethod
    def zeros(num_envs: int, device: str | torch.device) -> QuadState:
        z = lambda: torch.zeros(num_envs, device=device)
        return QuadState(
            arm_length=z(),
            mass=z(),
            inertia=torch.zeros(num_envs, 3, device=device),
            ipos=torch.zeros(num_envs, 3, device=device),
            iquat=torch.zeros(num_envs, 4, device=device),
            km_kf=z(),
            motor_tau=z(),
            max_thrust_per_motor=z(),
            hover_thrust=z(),
            twr=z(),
        )


# ---------------------------------------------------------------------------
# ScaleCfg — polynomial fit coefficients and sampling
# ---------------------------------------------------------------------------


@dataclass
class ScaleCfg:
    """Polynomial fit coefficients and arm-length sampling configuration.

    All polynomial coefficients are highest-degree-first (same as np.polyval).
    The coefficients model the vehicles used by this environment.
    """

    arm_length_range: tuple[float, float] = (0.05, 0.16)

    twr_range: tuple[float, float] = (2.0, 4.0)

    # Mass fits (degree-3)
    avg_mass_fit: list[float] = field(
        default_factory=lambda: [1.2498e02, -1.0555e01, 2.8744e00, -1.0497e-01]
    )
    std_mass_fit: list[float] = field(
        default_factory=lambda: [-1.2207e02, 4.6697e01, -3.3219e00, 7.1909e-02]
    )

    # Ixx fits (degree-5)
    avg_ixx_fit: list[float] = field(
        default_factory=lambda: [
            5.5211e01,
            -1.5131e01,
            2.7314e00,
            -2.0660e-01,
            7.1531e-03,
            -9.2833e-05,
        ]
    )
    std_ixx_fit: list[float] = field(
        default_factory=lambda: [
            2.7843e01,
            -9.8623e00,
            2.2186e00,
            -2.2685e-01,
            1.0597e-02,
            -1.8536e-04,
        ]
    )

    # Iyy fits (degree-5, same as ixx for symmetric frame)
    avg_iyy_fit: list[float] = field(
        default_factory=lambda: [
            5.5211e01,
            -1.5131e01,
            2.7314e00,
            -2.0660e-01,
            7.1531e-03,
            -9.2833e-05,
        ]
    )
    std_iyy_fit: list[float] = field(
        default_factory=lambda: [
            2.7843e01,
            -9.8623e00,
            2.2186e00,
            -2.2685e-01,
            1.0597e-02,
            -1.8536e-04,
        ]
    )

    # Izz fits (degree-5)
    avg_izz_fit: list[float] = field(
        default_factory=lambda: [
            1.1543e02,
            -3.7899e01,
            6.8413e00,
            -5.1748e-01,
            1.7916e-02,
            -2.3252e-04,
        ]
    )
    std_izz_fit: list[float] = field(
        default_factory=lambda: [
            8.7295e01,
            -4.2189e01,
            9.2968e00,
            -9.1877e-01,
            4.1948e-02,
            -7.2329e-04,
        ]
    )

    # km_kf fits (degree-1)
    avg_km_kf_fit: list[float] = field(
        default_factory=lambda: [1.4044e-01, -5.6717e-03]
    )
    std_km_kf_fit: list[float] = field(default_factory=lambda: [3.3319e-02, 4.8779e-04])

    # Motor time constant for larger prop response: tau ≈ 0.012 + 0.15 * L
    avg_tau_fit: list[float] = field(default_factory=lambda: [0.15, 0.012])
    std_tau_fit: list[float] = field(default_factory=lambda: [0.04, 0.003])

    def _to_coeffs(self, fit: list[float], device: torch.device) -> torch.Tensor:
        return torch.tensor(fit, device=device, dtype=torch.float32)

    def sample_arm_length(self, n: int, device: torch.device) -> torch.Tensor:
        """Uniform arm-length sampling."""
        lo, hi = self.arm_length_range
        return torch.empty(n, device=device).uniform_(lo, hi)


# ---------------------------------------------------------------------------
# Startup event — initialise QuadState
# ---------------------------------------------------------------------------


def initialize_quad_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Create env.quad_state with default values. Mode: startup."""
    del env_ids
    if not hasattr(env, "quad_state"):
        env.quad_state = QuadState.zeros(env.num_envs, env.device)


# ---------------------------------------------------------------------------
# Monolithic DR event function
# ---------------------------------------------------------------------------


@requires_model_fields(
    "body_mass",
    "body_ipos",
    "body_inertia",
    "body_iquat",
    "site_pos",
    "actuator_gear",
    "actuator_dynprm",
    "actuator_ctrlrange",
    recompute=RecomputeLevel.set_const,
)
def quadrotor_dynamics_randomization(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    scale_cfg: ScaleCfg,
    asset_cfg: SceneEntityCfg,
    site_cfg: SceneEntityCfg,
    actuator_cfg: SceneEntityCfg,
    iquat_max_angle: float = 0.175,
    u_perturbation_ranges: dict | None = None,
    motor_noise_xy_frac: float = 0.05,
    motor_noise_z_frac: float = 0.025,
) -> None:
    """Randomise all arm-length-correlated quadrotor dynamics parameters.

    Must run after ``reset_scene_to_default`` and before root-state reset.
    """
    if u_perturbation_ranges is None:
        u_perturbation_ranges = {
            "alpha": (-0.005, 0.005),
            "d": (-0.005, 0.005),
            "t": (-0.005, 0.005),
            "s": (-0.003, 0.003),
        }

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    device = env.device

    asset = env.scene[asset_cfg.name]
    body_indices = _get_entity_indices(asset.indexing, asset_cfg, "body", False).long()
    site_indices = _get_entity_indices(asset.indexing, site_cfg, "site", False).long()
    ctrl_indices = _get_entity_indices(asset.indexing, actuator_cfg, "actuator", False).long()
    n_bodies = len(body_indices)

    # --- 1. Sample arm_length ---
    arm_length = scale_cfg.sample_arm_length(n, device)

    # --- 2. Polynomial fits → sample mass, inertia ---
    def _sample_param(avg_fit: list[float], std_fit: list[float]) -> torch.Tensor:
        avg = torch_polyval(scale_cfg._to_coeffs(avg_fit, device), arm_length)
        std = torch_polyval(scale_cfg._to_coeffs(std_fit, device), arm_length).clamp(
            min=0
        )
        return torch.empty_like(avg).uniform_(-1, 1) * std + avg

    mass = _sample_param(scale_cfg.avg_mass_fit, scale_cfg.std_mass_fit).clamp(min=0.01)
    ixx = _sample_param(scale_cfg.avg_ixx_fit, scale_cfg.std_ixx_fit).clamp(min=1e-6)
    iyy = _sample_param(scale_cfg.avg_iyy_fit, scale_cfg.std_iyy_fit).clamp(min=1e-6)
    izz = _sample_param(scale_cfg.avg_izz_fit, scale_cfg.std_izz_fit).clamp(min=1e-6)
    inertia = torch.stack([ixx, iyy, izz], dim=-1)
    inertia = _project_inertia_to_pseudo_inertia_cone(inertia)

    # --- 3. Explicit iquat perturbation (axis-angle) ---
    axis = torch.randn(n, 3, device=device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    angle = torch.empty(n, device=device).uniform_(0, iquat_max_angle)
    half = angle / 2
    iquat_perturb = torch.cat(
        [half.cos().unsqueeze(-1), half.sin().unsqueeze(-1) * axis], dim=-1
    )

    # --- 4. Build pseudo-inertia J ---
    ipos_zero = torch.zeros(n, 3, device=device)
    J = _reconstruct_pseudo_inertia_J(
        mass.unsqueeze(1),
        ipos_zero.unsqueeze(1),
        inertia.unsqueeze(1),
        iquat_perturb.unsqueeze(1),
    )

    # --- 5. Cholesky ---
    L = _cholesky_4x4(J)

    # --- 6. Small U perturbation ---
    shape = (n, n_bodies)
    r = u_perturbation_ranges
    alpha = torch.empty(shape, device=device).uniform_(*r["alpha"])
    d1 = torch.empty(shape, device=device).uniform_(*r["d"])
    d2 = torch.empty(shape, device=device).uniform_(*r["d"])
    d3 = torch.empty(shape, device=device).uniform_(*r["d"])
    s12 = torch.empty(shape, device=device).uniform_(*r["s"])
    s13 = torch.empty(shape, device=device).uniform_(*r["s"])
    s23 = torch.empty(shape, device=device).uniform_(*r["s"])
    t1 = torch.empty(shape, device=device).uniform_(*r["t"])
    t2 = torch.empty(shape, device=device).uniform_(*r["t"])
    t3 = torch.empty(shape, device=device).uniform_(*r["t"])

    U = _build_perturbation_U(alpha, d1, d2, d3, s12, s13, s23, t1, t2, t3)

    # --- 7. Perturbed J → decompose ---
    L_new = U @ L
    J_new = L_new @ L_new.mT
    mass_final, ipos_final, inertia_final, iquat_final = _decompose_pseudo_inertia_J(
        J_new
    )

    # --- 8. Write inertial properties ---
    env_grid, body_grid = torch.meshgrid(env_ids, body_indices, indexing="ij")
    env.sim.model.body_mass[env_grid, body_grid] = mass_final
    env.sim.model.body_ipos[env_grid, body_grid] = ipos_final
    env.sim.model.body_inertia[env_grid, body_grid] = inertia_final
    env.sim.model.body_iquat[env_grid, body_grid] = iquat_final

    # --- 9. TWR → max_thrust → ctrlrange ---
    twr = torch.empty(n, device=device).uniform_(*scale_cfg.twr_range)
    max_thrust_per_motor = twr * mass_final.squeeze(-1) * 9.81 / 4.0

    env_grid_a, ctrl_grid = torch.meshgrid(env_ids, ctrl_indices, indexing="ij")
    env.sim.model.actuator_ctrlrange[env_grid_a, ctrl_grid, 1] = (
        max_thrust_per_motor.unsqueeze(-1).expand_as(ctrl_grid)
    )

    # --- 10. Motor site positions ---
    d = arm_length / math.sqrt(2)
    h = 0.0075

    sign_pattern = torch.tensor(
        [[1, 1], [1, -1], [-1, -1], [-1, 1]], device=device, dtype=torch.float32
    )
    nominal_xy = d.unsqueeze(-1).unsqueeze(-1) * sign_pattern
    nominal_z = torch.full((n, 4, 1), h, device=device)
    nominal_pos = torch.cat([nominal_xy, nominal_z], dim=-1)

    noise_xy = (
        torch.empty(n, 4, 2, device=device).uniform_(-1, 1)
        * motor_noise_xy_frac
        * arm_length[:, None, None]
    )
    noise_z = (
        torch.empty(n, 4, 1, device=device).uniform_(-1, 1)
        * motor_noise_z_frac
        * arm_length[:, None, None]
    )
    noise = torch.cat([noise_xy, noise_z], dim=-1)
    motor_pos = nominal_pos + noise

    env_grid_s, site_grid = torch.meshgrid(env_ids, site_indices, indexing="ij")
    env.sim.model.site_pos[env_grid_s, site_grid] = motor_pos

    # --- 11. km_kf → actuator_gear ---
    km_kf_avg = torch_polyval(
        scale_cfg._to_coeffs(scale_cfg.avg_km_kf_fit, device), arm_length
    ).abs()
    km_kf_std = torch_polyval(
        scale_cfg._to_coeffs(scale_cfg.std_km_kf_fit, device), arm_length
    ).clamp(min=0)

    # Guard: ensure positive lower bound
    mask = (km_kf_avg - km_kf_std) < 5e-4
    km_kf_std = torch.where(mask, km_kf_std * 0.9, km_kf_std)

    km_kf = (torch.empty_like(km_kf_avg).uniform_(-1, 1) * km_kf_std + km_kf_avg).clamp(
        min=5e-4
    )

    signs = torch.tensor([-1.0, 1.0, -1.0, 1.0], device=device)
    gear_yaw = km_kf.unsqueeze(-1) * signs
    env.sim.model.actuator_gear[env_grid_a, ctrl_grid, 5] = gear_yaw

    # --- 12. Motor time constant τ ---
    tau_avg = torch_polyval(
        scale_cfg._to_coeffs(scale_cfg.avg_tau_fit, device), arm_length
    )
    tau_std = torch_polyval(
        scale_cfg._to_coeffs(scale_cfg.std_tau_fit, device), arm_length
    ).clamp(min=0)
    tau = (torch.empty_like(tau_avg).uniform_(-1, 1) * tau_std + tau_avg).clamp(
        min=0.005
    )

    env.sim.model.actuator_dynprm[env_grid_a, ctrl_grid, 0] = tau.unsqueeze(
        -1
    ).expand_as(ctrl_grid.float())

    # --- 13. Store per-env state ---
    qs = env.quad_state
    qs.arm_length[env_ids] = arm_length
    qs.mass[env_ids] = mass_final.squeeze(-1)
    qs.inertia[env_ids] = inertia_final.squeeze(-2)
    qs.ipos[env_ids] = ipos_final.squeeze(-2)
    qs.iquat[env_ids] = iquat_final.squeeze(-2)
    qs.km_kf[env_ids] = km_kf
    qs.motor_tau[env_ids] = tau
    qs.max_thrust_per_motor[env_ids] = max_thrust_per_motor
    qs.hover_thrust[env_ids] = mass_final.squeeze(-1) * 9.81
    qs.twr[env_ids] = twr


if __name__ == "__main__":
    from unittest.mock import Mock

    from mjlab.entity import Entity, EntityCfg
    from mjlab.sim.sim import Simulation, SimulationCfg

    from payload_drones.assets.quadrotor_constants import (
        QUADROTOR_ARTICULATION,
        get_spec,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 16

    # --- Setup: real entity + simulation ---
    entity_cfg = EntityCfg(
        spec_fn=get_spec,
        articulation=QUADROTOR_ARTICULATION,
    )
    entity = Entity(entity_cfg)
    model = entity.compile()
    sim = Simulation(
        num_envs=num_envs, cfg=SimulationCfg(njmax=300), model=model, device=device
    )
    entity.initialize(model, sim.model, sim.data, device)

    mock_scene = Mock()
    mock_scene.__getitem__ = lambda self, key: entity

    mock_env = Mock(spec=[])
    mock_env.num_envs = num_envs
    mock_env.device = device
    mock_env.scene = mock_scene
    mock_env.sim = sim

    # --- Expand DR model fields ---
    dr_fields = (
        "body_mass",
        "body_ipos",
        "body_inertia",
        "body_iquat",
        "site_pos",
        "actuator_gear",
        "actuator_dynprm",
        "actuator_ctrlrange",
    )
    sim.expand_model_fields(dr_fields)

    # --- Resolve SceneEntityCfgs ---
    scale_cfg = ScaleCfg()
    asset_cfg = SceneEntityCfg("robot", body_names=("quad",))
    site_cfg = SceneEntityCfg(
        "robot",
        site_names=(
            "motor_front_left",
            "motor_front_right",
            "motor_back_right",
            "motor_back_left",
        ),
    )
    actuator_cfg = SceneEntityCfg("robot", actuator_names=("motor_.*",))
    asset_cfg.resolve(mock_scene)
    site_cfg.resolve(mock_scene)
    actuator_cfg.resolve(mock_scene)

    body_idx = _get_entity_indices(entity.indexing, asset_cfg, "body", False).long()
    site_idx = _get_entity_indices(entity.indexing, site_cfg, "site", False).long()
    ctrl_idx = _get_entity_indices(entity.indexing, actuator_cfg, "actuator", False).long()

    # ====================================================================
    # TEST 1: Unit tests — DR parameter correctness
    # ====================================================================
    print("=" * 60)
    print("TEST 1: Unit tests")
    print("=" * 60)

    initialize_quad_state(mock_env, None)
    assert hasattr(mock_env, "quad_state"), "quad_state not created"
    qs = mock_env.quad_state
    assert qs.arm_length.shape == (num_envs,), f"Wrong shape: {qs.arm_length.shape}"
    print("  [1/12] QuadState created OK")

    env_ids = torch.arange(num_envs, device=device)
    quadrotor_dynamics_randomization(
        mock_env,
        env_ids,
        scale_cfg=scale_cfg,
        asset_cfg=asset_cfg,
        site_cfg=site_cfg,
        actuator_cfg=actuator_cfg,
    )

    assert qs.arm_length.unique().numel() == num_envs, "arm_length not unique"
    print("  [2/12] arm_length unique across envs")

    assert qs.mass.unique().numel() == num_envs, "mass not unique"
    print("  [3/12] mass unique across envs")

    env_grid, bg = torch.meshgrid(env_ids, body_idx, indexing="ij")
    body_mass = sim.model.body_mass[env_grid, bg].squeeze(-1)
    assert (body_mass > 0).all(), f"body_mass has zeros: {body_mass}"
    print("  [4/12] body_mass > 0 in model")

    body_inertia = sim.model.body_inertia[env_grid, bg]
    assert (body_inertia > 0).all(), f"body_inertia has zeros"
    print("  [5/12] body_inertia > 0 in model")

    body_iquat = sim.model.body_iquat[env_grid, bg]
    qnorm = body_iquat.norm(dim=-1)
    assert torch.allclose(qnorm, torch.ones_like(qnorm), atol=1e-4), "iquat not unit"
    print("  [6/12] body_iquat unit quaternions")

    eg_s, sg = torch.meshgrid(env_ids, site_idx, indexing="ij")
    site_pos = sim.model.site_pos[eg_s, sg]
    arm_len = qs.arm_length
    site_xy_norm = site_pos[..., :2].norm(dim=-1)
    ratio = site_xy_norm / arm_len.unsqueeze(-1)
    assert ratio.min() > 0.85 and ratio.max() < 1.15, f"site_pos ratio OOB: {ratio}"
    print("  [7/12] site_pos proportional to arm_length")

    eg_a, ag = torch.meshgrid(env_ids, ctrl_idx, indexing="ij")
    gear_5 = sim.model.actuator_gear[eg_a, ag, 5]
    expected_signs = torch.tensor([-1, 1, -1, 1], device=device).float()
    actual_signs = gear_5.sign()
    assert (actual_signs == expected_signs).all(), f"gear signs wrong: {actual_signs[0]}"
    print("  [8/12] actuator_gear[:, 5] signs [-,+,-,+]")

    ctrlrange_max = sim.model.actuator_ctrlrange[eg_a, ag, 1]
    expected_max = qs.max_thrust_per_motor.unsqueeze(-1).expand_as(ctrlrange_max)
    assert torch.allclose(ctrlrange_max, expected_max, atol=1e-4), "ctrlrange mismatch"
    print("  [9/12] actuator_ctrlrange matches max_thrust")

    recomputed = qs.twr * qs.mass * 9.81 / 4.0
    assert torch.allclose(qs.max_thrust_per_motor, recomputed, atol=1e-4), "TWR derivation"
    print("  [10/12] max_thrust = TWR * mass * g / 4")

    dynprm = sim.model.actuator_dynprm[eg_a, ag, 0]
    assert (dynprm > 0.005).all() and (dynprm < 0.1).all(), f"tau OOB: {dynprm}"
    print("  [11/12] actuator_dynprm tau in range")

    hover = qs.mass * 9.81
    assert torch.allclose(qs.hover_thrust, hover, atol=1e-4), "hover_thrust wrong"
    print("  [12/12] hover_thrust = mass * g")

    # Partial reset: re-randomise half the envs and check others unchanged
    old_arm = qs.arm_length.clone()
    half_ids = torch.arange(num_envs // 2, device=device)
    quadrotor_dynamics_randomization(
        mock_env,
        half_ids,
        scale_cfg=scale_cfg,
        asset_cfg=asset_cfg,
        site_cfg=site_cfg,
        actuator_cfg=actuator_cfg,
    )
    unchanged = qs.arm_length[num_envs // 2 :]
    assert torch.allclose(unchanged, old_arm[num_envs // 2 :]), "Partial reset corrupted other envs"
    print("  [BONUS] Partial reset: other envs unchanged")

    print("TEST 1 PASSED\n")

    # ====================================================================
    # TEST 2: Smoke test — DR + physics stepping
    # ====================================================================
    print("=" * 60)
    print("TEST 2: Smoke test (500 physics steps)")
    print("=" * 60)

    quadrotor_dynamics_randomization(
        mock_env,
        env_ids,
        scale_cfg=scale_cfg,
        asset_cfg=asset_cfg,
        site_cfg=site_cfg,
        actuator_cfg=actuator_cfg,
    )

    hover_per_motor = qs.hover_thrust / 4.0
    ctrl = hover_per_motor.unsqueeze(-1).expand(num_envs, 4)
    entity.set_site_effort_target(ctrl, site_ids=site_cfg.site_ids)

    sim.step()
    sim.forward()

    init_pos = entity.data.root_link_pos_w.clone()

    for step in range(500):
        entity.set_site_effort_target(ctrl, site_ids=site_cfg.site_ids)
        sim.step()
        sim.forward()

    final_pos = entity.data.root_link_pos_w
    drift = (final_pos - init_pos).norm(dim=-1)

    assert not final_pos.isnan().any(), "NaN in final position!"
    assert drift.max() < 50.0, f"Position exploded: max drift = {drift.max():.2f}m"
    print(f"  Drift: mean={drift.mean():.3f}m, max={drift.max():.3f}m")

    # Re-randomise and step again
    quadrotor_dynamics_randomization(
        mock_env,
        env_ids,
        scale_cfg=scale_cfg,
        asset_cfg=asset_cfg,
        site_cfg=site_cfg,
        actuator_cfg=actuator_cfg,
    )
    hover_per_motor2 = qs.hover_thrust / 4.0
    ctrl2 = hover_per_motor2.unsqueeze(-1).expand(num_envs, 4)
    for _ in range(100):
        entity.set_site_effort_target(ctrl2, site_ids=site_cfg.site_ids)
        sim.step()
        sim.forward()

    final_pos2 = entity.data.root_link_pos_w
    assert not final_pos2.isnan().any(), "NaN after re-randomisation!"
    print("  Re-randomisation + stepping OK")
    print("TEST 2 PASSED\n")

    # ====================================================================
    # TEST 3: Viewer test (only with DISPLAY)
    # ====================================================================
    import os

    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        import time

        import mujoco
        import mujoco.viewer

        print("=" * 60)
        print("TEST 3: Viewer test — DR'd quad hover")
        print("=" * 60)

        quadrotor_dynamics_randomization(
            mock_env,
            env_ids,
            scale_cfg=scale_cfg,
            asset_cfg=asset_cfg,
            site_cfg=site_cfg,
            actuator_cfg=actuator_cfg,
        )

        hover_per_motor = qs.hover_thrust / 4.0
        ctrl = hover_per_motor.unsqueeze(-1).expand(num_envs, 4)

        mj_model = sim.mj_model
        mj_data = mujoco.MjData(mj_model)

        print(f"  env 0: L={qs.arm_length[0]:.4f}, m={qs.mass[0]:.4f}, TWR={qs.twr[0]:.2f}")

        with mujoco.viewer.launch_passive(mj_model, mj_data) as v:
            cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "trackcom")
            if cam_id >= 0:
                v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                v.cam.fixedcamid = cam_id

            for step in range(2000):
                entity.set_site_effort_target(ctrl, site_ids=site_cfg.site_ids)
                sim.step()
                sim.forward()

                mj_data.qpos[:] = sim.data.qpos[0].cpu().numpy()
                mj_data.qvel[:] = sim.data.qvel[0].cpu().numpy()
                mujoco.mj_forward(mj_model, mj_data)

                v.sync()
                time.sleep(0.002)

                if step == 999:
                    print("  Re-randomising mid-viewer...")
                    quadrotor_dynamics_randomization(
                        mock_env,
                        env_ids,
                        scale_cfg=scale_cfg,
                        asset_cfg=asset_cfg,
                        site_cfg=site_cfg,
                        actuator_cfg=actuator_cfg,
                    )
                    hover_per_motor = qs.hover_thrust / 4.0
                    ctrl = hover_per_motor.unsqueeze(-1).expand(num_envs, 4)
                    print(f"  env 0: L={qs.arm_length[0]:.4f}, m={qs.mass[0]:.4f}, TWR={qs.twr[0]:.2f}")

        del mj_data, mj_model
        print("TEST 3 PASSED\n")
    else:
        print("TEST 3: Skipped (no DISPLAY)")

    del entity, sim
    print("All checks passed.")
