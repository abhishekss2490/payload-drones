"""Vendored from mj-adapt-drones at commit 1d52e89 on 2026-07-23.

This is a vendored copy for payload-drones independence; do not edit casually.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.utils.lab_api.math import quat_apply


from mjlab.envs import ManagerBasedRlEnv

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class QuadrotorThrustActionCfg(BaseActionCfg):
    """Configuration for quadrotor thrust action with quadratic velocity-to-force mapping.

    The policy outputs actions in [-1, 1], mapped to u ∈ [0, 1] via scale/offset.
    The action term converts these to thrust forces: F = max_thrust_per_motor * u².
    max_thrust_per_motor is read from env.quad_state (set by domain randomisation).
    """

    default_max_thrust: float = 5.83
    """Fallback max thrust per motor when env.quad_state is not available."""

    vis_full_length: float = 0.20
    """World-space length (m) of a motor gauge at full thrust (F = F_max)."""

    def __post_init__(self):
        self.transmission_type = TransmissionType.SITE

    def build(self, env: ManagerBasedRlEnv) -> QuadrotorThrustAction:
        return QuadrotorThrustAction(self, env)


class QuadrotorThrustAction(BaseAction):
    """Apply quadratic velocity-to-thrust conversion for quadrotor motors.

    Signal chain:
        policy a ∈ [-1,1] → scale/offset → u ∈ [0,1] → clamp → F = max_thrust * u²
    """

    cfg: QuadrotorThrustActionCfg

    def __init__(self, cfg: QuadrotorThrustActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._motor_command = torch.zeros_like(self._raw_actions)

    @property
    def motor_command(self) -> torch.Tensor:
        """Normalised motor command u ∈ [0, 1], after scale/offset and clamp."""
        return self._motor_command

    def process_actions(self, actions: torch.Tensor) -> None:
        """Apply scale/offset, clamp to [0,1], then quadratic thrust conversion."""
        self._raw_actions[:] = actions
        u = self._raw_actions * self._scale + self._offset

        if self.cfg.clip is not None:
            u = torch.clamp(u, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

        u = u.clamp(0.0, 1.0)
        self._motor_command[:] = u

        if hasattr(self._env, "quad_state"):
            max_thrust = self._env.quad_state.max_thrust_per_motor.unsqueeze(-1)
        else:
            max_thrust = self.cfg.default_max_thrust

        self._processed_actions = max_thrust * u * u

    def apply_actions(self) -> None:
        self._entity.set_site_effort_target(
            self._processed_actions, site_ids=self._target_ids
        )

    # ------------------------------------------------------------------ #
    # Debug visualization
    # ------------------------------------------------------------------ #
    @staticmethod
    def _capacity_color(frac: float) -> tuple[float, float, float, float]:
        """Green (low) -> yellow (mid) -> red (saturated) ramp for frac in [0, 1]."""
        frac = float(np.clip(frac, 0.0, 1.0))
        green = np.array([0.15, 0.85, 0.20])
        yellow = np.array([0.95, 0.85, 0.10])
        red = np.array([0.95, 0.15, 0.15])
        if frac < 0.5:
            rgb = green + (yellow - green) * (frac / 0.5)
        else:
            rgb = yellow + (red - yellow) * ((frac - 0.5) / 0.5)
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)

    def debug_vis_motors(self, visualizer: "DebugVisualizer") -> None:
        """Per-motor thrust-capacity gauges (fraction of max thrust used).

        Each gauge fills to the **thrust fraction** ``F / F_max = u²`` — i.e. how
        much of the motor's force capacity is in use right now. Along the body
        thrust axis we draw:
          * a faint track from 0 to F_max so the empty headroom is legible;
          * a bright white cap disk at F = F_max (the full-capacity ceiling);
          * a solid bar whose length is the current thrust fraction, colour
            ramping green -> yellow -> red, thickening + hard red at saturation;
          * a cyan disk at the hover thrust fraction (= 1/TWR), so headroom above
            hover is visible at a glance.
        """
        u = self._motor_command  # (B, n_motors), commanded setpoint in [0, 1]
        thrust_frac = u * u  # F / F_max in [0, 1] — the quantity we draw

        # Site world positions, ordered to match the motor columns.
        site_pos = self._entity.data.site_pos_w[:, self._target_ids]  # (B, n, 3)

        # Body thrust axis (body +z) expressed in the world frame.
        quat = self._entity.data.root_link_quat_w  # (B, 4)
        body_z = torch.zeros_like(site_pos[:, 0])  # (B, 3)
        body_z[:, 2] = 1.0
        axis = quat_apply(quat, body_z)  # (B, 3)

        # Hover thrust fraction: F_hover / F_max = (hover_thrust / 4) / F_max = 1/TWR.
        if hasattr(self._env, "quad_state"):
            qs = self._env.quad_state
            hover_frac = (
                (qs.hover_thrust / 4.0) / qs.max_thrust_per_motor.clamp_min(1e-6)
            ).clamp(0.0, 1.0)  # (B,)
        else:
            hover_frac = torch.full((u.shape[0],), float("nan"), device=u.device)

        L = self.cfg.vis_full_length

        tf_cpu = thrust_frac.detach().cpu().numpy()
        pos_cpu = site_pos.detach().cpu().numpy()
        axis_cpu = axis.detach().cpu().numpy()
        hover_cpu = hover_frac.detach().cpu().numpy()

        for env_id in visualizer.get_env_indices(self.num_envs):
            ax = axis_cpu[env_id]
            ax = ax / (np.linalg.norm(ax) + 1e-9)
            for m in range(tf_cpu.shape[1]):
                base = pos_cpu[env_id, m]
                tf = float(tf_cpu[env_id, m])
                top = base + ax * L

                # Faint track from F=0 to F=F_max so the empty headroom is legible.
                visualizer.add_cylinder(
                    start=base,
                    end=top,
                    radius=0.0015,
                    color=(0.75, 0.75, 0.75, 0.6),
                )

                # Full-capacity ceiling: bright white cap disk at F = F_max.
                visualizer.add_cylinder(
                    start=top - ax * 0.002,
                    end=top + ax * 0.002,
                    radius=0.010,
                    color=(0.95, 0.95, 0.95, 0.95),
                )

                # Solid bar: current thrust fraction, coloured + saturation flash.
                saturated = tf >= 0.98
                color = (0.95, 0.05, 0.05, 1.0) if saturated else self._capacity_color(tf)
                radius = 0.010 if saturated else 0.006
                if tf > 1e-3:
                    visualizer.add_cylinder(
                        start=base,
                        end=base + ax * L * tf,
                        radius=radius,
                        color=color,
                        label=f"m{m}: {tf * 100:.0f}%",
                    )

                # Hover tick: cyan disk at the hover thrust fraction (1/TWR).
                hf = float(hover_cpu[env_id])
                if np.isfinite(hf):
                    center = base + ax * L * hf
                    visualizer.add_cylinder(
                        start=center - ax * 0.002,
                        end=center + ax * 0.002,
                        radius=0.008,
                        color=(0.15, 0.7, 0.95, 0.9),
                    )
