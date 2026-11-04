"""Per-quad and team motor thrust action terms for payload transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from robotics_common.mjlab.quadrotor import draw_motor_thrust_gauges

from payload_drones.mdp.team_scaling import N_AGENTS

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
        self._entity.set_site_effort_target(self._processed_actions, site_ids=self._target_ids)

    # ------------------------------------------------------------------ #
    # Debug visualization
    # ------------------------------------------------------------------ #
    def debug_vis_motors(self, visualizer: DebugVisualizer) -> None:
        """Draw this action's per-motor thrust-capacity gauges."""
        if hasattr(self._env, "quad_state"):
            qs = self._env.quad_state
            hover_frac = ((qs.hover_thrust / 4.0) / qs.max_thrust_per_motor.clamp_min(1e-6)).clamp(
                0.0, 1.0
            )
        else:
            hover_frac = torch.full(
                (self.num_envs,), float("nan"), device=self._motor_command.device
            )

        draw_motor_thrust_gauges(
            visualizer,
            self._motor_command,
            self._entity.data.site_pos_w[:, self._target_ids],
            self._entity.data.root_link_quat_w,
            hover_frac,
            self.cfg.vis_full_length,
        )


@dataclass(kw_only=True)
class TeamThrustActionCfg(ActionTermCfg):
    """One fixed-size 3 x 4 motor action term."""

    scale: float = 0.5
    offset: float = 0.5
    default_max_thrust: float = 5.83

    def build(self, env: ManagerBasedRlEnv) -> TeamThrustAction:
        return TeamThrustAction(self, env)


class TeamThrustAction(ActionTerm):
    cfg: TeamThrustActionCfg

    def __init__(self, cfg: TeamThrustActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._terms = []
        for i in range(N_AGENTS):
            quad_cfg = QuadrotorThrustActionCfg(
                entity_name=f"quad_{i}",
                actuator_names=("motor_.*",),
                scale=cfg.scale,
                offset=cfg.offset,
                clip={"motor_.*": (0.0, 1.0)},
                default_max_thrust=cfg.default_max_thrust,
            )
            self._terms.append(quad_cfg.build(env))
        self._raw_actions = torch.zeros(env.num_envs, N_AGENTS * 4, device=env.device)

    @property
    def action_dim(self) -> int:
        return N_AGENTS * 4

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def motor_command(self) -> torch.Tensor:
        """Commands after active-mask zeroing, shaped ``(B, 3, 4)``."""
        return torch.stack([term.motor_command for term in self._terms], dim=1)

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.stack([term._processed_actions for term in self._terms], dim=1)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        for i, term in enumerate(self._terms):
            # The per-quad term reads env.quad_state for its thrust ceiling.
            # Select the matching agent just for this conversion.
            if hasattr(self._env, "team_state"):
                self._env.quad_state = self._env.team_state.agents[i]
            term.process_actions(actions[:, 4 * i : 4 * (i + 1)])
            if hasattr(self._env, "team_state"):
                active = self._env.team_state.active_mask[:, i].unsqueeze(-1)
                # Force the actual u and force to zero, after u^2 conversion.
                term._motor_command.mul_(active)
                term._processed_actions.mul_(active)
        if hasattr(self._env, "team_state"):
            self._env.quad_state = self._env.team_state.agents[0]

    def apply_actions(self) -> None:
        for term in self._terms:
            term.apply_actions()

    def debug_vis_motors(self, visualizer: DebugVisualizer) -> None:
        """Draw the existing per-quad motor-capacity gauges for the whole team."""
        for term in self._terms:
            term.debug_vis_motors(visualizer)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0
        for term in self._terms:
            term.reset(env_ids)
