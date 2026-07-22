"""Twelve motor actions, delegated to the validated per-quad action term."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from mj_adapt_drones.mdp.actions import QuadrotorThrustActionCfg

from payload_drones.mdp.team_scaling import N_AGENTS


@dataclass(kw_only=True)
class TeamThrustActionCfg(ActionTermCfg):
    """One fixed-size 3 x 4 motor action term."""

    scale: float = 0.5
    offset: float = 0.5
    default_max_thrust: float = 5.83

    def build(self, env: ManagerBasedRlEnv) -> "TeamThrustAction":
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
            # The reused single-quad term reads env.quad_state for its thrust
            # ceiling.  Select the matching agent just for this conversion.
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

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0
        for term in self._terms:
            term.reset(env_ids)
