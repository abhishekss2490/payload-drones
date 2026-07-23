"""`play-stats`: mjlab play with cooperative payload-transport readouts.

The standard play entry point owns argument parsing, checkpoint resolution, and
viewer selection.  This module only replaces its vector-env wrapper so the
normal play path gains an episode summary without changing any of that logic.
"""

from __future__ import annotations

from typing import cast

import torch

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_error_magnitude

from payload_drones.mdp.observations import team_probe
from payload_drones.mdp.payload_command import PayloadTrajectoryCommand
from payload_drones.mdp.team_scaling import GRAVITY, N_AGENTS


class _TeamMotorGaugeVisualizer:
    """Expose the task action's existing motor gauges to mjlab viewers."""

    def __init__(self, env) -> None:
        self._env = env

    def debug_vis(self, visualizer) -> None:
        if "team_thrust" not in self._env.action_manager.active_terms:
            return
        self._env.action_manager.get_term("team_thrust").debug_vis_motors(visualizer)


class TeamStatsVecEnvWrapper(RslRlVecEnvWrapper):
    """RSL-RL wrapper that reports one team-and-payload block per reset."""

    SURVIVE_FRACTION = 0.95

    def __init__(self, *args, command_name: str = "payload_pose", **kwargs):
        super().__init__(*args, **kwargs)
        self._cmd_name = command_name
        n = self.num_envs
        dev = self.device
        self._max_len = float(self.unwrapped.max_episode_length)

        def z(*shape: int) -> torch.Tensor:
            return torch.zeros((n, *shape), device=dev)

        # Running per-episode values.  Each row is reset independently when
        # its environment auto-resets.
        self._steps = z()
        self._sum_speed, self._max_speed = z(), z()
        self._sum_ref_speed, self._max_ref_speed = z(), z()
        self._sum_perr, self._max_perr = z(), z()
        self._sum_verr, self._sum_aerr, self._sum_angrate = z(), z(), z()
        self._max_angrate = z()
        self._sum_tension = z(N_AGENTS)
        self._sum_motor_thrust = z(N_AGENTS)
        self._sum_wrench_residual = z()

        # Reset-time dynamics remain fixed during an episode, but the base
        # environment has already sampled the *next* team when `step()`
        # reports done.  Keep an episode-local snapshot so the block describes
        # the team that actually flew the completed episode.
        self._agent_mass = z(N_AGENTS)
        self._agent_arm = z(N_AGENTS)
        self._agent_twr = z(N_AGENTS)
        self._payload_mass = z()
        self._team_margin = z()
        self._cable_length = z(N_AGENTS)
        self._rho: torch.Tensor | None = None
        self._capture_episode_dynamics()

        # Action terms are not part of mjlab's default manager visualizer
        # registry.  Registering this tiny adapter preserves the existing
        # TeamThrustAction.debug_vis_motors implementation in both viewers;
        # its get_env_indices() handles selected and multi-env play paths.
        self.unwrapped.manager_visualizers.setdefault(
            "team_motor_gauges", _TeamMotorGaugeVisualizer(self.unwrapped)
        )

        self._episode = 0
        self._n_survived = 0
        print(
            "[play-stats] cooperative payload readout enabled "
            f"({n} env(s), max_len={self._max_len:.0f} steps; motor gauges enabled)"
        )

    def step(self, actions: torch.Tensor):
        obs, rew, dones, extras = super().step(actions)

        env = self.unwrapped
        payload = env.scene["payload"]
        cmd = cast(PayloadTrajectoryCommand, env.command_manager.get_term(self._cmd_name))
        command = cmd.command

        payload_vel = payload.data.root_link_lin_vel_w
        speed = payload_vel.norm(dim=-1)
        ref_speed = command[:, 7:10].norm(dim=-1)
        perr = (
            payload.data.root_link_pos_w
            - (command[:, :3] + env.scene.env_origins)
        ).norm(dim=-1)
        verr = (payload_vel - command[:, 7:10]).norm(dim=-1)
        aerr = quat_error_magnitude(cmd.quat_command, payload.data.root_link_quat_w)
        angrate = payload.data.root_link_ang_vel_b.norm(dim=-1)

        # The analysis channel intentionally comes straight from the task,
        # rather than from whatever observation groups a policy happens to use.
        probe = team_probe(env)
        tension = probe[:, :N_AGENTS]
        motor_thrust = probe[:, N_AGENTS : N_AGENTS + N_AGENTS * 4].reshape(
            self.num_envs, N_AGENTS, 4
        ).sum(dim=-1)
        wrench_residual = probe[:, N_AGENTS + N_AGENTS * 4 :].norm(dim=-1)

        self._steps += 1
        self._sum_speed += speed
        self._max_speed = torch.maximum(self._max_speed, speed)
        self._sum_ref_speed += ref_speed
        self._max_ref_speed = torch.maximum(self._max_ref_speed, ref_speed)
        self._sum_perr += perr
        self._max_perr = torch.maximum(self._max_perr, perr)
        self._sum_verr += verr
        self._sum_aerr += aerr
        self._sum_angrate += angrate
        self._max_angrate = torch.maximum(self._max_angrate, angrate)
        self._sum_tension += tension
        self._sum_motor_thrust += motor_thrust
        self._sum_wrench_residual += wrench_residual

        done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        for env_id in done_ids.tolist():
            self._print_episode(env_id)
            self._reset_env(env_id)
            self._capture_episode_dynamics(env_id)

        return obs, rew, dones, extras

    def _capture_episode_dynamics(self, env_id: int | None = None) -> None:
        """Snapshot reset-time randomization directly from ``env.team_state``."""
        env = self.unwrapped
        ids = slice(None) if env_id is None else env_id
        team = env.team_state
        self._agent_mass[ids] = torch.stack([agent.mass for agent in team.agents], dim=-1)[ids]
        self._agent_arm[ids] = torch.stack([agent.arm_length for agent in team.agents], dim=-1)[ids]
        self._agent_twr[ids] = torch.stack([agent.twr for agent in team.agents], dim=-1)[ids]
        self._payload_mass[ids] = team.payload_mass[ids]
        self._team_margin[ids] = team.margin[ids]
        self._cable_length[ids] = team.cable_length[ids]

        cmd = env.command_manager.get_term(self._cmd_name)
        rho_actual = getattr(cmd, "rho_actual", None)
        if isinstance(rho_actual, torch.Tensor):
            if self._rho is None:
                self._rho = torch.zeros(self.num_envs, device=self.device)
            self._rho[ids] = rho_actual[ids]

    def _print_episode(self, i: int) -> None:
        steps = self._steps[i].item()
        count = max(steps, 1.0)
        survived = steps >= self.SURVIVE_FRACTION * self._max_len
        self._episode += 1
        self._n_survived += int(survived)
        survival_rate = 100.0 * self._n_survived / self._episode

        def value(buf: torch.Tensor) -> float:
            return buf[i].item()

        rho = "n/a" if self._rho is None else f"{value(self._rho):.2f}"
        tag = "survived" if survived else "CRASHED "
        tension = self._sum_tension[i] / count
        payload_weight = (self._payload_mass[i] * GRAVITY).clamp_min(1.0e-6)
        shares = tension / payload_weight
        motor_thrust = self._sum_motor_thrust[i] / count
        cable = self._cable_length[i]

        lines = [
            "",
            f"┌─ episode {self._episode}  (env {i})  {tag}  "
            f"len {steps:.0f}/{self._max_len:.0f}  │ rho {rho}  "
            f"│ survival {survival_rate:.0f}% ─",
            "│ agents    mass / arm / TWR  (independently randomized)",
        ]
        for agent in range(N_AGENTS):
            lines.append(
                f"│  q{agent}       {self._agent_mass[i, agent].item() * 1e3:6.1f} g  "
                f"{self._agent_arm[i, agent].item() * 1e3:5.0f} mm  "
                f"TWR {self._agent_twr[i, agent].item():4.2f}"
            )
        lines.extend([
            f"│ payload   mass {value(self._payload_mass):5.3f} kg  "
            f"loaded team TWR {value(self._team_margin):4.2f}  "
            f"cables {cable[0].item():.3f}/{cable[1].item():.3f}/{cable[2].item():.3f} m",
            f"│ speed     payload avg {value(self._sum_speed) / count:5.3f}  "
            f"max {value(self._max_speed):5.3f}  │ ref avg {value(self._sum_ref_speed) / count:5.3f}  "
            f"max {value(self._max_ref_speed):5.3f} m/s",
            f"│ tracking  pos mean/max {value(self._sum_perr) / count:5.3f}/{value(self._max_perr):5.3f} m  "
            f"vel mean {value(self._sum_verr) / count:5.3f} m/s  "
            f"att mean {value(self._sum_aerr) / count:5.3f} rad",
            f"│ payload   angular-rate avg/max {value(self._sum_angrate) / count:5.3f}/{value(self._max_angrate):5.3f} rad/s",
            "│ load share  tension / payload weight  │ capability (TWR, mass) │ motor thrust fraction Σ/4",
        ])
        for agent in range(N_AGENTS):
            lines.append(
                f"│  q{agent}       {tension[agent].item():6.3f} N / {shares[agent].item():5.1%}  "
                f"│ TWR {self._agent_twr[i, agent].item():4.2f}, {self._agent_mass[i, agent].item() * 1e3:6.1f} g "
                f"│ {motor_thrust[agent].item():4.2f}/4"
            )
        lines.extend([
            f"│  total    tension-share {shares.sum().item():5.1%}  "
            f"│ payload wrench residual avg {value(self._sum_wrench_residual) / count:6.3f}",
            "└" + "─" * 104,
        ])
        print("\n".join(lines), flush=True)

    def _reset_env(self, i: int) -> None:
        for buf in (
            self._steps,
            self._sum_speed,
            self._max_speed,
            self._sum_ref_speed,
            self._max_ref_speed,
            self._sum_perr,
            self._max_perr,
            self._sum_verr,
            self._sum_aerr,
            self._sum_angrate,
            self._max_angrate,
            self._sum_tension,
            self._sum_motor_thrust,
            self._sum_wrench_residual,
        ):
            buf[i] = 0.0


def main() -> None:
    """Inject the cooperative stats wrapper into mjlab's normal play script."""
    from mjlab.scripts import play as _play

    _play.RslRlVecEnvWrapper = TeamStatsVecEnvWrapper
    _play.main()


if __name__ == "__main__":
    main()
