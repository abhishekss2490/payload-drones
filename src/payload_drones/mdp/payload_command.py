"""Payload pose commands with an optional shared translation trajectory."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from mj_adapt_drones.mdp.trajectory import (
    compose,
    compute_timing,
    lissajous_path,
)

from payload_drones.mdp.curriculum import team_twr_bin


@dataclass(kw_only=True)
class PayloadTrajectoryCommandCfg(CommandTermCfg):
    """Payload reference trajectory configuration.

    ``trajectory_enabled=False`` is the goal-reaching ablation: it keeps the
    legacy static position reference and zero linear-velocity reference while
    preserving every command and observation shape.
    """

    entity_name: str = "payload"
    trajectory_enabled: bool = True
    trajectory_decimation: int = 2
    observation_horizon: int = 200
    episode_length_s: float = 10.0

    # Per-team-TWR-bin curriculum.  The range covers the collective capability
    # after the payload mass has been included in the denominator.
    num_twr_bins: int = 8
    team_twr_range: tuple[float, float] = (1.0, 4.0)
    rho_min: float = 0.05
    rho_max: float = 1.0
    rho_step: float = 0.05
    promote_threshold: float = 0.20
    success_window: int = 15
    promote_rate: float = 0.85
    demote_rate: float = 0.50
    survival_fraction: float = 0.95

    # Translation-only Lissajous path. Orientation remains a sampled static
    # setpoint because the cable geometry provides weak roll/pitch authority.
    amplitude_range_xy: tuple[float, float] = (0.25, 0.75)
    amplitude_range_z: tuple[float, float] = (0.10, 0.30)
    offset_range_xy: tuple[float, float] = (-0.75, 0.75)
    offset_range_z: tuple[float, float] = (1.0, 1.5)
    freq_list: list[int] = field(default_factory=lambda: [1, 2, 3])
    delta_range: tuple[float, float] = (0.0, 2.0 * math.pi)
    num_loops: float = 1.0
    ramp_fraction: float = 0.1
    max_payload_speed: float = 1.5

    position_range_x: tuple[float, float] = (-1.5, 1.5)
    position_range_y: tuple[float, float] = (-1.5, 1.5)
    position_range_z: tuple[float, float] = (0.8, 2.0)
    roll_range: tuple[float, float] = (-0.3, 0.3)
    pitch_range: tuple[float, float] = (-0.3, 0.3)
    yaw_range: tuple[float, float] = (-math.pi, math.pi)
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    # Used only before reset-time domain randomization has created TeamState.
    default_quad_mass: float = 0.792
    default_payload_mass: float = 0.8
    default_max_thrust_per_motor: float = 5.83

    def build(self, env: ManagerBasedRlEnv) -> "PayloadTrajectoryCommand":
        return PayloadTrajectoryCommand(self, env)


class PayloadTrajectoryCommand(CommandTerm):
    """Translation trajectory plus a static payload-attitude setpoint."""

    cfg: PayloadTrajectoryCommandCfg

    def __init__(self, cfg: PayloadTrajectoryCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if cfg.observation_horizon < 200:
            raise ValueError("observation_horizon must be at least the 200-waypoint preview")
        if cfg.max_payload_speed <= 0:
            raise ValueError("max_payload_speed must be positive")

        self.physics_dt = env.physics_dt
        self.traj_dt = self.physics_dt * cfg.trajectory_decimation
        self.N_episode = int(round(cfg.episode_length_s / self.traj_dt))
        # This headroom is deliberate: the 200 future indices at the final
        # episode sample remain within the allocation, with no wraparound.
        self.N_buf = self.N_episode + cfg.observation_horizon
        self.position_ref = torch.zeros(env.num_envs, self.N_buf, 3, device=env.device)
        self.velocity_ref = torch.zeros_like(self.position_ref)
        self.acceleration_ref = torch.zeros_like(self.position_ref)
        self._quat_ref = torch.zeros(env.num_envs, 4, device=env.device)
        self._quat_ref[:, 0] = 1.0
        self._angular_velocity_ref = torch.zeros(env.num_envs, 3, device=env.device)
        self.episode_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        self.cumulative_sq_error = torch.zeros(env.num_envs, device=env.device)
        self.step_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.rho_actual = torch.zeros(env.num_envs, device=env.device)
        self.team_twr = torch.zeros(env.num_envs, device=env.device)
        self.twr_bin_edges = torch.linspace(
            *cfg.team_twr_range, cfg.num_twr_bins + 1, device=env.device
        )
        self.bin_rho_ceiling = torch.full((cfg.num_twr_bins,), cfg.rho_min, device=env.device)
        self.bin_attempts = torch.zeros(cfg.num_twr_bins, dtype=torch.long, device=env.device)
        self.bin_successes = torch.zeros(cfg.num_twr_bins, dtype=torch.long, device=env.device)
        self.env_bin = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        self.metrics["position_error"] = torch.zeros(env.num_envs, device=env.device)
        self.metrics["velocity_error"] = torch.zeros(env.num_envs, device=env.device)
        self.metrics["team_twr"] = torch.zeros(env.num_envs, device=env.device)

    def _buf_idx(self) -> torch.Tensor:
        """Current reference index; episode_step is measured in physics steps."""
        return (self.episode_step // self.cfg.trajectory_decimation).clamp(max=self.N_episode - 1)

    @property
    def command(self) -> torch.Tensor:
        """``[pos_ref_local(3), quat_ref(4), v_ref(3), omega_ref(3)]``."""
        idx = self._buf_idx()[:, None, None].expand(-1, 1, 3)
        pos = self.position_ref.gather(1, idx).squeeze(1)
        vel = self.velocity_ref.gather(1, idx).squeeze(1)
        return torch.cat([pos, self._quat_ref, vel, self._angular_velocity_ref], dim=-1)

    @property
    def quat_command(self) -> torch.Tensor:
        return self._quat_ref

    def _sample_static_orientation(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        roll = torch.empty(n, device=self.device).uniform_(*self.cfg.roll_range)
        pitch = torch.empty(n, device=self.device).uniform_(*self.cfg.pitch_range)
        yaw = torch.empty(n, device=self.device).uniform_(*self.cfg.yaw_range)
        self._quat_ref[env_ids] = quat_from_euler_xyz(roll, pitch, yaw)
        self._angular_velocity_ref[env_ids] = 0.0

    def _update_curriculum(self, env_ids: torch.Tensor) -> None:
        """Score completed episodes in the bins assigned for those episodes."""
        graded = self.step_count[env_ids] > 0
        if not bool(graded.any()):
            return
        cfg = self.cfg
        rmse = torch.sqrt(self.cumulative_sq_error[env_ids] / self.step_count[env_ids].clamp_min(1))
        survived = self.episode_step[env_ids] >= int(cfg.episode_length_s / self.physics_dt * cfg.survival_fraction)
        bins = self.env_bin[env_ids]
        count = graded
        success = count & survived & (rmse < cfg.promote_threshold)
        self.bin_attempts.scatter_add_(0, bins[count], torch.ones_like(bins[count]))
        self.bin_successes.scatter_add_(0, bins[success], torch.ones_like(bins[success]))
        ready = self.bin_attempts >= cfg.success_window
        if bool(ready.any()):
            rate = self.bin_successes.float() / self.bin_attempts.clamp_min(1)
            self.bin_rho_ceiling = torch.where(ready & (rate >= cfg.promote_rate), self.bin_rho_ceiling + cfg.rho_step, self.bin_rho_ceiling)
            self.bin_rho_ceiling = torch.where(ready & (rate <= cfg.demote_rate), self.bin_rho_ceiling - cfg.rho_step, self.bin_rho_ceiling)
            self.bin_rho_ceiling.clamp_(cfg.rho_min, cfg.rho_max)
            self.bin_attempts[ready] = 0
            self.bin_successes[ready] = 0

    def _sample_trajectory_difficulty(self, env_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if hasattr(self._env, "team_state"):
            twr, bins = team_twr_bin(self._env, self.twr_bin_edges, cfg.num_twr_bins, env_ids)
        else:
            total_thrust = 3.0 * 4.0 * cfg.default_max_thrust_per_motor
            total_mass = 3.0 * cfg.default_quad_mass + cfg.default_payload_mass
            twr = torch.full((len(env_ids),), total_thrust / (total_mass * 9.81), device=self.device)
            bins = torch.bucketize(twr, self.twr_bin_edges[1:-1]).clamp(0, cfg.num_twr_bins - 1)
        self.team_twr[env_ids] = twr
        self.env_bin[env_ids] = bins
        ceiling = self.bin_rho_ceiling[bins]
        rho = cfg.rho_min + torch.rand(len(env_ids), device=self.device) * (ceiling - cfg.rho_min)
        self.rho_actual[env_ids] = rho
        return (rho * (twr - 1.0) * 9.81).clamp_min(0.1)

    def _generate_trajectory(self, env_ids: torch.Tensor, max_acceleration: torch.Tensor) -> None:
        """Generate derivatives through the base repository's trajectory utility."""
        n = len(env_ids)
        cfg = self.cfg
        A = torch.empty(n, device=self.device).uniform_(*cfg.amplitude_range_xy)
        B = torch.empty(n, device=self.device).uniform_(*cfg.amplitude_range_xy)
        C = torch.empty(n, device=self.device).uniform_(*cfg.amplitude_range_z)
        frequencies = torch.tensor(cfg.freq_list, device=self.device)
        a = frequencies[torch.randint(len(frequencies), (n,), device=self.device)]
        b = frequencies[torch.randint(len(frequencies), (n,), device=self.device)]
        c = frequencies[torch.randint(len(frequencies), (n,), device=self.device)]
        offset = torch.empty(n, 3, device=self.device)
        offset[:, :2].uniform_(*cfg.offset_range_xy)
        offset[:, 2].uniform_(*cfg.offset_range_z)
        path_kwargs = {
            "A": A, "B": B, "C": C, "a": a, "b": b, "c": c,
            "delta": torch.empty(n, device=self.device).uniform_(*cfg.delta_range),
        }
        # Evaluate the actual sampled path over a full loop.  This removes the
        # frequency-ratio artefact of bounding all sinusoid components at once.
        # The grid resolution matches the generated reference buffer, and the
        # base path utility supplies both derivatives; no trajectory math is
        # duplicated here.
        s_final = cfg.num_loops * 2.0 * math.pi
        s_grid = torch.linspace(0.0, s_final, self.N_buf, device=self.device)
        _, dp_grid, ddp_grid = lissajous_path(s_grid, **path_kwargs, device=self.device)
        max_dp = dp_grid.norm(dim=-1).amax(dim=1).clamp_min(1.0e-8)
        max_ddp = ddp_grid.norm(dim=-1).amax(dim=1).clamp_min(1.0e-8)

        # ``max_acceleration`` is the curriculum's physical acceleration
        # request.  Convert it to its matching angular rate, cap that rate by
        # the desired true peak speed, then compose the existing timing law in
        # one generation pass.
        curriculum_omega = torch.sqrt(max_acceleration / max_ddp)
        cap_omega = cfg.max_payload_speed / max_dp
        omega = torch.minimum(curriculum_omega, cap_omega)
        total_time = s_final / (omega * (1.0 - cfg.ramp_fraction))
        t = torch.arange(self.N_buf, device=self.device, dtype=torch.float32) * self.traj_dt
        t = t.unsqueeze(0).expand(n, -1).clamp(max=total_time.unsqueeze(1))
        timing = compute_timing(
            t,
            total_time.unsqueeze(1),
            omega.unsqueeze(1),
            cfg.ramp_fraction,
        )
        position, velocity, acceleration = compose(
            lissajous_path(timing[0], **path_kwargs, device=self.device), timing
        )
        self.position_ref[env_ids] = position + offset[:, None, :]
        self.velocity_ref[env_ids] = velocity
        self.acceleration_ref[env_ids] = acceleration

    def _generate_static_setpoint(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        target = torch.empty(n, 3, device=self.device)
        target[:, 0].uniform_(*self.cfg.position_range_x)
        target[:, 1].uniform_(*self.cfg.position_range_y)
        target[:, 2].uniform_(*self.cfg.position_range_z)
        self.position_ref[env_ids] = target[:, None, :]
        self.velocity_ref[env_ids] = 0.0
        self.acceleration_ref[env_ids] = 0.0
        self.rho_actual[env_ids] = 0.0

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._update_curriculum(env_ids)
        self._sample_static_orientation(env_ids)
        if self.cfg.trajectory_enabled:
            self._generate_trajectory(env_ids, self._sample_trajectory_difficulty(env_ids))
        else:
            self._generate_static_setpoint(env_ids)
        self.episode_step[env_ids] = 0
        self.cumulative_sq_error[env_ids] = 0.0
        self.step_count[env_ids] = 0

    def _update_metrics(self) -> None:
        payload = self._env.scene[self.cfg.entity_name]
        ref = self.command
        pos_error = (payload.data.root_link_pos_w - (ref[:, :3] + self._env.scene.env_origins)).norm(dim=-1)
        vel_error = (payload.data.root_link_lin_vel_w - ref[:, 7:10]).norm(dim=-1)
        self.metrics["position_error"] = pos_error
        self.metrics["velocity_error"] = vel_error
        self.metrics["team_twr"] = self.team_twr.clone()
        self.cumulative_sq_error += pos_error.square()
        self.step_count += 1

    def _update_command(self) -> None:
        self.episode_step += round(self._env.step_dt / self.physics_dt)


# Compatibility name for existing downstream imports; the default is now moving.
PayloadPoseCommandCfg = PayloadTrajectoryCommandCfg
PayloadPoseCommand = PayloadTrajectoryCommand
