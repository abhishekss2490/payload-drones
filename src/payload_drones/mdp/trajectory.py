"""Differentiable trajectory generation for payload transport commands."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


def _col(x: float | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Scalar or (B,) -> (B, 1) for broadcasting with (N,) time arrays."""
    if isinstance(x, (int, float)):
        return torch.tensor([[x]], dtype=torch.float32, device=device)
    if x.dim() == 0:
        return x.reshape(1, 1)
    return x.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Timing law
# ---------------------------------------------------------------------------


def _quintic_smoothstep(u: torch.Tensor):
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    h = 10 * u3 - 15 * u4 + 6 * u5
    dh = 30 * u2 - 60 * u3 + 30 * u4
    ddh = 60 * u - 180 * u2 + 120 * u3
    return h, dh, ddh


def _quintic_integral(u: torch.Tensor):
    u2 = u * u
    u4 = u2 * u2
    return 2.5 * u4 - 3 * u4 * u + u4 * u2


def compute_timing(t, total_time, omega, ramp_fraction):
    """Compute (s, sd, sdd, sddd).

    Supports both shared timing (t: (N,), total_time/omega: scalar)
    and batched timing (t: (B, N), total_time/omega: (B, 1) tensors).
    """
    t_r = ramp_fraction * total_time

    in_ramp_up = t <= t_r
    in_ramp_down = t >= (total_time - t_r)

    u_up = (t / t_r).clamp(0, 1)
    h_up, dh_up, ddh_up = _quintic_smoothstep(u_up)

    u_down = ((t - (total_time - t_r)) / t_r).clamp(0, 1)
    h_down, dh_down, ddh_down = _quintic_smoothstep(u_down)

    s_ramp_end = omega * t_r * 0.5
    s_down_start = s_ramp_end + omega * (total_time - 2 * t_r)

    s_up = omega * t_r * _quintic_integral(u_up)
    s_cruise = s_ramp_end + omega * (t - t_r)
    s_down = s_down_start + omega * t_r * (u_down - _quintic_integral(u_down))

    sd_up = omega * h_up
    if isinstance(omega, torch.Tensor):
        sd_cruise = omega.expand_as(t)
    else:
        sd_cruise = torch.full_like(t, omega)
    sd_down = omega * (1.0 - h_down)

    sdd_up = omega / t_r * dh_up
    sdd_cruise = torch.zeros_like(t)
    sdd_down = -omega / t_r * dh_down

    sddd_up = omega / (t_r**2) * ddh_up
    sddd_cruise = torch.zeros_like(t)
    sddd_down = -omega / (t_r**2) * ddh_down

    s = torch.where(in_ramp_up, s_up, torch.where(in_ramp_down, s_down, s_cruise))
    sd = torch.where(in_ramp_up, sd_up, torch.where(in_ramp_down, sd_down, sd_cruise))
    sdd = torch.where(
        in_ramp_up, sdd_up, torch.where(in_ramp_down, sdd_down, sdd_cruise)
    )
    sddd = torch.where(
        in_ramp_up, sddd_up, torch.where(in_ramp_down, sddd_down, sddd_cruise)
    )

    return s, sd, sdd, sddd


def compose(path_derivs, timing):
    """Chain rule: path (B, N, 3) + timing -> trajectory (B, N, 3).

    Timing can be (N,) [shared] or (B, N) [per-env].
    """
    p, dp, ddp = path_derivs
    _, sd, sdd, _ = timing

    if sd.dim() == 1:
        sd = sd[None, :, None]
        sdd = sdd[None, :, None]
    else:
        sd = sd.unsqueeze(-1)
        sdd = sdd.unsqueeze(-1)

    position = p
    velocity = dp * sd
    acceleration = ddp * sd**2 + dp * sdd

    return position, velocity, acceleration


# ---------------------------------------------------------------------------
# Path function
# ---------------------------------------------------------------------------


def _tensor_lcm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise LCM for integer tensors: lcm(a,b) = |a*b| / gcd(a,b)."""
    return (a * b).abs() // torch.gcd(a, b)


def lissajous_path(s, A, B, C, a=1, b=1, c=None, delta=math.pi / 2, device=None):
    """Lissajous path. A, B, C, delta: scalar or (batch,). a, b, c: int or (batch,) int tensor."""
    if device is None:
        device = s.device

    if c is None:
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            c = _tensor_lcm(torch.as_tensor(a, device=device), torch.as_tensor(b, device=device))
        else:
            c = math.lcm(a, b)

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor) or isinstance(c, torch.Tensor):
        a = _col(torch.as_tensor(a, dtype=torch.float32, device=device), device)
        b = _col(torch.as_tensor(b, dtype=torch.float32, device=device), device)
        c = _col(torch.as_tensor(c, dtype=torch.float32, device=device), device)
    else:
        pass

    A, B, C, delta = (
        _col(A, device),
        _col(B, device),
        _col(C, device),
        _col(delta, device),
    )

    phase_a = a * s + delta
    phase_b = b * s
    phase_c = c * s
    sin_a, cos_a = torch.sin(phase_a), torch.cos(phase_a)
    sin_b, cos_b = torch.sin(phase_b), torch.cos(phase_b)
    sin_c, cos_c = torch.sin(phase_c), torch.cos(phase_c)

    px, py, pz = A * sin_a, B * sin_b, C * sin_c
    ref = torch.broadcast_shapes(px.shape, py.shape, pz.shape)

    p = torch.stack([px.expand(ref), py.expand(ref), pz.expand(ref)], dim=-1)
    dpx, dpy, dpz = A * a * cos_a, B * b * cos_b, C * c * cos_c
    dp = torch.stack([dpx.expand(ref), dpy.expand(ref), dpz.expand(ref)], dim=-1)
    ddp = torch.stack([
        (-A * a**2 * sin_a).expand(ref),
        (-B * b**2 * sin_b).expand(ref),
        (-C * c**2 * sin_c).expand(ref),
    ], dim=-1)

    return p, dp, ddp


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryOutput:
    t: torch.Tensor  # (N,) or (B, N)
    position: torch.Tensor  # (B, N, 3)
    velocity: torch.Tensor  # (B, N, 3)
    acceleration: torch.Tensor  # (B, N, 3)

    def __post_init__(self):
        for name in ("t", "position", "velocity", "acceleration"):
            tensor = getattr(self, name)
            if tensor.dtype != torch.float32:
                tensor = tensor.to(dtype=torch.float32)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            object.__setattr__(self, name, tensor)

    def to(self, device: torch.device | str) -> TrajectoryOutput:
        device = torch.device(device)
        if self.t.device == device:
            return self
        return TrajectoryOutput(
            t=self.t.to(device),
            position=self.position.to(device),
            velocity=self.velocity.to(device),
            acceleration=self.acceleration.to(device),
        )


def _apply_offset(position, offset, device):
    if offset is None:
        return position
    offset = torch.as_tensor(offset, dtype=torch.float32, device=device)
    if offset.dim() == 1:
        offset = offset[None, None, :]
    elif offset.dim() == 2:
        offset = offset[:, None, :]
    return position + offset


def generate_trajectory(
    path_fn,
    path_kwargs: dict,
    total_time: float = 20.0,
    dt: float = 0.01,
    num_loops: float = 1.0,
    ramp_fraction: float = 0.1,
    offset: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> TrajectoryOutput:
    """Generate a trajectory with shared timing (single total_time for all envs).

    Args:
        path_fn: path_fn(s, **kwargs, device=device) -> (p, p', p'').
        path_kwargs: Values can be scalars or (B,) tensors.
        total_time: Duration in seconds (shared across batch).
        dt: Timestep.
        num_loops: Number of complete loops (S_f = num_loops * 2*pi).
        ramp_fraction: Fraction of total_time for each ramp. In (0, 0.5].
        offset: Optional (B, 3) or (3,) translation on positions.
        device: Torch device.
    """
    ramp_fraction = max(1e-6, min(ramp_fraction, 0.5))

    N = int(total_time / dt)
    t = torch.linspace(0, total_time, N, device=device)

    S_f = num_loops * 2 * math.pi
    omega = S_f / (total_time * (1.0 - ramp_fraction))

    timing = compute_timing(t, total_time, omega, ramp_fraction)
    path_derivs = path_fn(timing[0], **path_kwargs, device=device)
    position, velocity, acceleration = compose(path_derivs, timing)

    return TrajectoryOutput(
        t=t,
        position=_apply_offset(position, offset, device),
        velocity=velocity,
        acceleration=acceleration,
    )


def _eval_path_batched(path_fn, path_kwargs, s, B, device):
    """Evaluate path on (B, N) s-grid. Returns (p, dp, ddp) each (B, N, 3)."""
    derivs = path_fn(s, **path_kwargs, device=device)
    N = s.shape[1]
    return tuple(d.reshape(B, N, 3) if d.shape[0] == 1 else d for d in derivs)


def generate_trajectory_for_max_acceleration(
    path_fn,
    path_kwargs: dict,
    max_acceleration: float | torch.Tensor,
    dt: float = 0.01,
    num_loops: float = 1.0,
    ramp_fraction: float = 0.1,
    offset: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    N_out: int | None = None,
    spawn_offset: torch.Tensor | None = None,
) -> TrajectoryOutput:
    """Generate trajectory(ies) scaled so max||a_ref|| ≈ max_acceleration.

    Uses the fact that peak acceleration occurs during cruise where acc = ddp * omega².
    max||ddp|| is computed analytically from path params (upper bound: all sines peak together).
    Only evaluates the output window needed (N_out steps starting at spawn_offset).

    Args:
        max_acceleration: Scalar or (B,) tensor. Per-env aggressiveness.
        N_out: If set, only output this many steps (starting at spawn_offset).
            If None, outputs the full trajectory length.
        spawn_offset: (B,) int tensor of step offsets into the full trajectory.
            Only used when N_out is set. If None with N_out set, defaults to zeros.
        Other args: same as generate_trajectory.

    Returns:
        TrajectoryOutput with (B, N, 3) tensors and t as (B, N).
    """
    ramp_fraction = max(1e-6, min(ramp_fraction, 0.5))
    S_f = num_loops * 2 * math.pi

    max_a = torch.as_tensor(max_acceleration, dtype=torch.float32, device=device)
    if max_a.dim() == 0:
        max_a = max_a.unsqueeze(0)
    B = max_a.shape[0]

    # --- Analytic scaling: max||acc|| = omega² * max||ddp|| during cruise ---
    # For lissajous: ddp = (-A*a²*sin, -B*b²*sin, -C*c²*sin)
    # Upper bound: ||ddp|| <= sqrt((A*a²)² + (B*b²)² + (C*c²)²)
    A_t = torch.as_tensor(path_kwargs["A"], dtype=torch.float32, device=device)
    B_t = torch.as_tensor(path_kwargs["B"], dtype=torch.float32, device=device)
    C_t = torch.as_tensor(path_kwargs["C"], dtype=torch.float32, device=device)
    a_t = torch.as_tensor(path_kwargs.get("a", 1), dtype=torch.float32, device=device)
    b_t = torch.as_tensor(path_kwargs.get("b", 1), dtype=torch.float32, device=device)
    c_t = torch.as_tensor(path_kwargs.get("c", 1), dtype=torch.float32, device=device)
    max_ddp_norm = torch.sqrt((A_t * a_t**2)**2 + (B_t * b_t**2)**2 + (C_t * c_t**2)**2).clamp(min=1e-8)

    omega = torch.sqrt(max_a / max_ddp_norm)
    total_time = S_f / (omega * (1.0 - ramp_fraction))

    # --- Build time grid for the output window ---
    T_max = total_time.max().item()
    N_full = max(int(T_max / dt), 2)

    if N_out is not None:
        N = N_out
        if spawn_offset is None:
            spawn_offset = torch.zeros(B, dtype=torch.long, device=device)
        t_start = (spawn_offset.float() * dt).unsqueeze(1)
        t = t_start + torch.linspace(0, (N - 1) * dt, N, device=device).unsqueeze(0)
    else:
        N = N_full
        tau = torch.linspace(0, 1, N, device=device).unsqueeze(0)
        t = tau * total_time.unsqueeze(1)

    # Clamp to each env's total_time
    t = t.clamp(max=total_time.unsqueeze(1))

    timing = compute_timing(
        t, total_time.unsqueeze(1), omega.unsqueeze(1), ramp_fraction
    )

    path_derivs = _eval_path_batched(path_fn, path_kwargs, timing[0], B, device)
    position, velocity, acceleration = compose(path_derivs, timing)

    return TrajectoryOutput(
        t=t,
        position=_apply_offset(position, offset, device),
        velocity=velocity,
        acceleration=acceleration,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def compute_body_frame(acc: np.ndarray, g: float = 9.81):
    """Compute required body frame from differential flatness (yaw=0)."""
    thrust_vec = acc + np.array([0, 0, g])
    z_B = thrust_vec / np.maximum(
        np.linalg.norm(thrust_vec, axis=1, keepdims=True), 1e-8
    )

    x_C = np.array([[1.0, 0.0, 0.0]])
    y_B = np.cross(z_B, x_C)
    y_B = y_B / np.maximum(np.linalg.norm(y_B, axis=1, keepdims=True), 1e-8)
    x_B = np.cross(y_B, z_B)

    return x_B, y_B, z_B


def plot_trajectory(
    traj: TrajectoryOutput,
    batch_idx: int = 0,
    title: str = "Trajectory",
    show_body_frame: bool = True,
    n_frames: int = 25,
    axis_length: float = 0.08,
):
    import matplotlib.pyplot as plt

    t_tensor = traj.t
    if t_tensor.dim() == 2:
        t = t_tensor[batch_idx].cpu().numpy()
    else:
        t = t_tensor.cpu().numpy()

    pos = traj.position[batch_idx].cpu().numpy()
    vel = traj.velocity[batch_idx].cpu().numpy()
    acc = traj.acceleration[batch_idx].cpu().numpy()

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    labels = ["x", "y", "z"]
    colors = ["tab:red", "tab:green", "tab:blue"]

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], alpha=0.5)
    ax3d.scatter(*pos[0], color="green", s=50, label="start")
    ax3d.scatter(*pos[-1], color="red", s=50, label="end")

    if show_body_frame:
        x_B, y_B, z_B = compute_body_frame(acc)
        indices = np.linspace(0, len(t) - 1, n_frames, dtype=int)
        for idx in indices:
            p = pos[idx]
            for axis, color in zip([x_B, y_B, z_B], ["red", "green", "blue"]):
                ax3d.quiver(
                    p[0],
                    p[1],
                    p[2],
                    axis[idx, 0],
                    axis[idx, 1],
                    axis[idx, 2],
                    length=axis_length,
                    color=color,
                    arrow_length_ratio=0.15,
                    linewidth=1.5,
                )

    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("3D Path")
    ax3d.legend()

    for subplot_idx, (data, ylabel, subplot_title) in enumerate(
        [
            (pos, "position [m]", "Position"),
            (vel, "velocity [m/s]", "Velocity"),
            (acc, "acceleration [m/s²]", "Acceleration"),
        ]
    ):
        ax = fig.add_subplot(2, 2, subplot_idx + 2)
        for i, (l, c) in enumerate(zip(labels, colors)):
            ax.plot(t, data[:, i], color=c, label=l)
        ax.set_xlabel("t [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(subplot_title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":
    traj = generate_trajectory(
        path_fn=lissajous_path,
        path_kwargs={
            "A": 1.0,
            "B": 1.0,
            "C": 0.5,
            "a": 2,
            "b": 2,
            "delta": np.pi - 0.25,
        },
        total_time=20.0,
        dt=0.01,
        num_loops=3.0,
        ramp_fraction=0.1,
        offset=np.array([0.0, 0.0, 1.0]),
    )
    plot_trajectory(traj, title="Lissajous Trajectory", n_frames=100)

    # traj_scaled = generate_trajectory_for_max_acceleration(
    #     path_fn=lissajous_path,
    #     path_kwargs={"A": 1.0, "B": 1.0, "C": 0.5, "a": 2, "b": 3, "delta": np.pi / 4},
    #     max_acceleration=np.array([4.0, 8.0, 12.0]),
    #     dt=0.01,
    #     num_loops=1.0,
    #     ramp_fraction=0.1,
    #     offset=np.array([0.0, 0.0, 1.0]),
    # )

    # # plot the scaled trajectories for each env present in traj_scaled on different figures
    # for i in range(traj_scaled.position.shape[0]):
    #     plot_trajectory(
    #         TrajectoryOutput(
    #             t=traj_scaled.t[i],
    #             position=traj_scaled.position[i : i + 1],
    #             velocity=traj_scaled.velocity[i : i + 1],
    #             acceleration=traj_scaled.acceleration[i : i + 1],
    #             jerk=traj_scaled.jerk[i : i + 1],
    #         ),
    #         title=f"Scaled Trajectory for Env {i}",
    #         n_frames=100,
    #     )
