"""Configuration dataclasses for folded payload-transport RMA."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlBaseRunnerCfg, RslRlModelCfg


@dataclass
class MARLTeacherModelCfg(RslRlModelCfg):
    class_name: str = "payload_drones.rl.models.MARLTeacherModel"
    hidden_dims: tuple[int, ...] = (32, 32)
    activation: str = "elu"
    dyn_encoder_dims: tuple[int, ...] = (32, 16)
    dyn_latent_dim: int = 8
    traj_latent_dim: int = 16
    traj_conv_channels: tuple[int, ...] = (16, 8)
    traj_kernel_sizes: tuple[int, ...] = (5, 3)
    traj_strides: tuple[int, ...] = (2, 2)


@dataclass
class MARLStudentModelCfg(RslRlModelCfg):
    class_name: str = "payload_drones.rl.models.MARLStudentModel"
    hidden_dims: tuple[int, ...] = (32, 32)
    activation: str = "elu"
    tcn_channels: tuple[int, ...] = (32, 16)
    tcn_kernel_size: int = 4
    history_length: int = 50
    dyn_latent_dim: int = 8
    traj_latent_dim: int = 16
    traj_conv_channels: tuple[int, ...] = (16, 8)
    traj_kernel_sizes: tuple[int, ...] = (5, 3)
    traj_strides: tuple[int, ...] = (2, 2)


@dataclass
class DynamicsDistillationAlgorithmCfg:
    class_name: str = "robotics_common.learning.DynamicsDistillation"
    loss_type: str = "mse"
    learning_rate: float = 1e-3
    num_learning_epochs: int = 1
    gradient_length: int = 15
    max_grad_norm: float | None = 1.0
    optimizer: str = "adam"


@dataclass
class MARLDistillationRunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "DistillationRunner"
    student: MARLStudentModelCfg = field(default_factory=MARLStudentModelCfg)
    teacher: MARLTeacherModelCfg = field(default_factory=MARLTeacherModelCfg)
    algorithm: DynamicsDistillationAlgorithmCfg = field(
        default_factory=DynamicsDistillationAlgorithmCfg
    )
