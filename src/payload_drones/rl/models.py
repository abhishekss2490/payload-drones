"""Folded, parameter-shared RMA models for the three-quad payload task."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization, MLP, HiddenState
from rsl_rl.modules import GaussianDistribution
from rsl_rl.utils import unpad_trajectories

from mj_adapt_drones.rl.models import _squashed_forward
from mj_adapt_drones.rl.modules import CausalTCN, TrajectoryEncoder


N_AGENTS = 3
POLICY_DIM = 38
DYNAMICS_DIM = 15
HISTORY_LENGTH = 50
ACTION_DIM_PER_AGENT = 4


def fold_agents(x: torch.Tensor, per_agent_dim: int) -> torch.Tensor:
    """Fold ``(B, N * D)`` to ``(B * N, D)`` in env-major agent order."""
    if x.ndim != 2 or x.shape[-1] != N_AGENTS * per_agent_dim:
        raise ValueError(
            f"expected (B, {N_AGENTS * per_agent_dim}) agent tensor, got {tuple(x.shape)}"
        )
    return x.view(x.shape[0], N_AGENTS, per_agent_dim).reshape(-1, per_agent_dim)


def unfold_agents(x: torch.Tensor, per_agent_dim: int) -> torch.Tensor:
    """Unfold ``(B * N, D)`` to ``(B, N * D)`` in the inverse order."""
    if x.ndim != 2 or x.shape[-1] != per_agent_dim or x.shape[0] % N_AGENTS:
        raise ValueError(
            f"expected (B * {N_AGENTS}, {per_agent_dim}) folded agent tensor, got {tuple(x.shape)}"
        )
    return x.view(-1, N_AGENTS, per_agent_dim).reshape(-1, N_AGENTS * per_agent_dim)


class SharedAgentGaussianDistribution(GaussianDistribution):
    """Diagonal Gaussian with one four-motor std vector shared by all agents."""

    def __init__(self, output_dim: int, **kwargs) -> None:
        if output_dim % ACTION_DIM_PER_AGENT:
            raise ValueError(f"output_dim={output_dim} is not a multiple of {ACTION_DIM_PER_AGENT}")
        super().__init__(output_dim, **kwargs)
        if self.std_type != "scalar":
            raise ValueError("SharedAgentGaussianDistribution supports std_type='scalar' only")
        init_std = float(kwargs.get("init_std", 1.0))
        self.std_param = torch.nn.Parameter(torch.full((ACTION_DIM_PER_AGENT,), init_std))
        self.n_agents = output_dim // ACTION_DIM_PER_AGENT

    def _tiled_std(self) -> torch.Tensor:
        return self.std_param.clamp(self.std_range[0], self.std_range[1]).repeat(self.n_agents)

    def update(self, mlp_output: torch.Tensor) -> None:
        self._distribution = torch.distributions.Normal(mlp_output, self._tiled_std())

    @property
    def std(self) -> torch.Tensor:
        # Match rsl-rl's public API: batch-broadcasted action-space std.
        return self._distribution.stddev  # type: ignore[union-attr]


class _MARLModelBase(MLPModel):
    """Shared actor/centralized-critic dispatch and strict agent-axis helpers."""

    is_centralized_critic: bool

    def _init_layout(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> None:
        self.is_centralized_critic = tuple(obs_groups[obs_set]) == ("critic",)
        if self.is_centralized_critic:
            self.critic_dim = obs["critic"].shape[-1]
            return
        required = {"policy", "payload_trajectory"}
        missing = required - set(obs_groups[obs_set])
        if missing:
            raise ValueError(f"actor observation set is missing {sorted(missing)}")
        if obs["policy"].shape[-1] != N_AGENTS * POLICY_DIM:
            raise ValueError("policy observation does not contain three 38-dim ego blocks")
        if obs["payload_trajectory"].shape[-1] != 1200:
            raise ValueError("payload trajectory must be one shared 1200-dim tensor per environment")

    def _get_latent_dim(self) -> int:
        if self.is_centralized_critic:
            return self.critic_dim
        return POLICY_DIM + self.traj_latent_dim + self.dyn_latent_dim

    def _shared_trajectory_latent(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Encode exactly once per environment, then view-broadcast over agents."""
        z_traj = self.trajectory_encoder(trajectory)
        return z_traj.unsqueeze(1).expand(-1, N_AGENTS, -1).reshape(-1, self.traj_latent_dim)

    def _critic_latent(self, obs: TensorDict) -> torch.Tensor:
        return self.obs_normalizer(obs["critic"])

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        if self.is_centralized_critic:
            return MLPModel.forward(self, obs, masks, hidden_state, stochastic_output)
        # The shared MLP/distribution operate on one folded agent row and emit
        # four motors.  Restore the ordinary mjlab action boundary afterwards.
        actions = _squashed_forward(self, obs, masks, hidden_state, stochastic_output)
        return unfold_agents(actions, ACTION_DIM_PER_AGENT)

    @property
    def output_mean(self) -> torch.Tensor:
        if self.is_centralized_critic:
            return super().output_mean
        return unfold_agents(self.distribution.mean, ACTION_DIM_PER_AGENT)  # type: ignore[union-attr]

    @property
    def output_std(self) -> torch.Tensor:
        if self.is_centralized_critic:
            return super().output_std
        return unfold_agents(self.distribution.std, ACTION_DIM_PER_AGENT)  # type: ignore[union-attr]

    @property
    def output_entropy(self) -> torch.Tensor:
        if self.is_centralized_critic:
            return super().output_entropy
        return self.distribution.entropy.view(-1, N_AGENTS).sum(dim=-1)  # type: ignore[union-attr]

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        if self.is_centralized_critic:
            return super().output_distribution_params
        return tuple(
            unfold_agents(param, ACTION_DIM_PER_AGENT)
            for param in self.distribution.params  # type: ignore[union-attr]
        )

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.is_centralized_critic:
            return super().get_output_log_prob(outputs)
        log_prob = self.distribution.log_prob(fold_agents(outputs, ACTION_DIM_PER_AGENT))  # type: ignore[union-attr]
        return log_prob.view(-1, N_AGENTS).sum(dim=-1)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if self.is_centralized_critic:
            return super().get_kl_divergence(old_params, new_params)
        old_folded = tuple(fold_agents(param, ACTION_DIM_PER_AGENT) for param in old_params)
        new_folded = tuple(fold_agents(param, ACTION_DIM_PER_AGENT) for param in new_params)
        kl = self.distribution.kl_divergence(old_folded, new_folded)  # type: ignore[union-attr]
        return kl.view(-1, N_AGENTS).sum(dim=-1)


class MARLTeacherModel(_MARLModelBase):
    """Phase-1 teacher: shared actor plus a centralized un-folded critic."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        dyn_encoder_dims: tuple[int, ...] = (32, 16),
        dyn_latent_dim: int = 8,
        traj_latent_dim: int = 16,
        traj_conv_channels: tuple[int, ...] = (16, 8),
        traj_kernel_sizes: tuple[int, ...] = (5, 3),
        traj_strides: tuple[int, ...] = (2, 2),
        output_activation: str | None = None,
        hidden_dims: tuple[int, ...] | list[int] = (32, 32),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        if output_activation is not None:
            raise ValueError("payload actions use tanh in _squashed_forward; output_activation must be None")
        self.traj_latent_dim = traj_latent_dim
        self.dyn_latent_dim = dyn_latent_dim
        self._init_layout(obs, obs_groups, obs_set)
        super().__init__(
            obs, obs_groups, obs_set,
            1 if self.is_centralized_critic else ACTION_DIM_PER_AGENT,
            hidden_dims=hidden_dims, activation=activation,
            obs_normalization=obs_normalization, distribution_cfg=distribution_cfg,
        )
        if self.is_centralized_critic:
            return
        self.trajectory_encoder = TrajectoryEncoder(
            input_dim=obs["payload_trajectory"].shape[-1], output_dim=traj_latent_dim,
            conv_channels=traj_conv_channels, kernel_sizes=traj_kernel_sizes,
            strides=traj_strides, activation=activation,
        )
        self.dynamics_encoder = MLP(DYNAMICS_DIM, dyn_latent_dim, dyn_encoder_dims, activation=activation)
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self._get_latent_dim())

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        if self.is_centralized_critic:
            return self._critic_latent(obs)
        return self.obs_normalizer(self._raw_actor_latent(obs))

    def _raw_actor_latent(self, obs: TensorDict) -> torch.Tensor:
        policy = fold_agents(obs["policy"], POLICY_DIM)
        dynamics = fold_agents(obs["agent_dynamics"], DYNAMICS_DIM)
        z_traj = self._shared_trajectory_latent(obs["payload_trajectory"])
        z_dyn = self.dynamics_encoder(dynamics)
        return torch.cat((policy, z_traj, z_dyn), dim=-1)

    def get_dynamics_latent(self, obs: TensorDict) -> torch.Tensor:
        if self.is_centralized_critic:
            raise RuntimeError("centralized critic has no dynamics latent")
        return self.dynamics_encoder(fold_agents(obs["agent_dynamics"], DYNAMICS_DIM))

    def update_normalization(self, obs: TensorDict) -> None:
        if not self.obs_normalization:
            return
        with torch.no_grad():
            self.obs_normalizer.update(self._raw_actor_latent(obs))


class MARLStudentModel(_MARLModelBase):
    """Phase-2 student whose TCN sees one agent's own state-action history only."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        tcn_channels: tuple[int, ...] = (32, 16),
        tcn_kernel_size: int = 4,
        history_length: int = HISTORY_LENGTH,
        dyn_latent_dim: int = 8,
        traj_latent_dim: int = 16,
        traj_conv_channels: tuple[int, ...] = (16, 8),
        traj_kernel_sizes: tuple[int, ...] = (5, 3),
        traj_strides: tuple[int, ...] = (2, 2),
        output_activation: str | None = None,
        hidden_dims: tuple[int, ...] | list[int] = (32, 32),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        if output_activation is not None:
            raise ValueError("payload actions use tanh in _squashed_forward; output_activation must be None")
        self.traj_latent_dim = traj_latent_dim
        self.dyn_latent_dim = dyn_latent_dim
        self.history_length = history_length
        self.sa_dim = POLICY_DIM + ACTION_DIM_PER_AGENT
        self._init_layout(obs, obs_groups, obs_set)
        super().__init__(
            obs, obs_groups, obs_set,
            1 if self.is_centralized_critic else ACTION_DIM_PER_AGENT,
            hidden_dims=hidden_dims, activation=activation,
            obs_normalization=obs_normalization, distribution_cfg=distribution_cfg,
        )
        if self.is_centralized_critic:
            return
        expected_history_dim = N_AGENTS * history_length * self.sa_dim
        if obs["agent_history"].shape[-1] != expected_history_dim:
            raise ValueError(f"agent_history must have {expected_history_dim} values")
        self.trajectory_encoder = TrajectoryEncoder(
            input_dim=obs["payload_trajectory"].shape[-1], output_dim=traj_latent_dim,
            conv_channels=traj_conv_channels, kernel_sizes=traj_kernel_sizes,
            strides=traj_strides, activation=activation,
        )
        self.adaptation_module = CausalTCN(
            input_channels=self.sa_dim, hidden_channels=tcn_channels, kernel_size=tcn_kernel_size,
            output_dim=dyn_latent_dim, activation=activation,
        )
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self._get_latent_dim())

    def _fold_history(self, history: torch.Tensor) -> torch.Tensor:
        return fold_agents(history, self.history_length * self.sa_dim).view(
            -1, self.history_length, self.sa_dim
        )

    def get_adaptation_latent(self, obs: TensorDict) -> torch.Tensor:
        if self.is_centralized_critic:
            raise RuntimeError("centralized critic has no adaptation module")
        return self.adaptation_module(self._fold_history(obs["agent_history"]))

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        if self.is_centralized_critic:
            return self._critic_latent(obs)
        return self.obs_normalizer(self._raw_actor_latent(obs))

    def _raw_actor_latent(self, obs: TensorDict) -> torch.Tensor:
        policy = fold_agents(obs["policy"], POLICY_DIM)
        z_traj = self._shared_trajectory_latent(obs["payload_trajectory"])
        z_dyn_hat = self.get_adaptation_latent(obs)
        return torch.cat((policy, z_traj, z_dyn_hat), dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if not self.obs_normalization:
            return
        with torch.no_grad():
            self.obs_normalizer.update(self._raw_actor_latent(obs))
