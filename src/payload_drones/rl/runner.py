"""Training runners and checkpoint persistence for payload-drone RMA."""

from __future__ import annotations

import os

import torch
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from rsl_rl.env import VecEnv
from rsl_rl.runners import DistillationRunner


def _save_trajectory_curriculum(unwrapped_env) -> dict | None:
    """Serialize the trajectory command term's curriculum state, if present."""
    try:
        term = unwrapped_env.command_manager.get_term("trajectory")
    except KeyError:
        return None
    state_fn = getattr(term, "state_dict", None)
    return state_fn() if callable(state_fn) else None


def _load_trajectory_curriculum(unwrapped_env, state: dict | None) -> None:
    """Restore the trajectory command term's curriculum state, if both exist."""
    if not state:
        return
    try:
        term = unwrapped_env.command_manager.get_term("trajectory")
    except KeyError:
        return
    load_fn = getattr(term, "load_state_dict", None)
    if callable(load_fn):
        load_fn(state)


class MARLPhase1Runner(MjlabOnPolicyRunner):
    """Phase-1 PPO runner that additionally persists the trajectory curriculum.

    Mirrors ``MjlabOnPolicyRunner.save`` but adds ``trajectory_curriculum`` to
    ``env_state``; ``load`` defers to the parent (preserving legacy-checkpoint
    migration and ``common_step_counter`` restore) then restores the curriculum.
    """

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        # Keep an iteration equal to the actual PPO rollout length, including CLI overrides.
        term = env.unwrapped.command_manager.get_term("payload_pose")
        term.cfg.steps_per_iteration = train_cfg["num_steps_per_env"]
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path: str, infos=None) -> None:
        env_state = {
            "common_step_counter": self.env.unwrapped.common_step_counter,
            "trajectory_curriculum": _save_trajectory_curriculum(self.env.unwrapped),
        }
        infos = {**(infos or {}), "env_state": env_state}
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        if self.cfg["upload_model"]:
            self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)
        if load_cfg is None or load_cfg.get("iteration", False):
            self.current_learning_iteration += 1
        if infos and "env_state" in infos:
            _load_trajectory_curriculum(
                self.env.unwrapped, infos["env_state"].get("trajectory_curriculum")
            )
            # The wrapper generated its first references before checkpoint loading.
            # Rebuild them at the restored cap instead of resuming on the initial cap.
            term = self.env.unwrapped.command_manager.get_term("payload_pose")
            if term.cfg.speed_anneal_iterations > 0:
                self.env.reset()
        return infos


class MARLDistillationRunner(DistillationRunner):
    """Distillation runner that persists environment state across checkpoints.

    Mirrors MjlabOnPolicyRunner but for the student-teacher distillation loop:
    strips None-valued optional configs from student/teacher keys and
    saves/restores the common_step_counter for curriculum persistence.
    """

    env: RslRlVecEnvWrapper

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        for key in ("student", "teacher"):
            if key in train_cfg:
                for opt in ("cnn_cfg", "distribution_cfg"):
                    if train_cfg[key].get(opt) is None:
                        train_cfg[key].pop(opt, None)
                if train_cfg[key].get("rnn_type") is None:
                    for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                        train_cfg[key].pop(opt, None)
        super().__init__(env, train_cfg, log_dir, device)

    def export_policy_to_onnx(
        self,
        path: str,
        filename: str = "policy.onnx",
        verbose: bool = False,
    ) -> None:
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()
        os.makedirs(path, exist_ok=True)
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,
            output_names=onnx_model.output_names,
            dynamic_axes={},
            dynamo=False,
        )

    def save(self, path: str, infos=None) -> None:
        env_state = {
            "common_step_counter": self.env.unwrapped.common_step_counter,
            "trajectory_curriculum": _save_trajectory_curriculum(self.env.unwrapped),
        }
        infos = {**(infos or {}), "env_state": env_state}
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        if self.cfg.get("upload_model", True):
            self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

        # Migrate rsl-rl 4.x actor keys to 5.x distribution keys
        actor_sd = loaded_dict.get("actor_state_dict", {})
        if "std" in actor_sd:
            actor_sd["distribution.std_param"] = actor_sd.pop("std")
        if "log_std" in actor_sd:
            actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"] + 1

        infos = loaded_dict.get("infos", {})
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
            _load_trajectory_curriculum(
                self.env.unwrapped, infos["env_state"].get("trajectory_curriculum")
            )
        return infos
