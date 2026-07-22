"""Phase-1 and Phase-2 configurations for payload transport RMA."""

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from payload_drones.rl.configs import (
    MARLDistillationRunnerCfg,
    MARLStudentModelCfg,
    MARLTeacherModelCfg,
    TeamLatentDistillationAlgorithmCfg,
)


# Rollout length is memory-bound, not just a learning hyperparameter:
# RolloutStorage allocates every observation group for
# num_steps_per_env * num_envs transitions.
#
# Phase 1 stores policy(114) + agent_dynamics(45) + critic(170) +
# payload_trajectory(1200) = 1529 dims. At the default 4096 envs:
#   1529 * 4096 * 256 * 4 B = 5.98 GiB -- fits alongside the sim on 15.6 GiB.
#
# Phase 2 additionally stores agent_history(6300), so 7829 dims. The same 256
# steps would need 30.6 GiB, so the rollout is shortened. Distillation is
# supervised regression onto the frozen teacher's latent and does not need long
# rollouts, so this costs nothing:
#   7829 * 4096 * 48 * 4 B = 5.74 GiB.
NUM_STEPS_PER_ENV = 256
NUM_STEPS_PER_ENV_PHASE2 = 48
# init_std=0.3 is the base repo's value for a single free-flying quadrotor,
# where thrashing merely wobbles the airframe. Here it spans 0.16-1.00 x max
# thrust within 2 sigma, independently per motor per quad, on a cable-coupled
# three-body pendulum -- it excites the swing mode immediately. Measured effect:
# minimum inter-quad separation collapsed 0.520 -> 0.223 m by step ~50 while the
# mean *increased*, i.e. the team splays and converges rather than translating,
# and quad_collision then dominated terminations 40:1 over every other cause.
# 0.15 keeps 2-sigma thrust within 0.30-0.72 x max, around the ~0.49 hover point.
_SHARED_STD = {
    "class_name": "payload_drones.rl.models.SharedAgentGaussianDistribution",
    "init_std": 0.15,
    "std_type": "scalar",
}


def payload_rma_phase1_cfg() -> RslRlOnPolicyRunnerCfg:
    actor = MARLTeacherModelCfg(obs_normalization=False, distribution_cfg=_SHARED_STD.copy())
    critic = MARLTeacherModelCfg(obs_normalization=False)
    return RslRlOnPolicyRunnerCfg(
        seed=4551, num_steps_per_env=NUM_STEPS_PER_ENV, max_iterations=600, save_interval=100,
        experiment_name="payload_rma_phase1", logger="wandb", wandb_project="payload-drones",
        clip_actions=1.0, actor=actor, critic=critic,
        algorithm=RslRlPpoAlgorithmCfg(
            # The policy distribution is joint over three agents, and its KL
            # is the sum of their per-agent KLs.  Use a conservative update and
            # make the configured KL target active; Tyro exposes every field as
            # ``--agent.algorithm.*`` for the aggressive-PPO ablation.
            num_learning_epochs=5, num_mini_batches=16, learning_rate=1e-3, schedule="adaptive",
            gamma=0.995, lam=0.95, entropy_coef=0.0, desired_kl=0.015, max_grad_norm=1.0,
            clip_param=0.2,
        ),
        obs_groups={
            "actor": ("policy", "payload_trajectory", "agent_dynamics"),
            "critic": ("critic",),
        },
    )


def payload_rma_phase2_cfg() -> MARLDistillationRunnerCfg:
    student = MARLStudentModelCfg(obs_normalization=False, distribution_cfg=_SHARED_STD.copy())
    teacher = MARLTeacherModelCfg(obs_normalization=False, distribution_cfg=_SHARED_STD.copy())
    return MARLDistillationRunnerCfg(
        seed=4551, num_steps_per_env=NUM_STEPS_PER_ENV_PHASE2, max_iterations=500, save_interval=50,
        experiment_name="payload_rma_phase2", logger="wandb", wandb_project="payload-drones",
        clip_actions=1.0, resume=True, student=student, teacher=teacher,
        algorithm=TeamLatentDistillationAlgorithmCfg(
            loss_type="mse", learning_rate=1e-3, num_learning_epochs=1, gradient_length=15,
            max_grad_norm=1.0,
        ),
        obs_groups={
            "student": ("policy", "payload_trajectory", "agent_history"),
            "teacher": ("policy", "payload_trajectory", "agent_dynamics"),
        },
    )
