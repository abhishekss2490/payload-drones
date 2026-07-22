"""Payload-task runner names and checkpoint compatibility with base RMA."""

from payload_drones.vendor.rl.runner import MjlabDistillationRunner, MjlabRMAPhase1Runner


class MARLPhase1Runner(MjlabRMAPhase1Runner):
    """Phase-1 PPO runner for the folded teacher policy."""


class MARLDistillationRunner(MjlabDistillationRunner):
    """Phase-2 runner for ego-paired team latent distillation."""
