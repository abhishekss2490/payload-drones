"""Cooperative aerial payload transport environment.

Task registration deliberately does NOT happen here yet.

This project registers through mjlab's own task registry
(``mjlab.tasks.registry.register_mjlab_task``), exposed via the
``[project.entry-points."mjlab.tasks"]`` hook in ``pyproject.toml`` and launched
as ``uv run train <task-id>`` — the same pattern as the base repo's
``mj_adapt_drones.config``. It is *not* a Gymnasium environment; mjlab does not
use Gymnasium and it is not installed in this venv.

``register_mjlab_task`` requires an ``rl_cfg`` (and optionally a ``runner_cls``),
which belong to the MARL model/training stage. Registration therefore lands in
``payload_drones/config/__init__.py`` once those exist. Until then the env is
constructed directly from ``payload_drones.payload_env_cfg.make_payload_env_cfg``.
"""
