"""Project training launcher with reproducible W&B provenance."""

import os
import sys
from importlib import import_module
from pathlib import Path

from payload_drones.provenance import REPO_ROOT, require_clean_git, tag_run


def main() -> None:
    import mjlab
    import mjlab.tasks
    import tyro
    import wandb
    from mjlab.scripts._cli import maybe_print_top_level_help
    from mjlab.scripts.train import TrainConfig, run_train
    from mjlab.tasks.registry import list_tasks
    from mjlab.utils.gpu import select_gpus

    import_module("payload_drones.config")

    maybe_print_top_level_help("train")
    project_tasks = [
        task for task in list_tasks() if task.startswith("Mjlab-PayloadTransport-")
    ]
    task_id, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(project_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    args = tyro.cli(
        TrainConfig,
        args=remaining,
        default=TrainConfig.from_task(task_id),
        prog=sys.argv[0] + f" {task_id}",
        config=mjlab.TYRO_FLAGS,
    )

    commit, branch = require_clean_git()
    selected_gpus, num_gpus = select_gpus(args.gpu_ids)
    if num_gpus > 1:
        raise ValueError("payload_drones.train supports one local GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "" if selected_gpus is None else ",".join(map(str, selected_gpus))
    )

    notes = f"commit: {commit}\nbranch: {branch}"
    run = wandb.init(
        project=args.agent.wandb_project,
        name=None,
        group=task_id,
        notes=notes,
        dir=REPO_ROOT,
    )
    run_name = run.name or run.id
    if not run_name:
        run.finish(exit_code=1)
        raise RuntimeError("W&B did not assign a run name or id")
    args.agent.run_name = run_name
    args.agent.logger = {  # type: ignore[assignment]
        "class_name": "WandbLogWriter",
        "project_name": args.agent.wandb_project,
        "run_name": run_name,
        "group": task_id,
        "notes": notes,
        "tags": list(args.agent.wandb_tags),
    }
    try:
        tag_run(task_id, run_name)
        log_root = (
            REPO_ROOT / "logs"
            if args.log_root == "logs/rsl_rl"
            else Path(args.log_root)
        )
        run_train(task_id, args, log_root.resolve() / task_id / run_name)
    except BaseException:
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()

