"""Project evaluation launcher guarded by git provenance."""

import sys
from importlib import import_module

from payload_drones.provenance import REPO_ROOT, require_clean_git


def main() -> None:
    import mjlab
    import mjlab.tasks
    import tyro
    from mjlab.scripts._cli import maybe_print_top_level_help
    from mjlab.scripts.play import PlayConfig, run_play
    from mjlab.tasks.registry import list_tasks

    import_module("payload_drones.config")

    maybe_print_top_level_help("play")
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
        PlayConfig,
        args=remaining,
        default=PlayConfig(log_root=str(REPO_ROOT / "logs")),
        prog=sys.argv[0] + f" {task_id}",
        config=mjlab.TYRO_FLAGS,
    )
    require_clean_git()
    run_play(task_id, args)


if __name__ == "__main__":
    main()

