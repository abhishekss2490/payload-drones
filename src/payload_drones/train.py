"""Training entry point."""

import os

from robotics_common.mjlab.launch import Phase, train_pipeline

from payload_drones import config as _config  # noqa: F401  registers the tasks

PIPELINES = {
    "Mjlab-PayloadTransport-RMA": (
        Phase("teacher", "Mjlab-PayloadTransport-RMA-Phase1"),
        Phase("distill", "Mjlab-PayloadTransport-RMA-Phase2"),
    ),
}


def main() -> None:
    # The shared launcher passes an explicit W&B directory. Honor the laptop
    # launcher's WANDB_DIR so its local metrics stay with its checkpoints.
    wandb_kwargs = {"dir": os.environ["WANDB_DIR"]} if os.environ.get("WANDB_DIR") else {}
    train_pipeline(__file__, pipelines=PIPELINES, **wandb_kwargs)


if __name__ == "__main__":
    main()
