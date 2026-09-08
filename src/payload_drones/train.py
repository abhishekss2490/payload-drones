"""Training entry point."""

from robotics_common.mjlab.launch import Phase, train_pipeline

from payload_drones import config as _config  # noqa: F401  registers the tasks


def main() -> None:
    train_pipeline(
        __file__,
        pipelines={
            "Mjlab-PayloadTransport-RMA": (
                Phase("teacher", "Mjlab-PayloadTransport-RMA-Phase1"),
                Phase("distill", "Mjlab-PayloadTransport-RMA-Phase2"),
            ),
        },
    )


if __name__ == "__main__":
    main()
