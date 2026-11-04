"""Run one independent episode for every phase of a trained pipeline."""

from robotics_common.mjlab.launch import evaluate_pipeline

from payload_drones.train import PIPELINES


def main() -> None:
    evaluate_pipeline(__file__, pipelines=PIPELINES)


if __name__ == "__main__":
    main()
