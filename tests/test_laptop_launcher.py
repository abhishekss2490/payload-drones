"""Launcher contracts: forwarding, resume, diagnostics, and failure propagation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "laptop-train"
PIPELINE = "Mjlab-PayloadTransport-RMA"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="Linux/WSL bash required")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project = tmp_path / "project with spaces"
    (project / "scripts").mkdir(parents=True)
    shutil.copyfile(SOURCE, project / "scripts/laptop-train")
    python = project / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        """#!/usr/bin/env python3
import json, os, sys
if sys.argv[1:] == ["-"]:
    sys.stdin.read()
    print("CUDA preflight")
    sys.exit(int(os.environ.get("PREFLIGHT_EXIT", "0")))
print("TRAIN_ARGS=" + json.dumps(sys.argv[1:]))
print("TRAIN_CWD=" + os.getcwd())
print("WANDB_PATHS=" + json.dumps({key: os.environ.get(key) for key in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR", "WANDB_NAME")}))
print("trainer stderr", file=sys.stderr)
sys.exit(int(os.environ.get("TRAIN_EXIT", "0")))
"""
    )
    python.chmod(0o755)
    return project


def run(project: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    for key in ("LAPTOP_NUM_ENVS", "PREFLIGHT_EXIT", "TRAIN_EXIT", "WANDB_NAME", "WANDB_MODE"):
        environment.pop(key, None)
    environment.update(env)
    return subprocess.run(
        ["bash", str(project / "scripts/laptop-train"), *args],
        cwd=project.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )


def forwarded(result: subprocess.CompletedProcess[str]) -> list[str]:
    line = next(line for line in result.stdout.splitlines() if line.startswith("TRAIN_ARGS="))
    return json.loads(line.removeprefix("TRAIN_ARGS="))


def test_defaults_forwarding_and_console_log(project: Path):
    extra = ["--log-root", "logs/custom folder", "--phases.teacher.agent.seed", "123"]
    result = run(project, "baseline", PIPELINE, *extra)
    assert result.returncode == 0, result.stderr
    assert forwarded(result) == [
        "-m",
        "payload_drones.train",
        PIPELINE,
        "--phases.teacher.env.scene.num-envs",
        "1024",
        "--phases.distill.env.scene.num-envs",
        "1024",
        *extra,
    ]
    assert f"TRAIN_CWD={project}" in result.stdout
    (log,) = (project / "logs/custom folder/console").glob("*.log")
    assert "trainer stderr" in log.read_text()
    assert "CUDA preflight" in log.read_text()
    paths = json.loads(next(line.removeprefix("WANDB_PATHS=") for line in result.stdout.splitlines() if line.startswith("WANDB_PATHS=")))
    root = project / "logs/custom folder"
    assert paths["WANDB_DIR"] == str(root)
    assert all(Path(paths[key]).is_relative_to(root) for key in ("WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR"))


@pytest.mark.parametrize("resume", [["--resume", "old-run"], ["--resume=old-run"]])
def test_resume_does_not_inject_phase_overrides(project: Path, resume: list[str]):
    result = run(project, "continued", PIPELINE, *resume, LAPTOP_NUM_ENVS="invalid")
    assert result.returncode == 0
    assert forwarded(result) == ["-m", "payload_drones.train", PIPELINE, "--log-root", "logs/laptop", *resume]


def test_explicit_phase_sizes_take_precedence(project: Path):
    extra = ["--phases.teacher.env.scene.num-envs=128", "--phases.distill.env.scene.num-envs", "64"]
    result = run(project, "custom", PIPELINE, *extra, LAPTOP_NUM_ENVS="512")
    assert forwarded(result) == ["-m", "payload_drones.train", PIPELINE, "--log-root", "logs/laptop", *extra]


def test_default_applies_only_to_unspecified_phase(project: Path):
    extra = ["--phases.teacher.env.scene.num-envs", "128"]
    result = run(project, "custom", PIPELINE, *extra, LAPTOP_NUM_ENVS="512")
    assert forwarded(result) == [
        "-m",
        "payload_drones.train",
        PIPELINE,
        "--log-root",
        "logs/laptop",
        "--phases.distill.env.scene.num-envs",
        "512",
        *extra,
    ]


@pytest.mark.parametrize("variable,code", [("PREFLIGHT_EXIT", "7"), ("TRAIN_EXIT", "23")])
def test_failures_propagate_through_console_logging(project: Path, variable: str, code: str):
    result = run(project, "failed", PIPELINE, **{variable: code})
    assert result.returncode == int(code)
    if variable == "PREFLIGHT_EXIT":
        assert "TRAIN_ARGS=" not in result.stdout


def test_dry_run_does_not_start_python_or_create_logs(project: Path):
    result = run(project, "--dry-run", "preview", PIPELINE)
    assert result.returncode == 0
    assert "--phases.teacher.env.scene.num-envs 1024" in result.stdout
    assert "CUDA preflight" not in result.stdout
    assert "TRAIN_ARGS=" not in result.stdout
    assert not (project / "logs").exists()


def test_training_help_bypasses_cuda_and_defaults(project: Path):
    result = run(project, "help", PIPELINE, "--help", PREFLIGHT_EXIT="7")
    assert result.returncode == 0
    assert forwarded(result) == ["-m", "payload_drones.train", PIPELINE, "--help"]
    assert not (project / "logs").exists()


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_invalid_environment_count_fails_before_training(project: Path, value: str):
    result = run(project, "bad", PIPELINE, LAPTOP_NUM_ENVS=value)
    assert result.returncode == 2
    assert "positive integer" in result.stderr
    assert not (project / "logs").exists()


def test_missing_environment_has_actionable_error(project: Path):
    (project / ".venv/bin/python").unlink()
    result = run(project, "missing", PIPELINE)
    assert result.returncode == 2
    assert "uv sync --locked" in result.stderr


@pytest.mark.parametrize("absolute", [False, True])
def test_all_outputs_follow_equals_log_root(project: Path, absolute: bool):
    root = project / "outputs with spaces" if absolute else Path("outputs with spaces")
    result = run(project, "custom", PIPELINE, f"--log-root={root}")
    assert result.returncode == 0
    resolved = root if absolute else project / root
    assert len(list((resolved / "console").glob("*.log"))) == 1
    assert not (project / "logs").exists()
    assert forwarded(result).count(f"--log-root={root}") == 1
    assert "--log-root" not in forwarded(result)


@pytest.mark.parametrize("args", [["--log-root"], ["--log-root="], ["--log-root", "--resume", "old-run"]])
def test_invalid_log_root_fails_before_training(project: Path, args: list[str]):
    result = run(project, "bad", PIPELINE, *args)
    assert result.returncode == 2
    assert "--log-root requires" in result.stderr
    assert not (project / "logs").exists()


def test_offline_logging_gets_a_run_name(project: Path):
    result = run(project, "local", PIPELINE, WANDB_MODE="offline")
    assert result.returncode == 0
    paths = json.loads(next(line.removeprefix("WANDB_PATHS=") for line in result.stdout.splitlines() if line.startswith("WANDB_PATHS=")))
    assert paths["WANDB_NAME"].startswith("laptop-local-")
    assert paths["WANDB_DIR"] == str(project / "logs/laptop")


@pytest.mark.parametrize("directory", [None, "/tmp/laptop-results"])
def test_entry_point_honors_wandb_directory(monkeypatch, directory):
    from payload_drones import train

    monkeypatch.delenv("WANDB_DIR", raising=False)
    if directory:
        monkeypatch.setenv("WANDB_DIR", directory)
    calls = []
    monkeypatch.setattr(train, "train_pipeline", lambda *args, **kwargs: calls.append(kwargs))
    train.main()
    assert calls[0].get("dir") == directory
