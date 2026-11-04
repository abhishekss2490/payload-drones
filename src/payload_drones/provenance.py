"""Git provenance required before running project code."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_clean_git() -> tuple[str, str]:
    """Return (commit, branch), refusing tracked or untracked changes."""
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(f"Refusing to run with a dirty git tree:\n{status}")
    return _git("rev-parse", "HEAD"), _git("rev-parse", "--abbrev-ref", "HEAD")


def tag_run(task_id: str, run_name: str) -> None:
    """Tag the current commit locally; duplicate tags fail loudly."""
    _git("tag", f"runs/{task_id}/{run_name}")

