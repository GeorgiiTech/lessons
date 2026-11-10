"""Runs every lesson.py and solution.py as a script (quick sizes, CPU or the local GPU)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p for d in (ROOT / "lessons").iterdir() if d.is_dir()
                 for p in (d / "lesson.py", d / "solution.py") if p.exists())


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_script_runs(script: Path) -> None:
    env = {**os.environ, "LESSONS_QUICK": "1"}
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def test_gyms_are_unsolved() -> None:
    """Exercises must ship with TODOs, otherwise there's nothing to practise."""
    for gym in (ROOT / "lessons").glob("*/gym.py"):
        assert "TODO" in gym.read_text(), f"{gym} has no TODO"
