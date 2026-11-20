"""Lesson 18 · From notebook to project: run the package like a user would.

Run:  uv run python lessons/18-project/lesson.py
It calls the command line of the `fashion` package and its tests, exactly
as you would type them in a terminal from lessons/18-project/.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
QUICK = os.environ.get("LESSONS_QUICK") == "1"


def run(*args: str) -> None:
    print("$", " ".join(args))
    result = subprocess.run([sys.executable, *args], cwd=HERE,
                            capture_output=True, text=True)
    lines = (result.stdout + result.stderr).strip().splitlines()
    print("\n".join(line for line in lines if line.strip()))
    assert result.returncode == 0, "command failed"


if __name__ == "__main__":
    os.environ["PYTHONPATH"] = str(HERE)
    # %% [1] The help text comes for free from argparse.
    run("-m", "fashion.train", "--help")
    # %% [2] Training from the command line, settings as flags.
    n = "1000" if QUICK else "10000"
    run("-m", "fashion.train", "--epochs", "2" if QUICK else "3",
        "--n-train", n, "--data", "../../data", "--out", "../../runs/l18")
    # %% [3] The tests: seconds, no dataset needed.
    run("-m", "pytest", "tests", "-q", "-p", "no:cacheprovider")
