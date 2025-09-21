"""Sequential launcher for running training.py on multiple Codex runs."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


DEFAULT_TIMEOUT_SECONDS = 600  # 10 minutes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch training.py sequentially in separate subprocesses with a fixed timeout. "
            "Each positional JOB argument accepts a comma-separated list of Codex run "
            "directories that will be passed to training.py in a single invocation."
        )
    )
    parser.add_argument(
        "jobs",
        nargs="+",
        help=(
            "Whitespace-separated job definitions. Within a job, provide one or more Codex run "
            "directories separated by commas (e.g. codex_runs/24/gpt-5/TS,codex_runs/json/gpt-5/TS)."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Timeout in seconds for each training.py invocation (default: 600).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use when launching training.py (default: current interpreter).",
    )
    parser.add_argument(
        "--training-args",
        default=[],
        nargs=argparse.REMAINDER,
        help=(
            "Optional additional arguments to append to every training.py invocation. "
            "Use -- before flags intended for training.py."
        ),
    )
    return parser.parse_args()


def build_command(python: str, job_paths: Sequence[str], extra_args: Sequence[str]) -> list[str]:
    command = [python, "training.py", *job_paths]
    command.extend(extra_args)
    return command


def run_job(
    job_index: int,
    job_paths: Sequence[str],
    *,
    python_exec: str,
    timeout: int,
    extra_args: Sequence[str],
) -> int:
    command = build_command(python_exec, job_paths, extra_args)
    display_command = " ".join(shlex.quote(part) for part in command)
    print(f"[job {job_index}] starting: {display_command}")

    start = time.time()
    try:
        result = subprocess.run(command, cwd=Path.cwd(), timeout=timeout)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(
            f"[job {job_index}] timed out after {elapsed:.1f}s (limit {timeout}s). Command was: {display_command}"
        )
        return 124  # mimic timeout exit code convention

    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"[job {job_index}] completed successfully in {elapsed:.1f}s")
    else:
        print(f"[job {job_index}] failed in {elapsed:.1f}s with exit code {result.returncode}")
    return result.returncode


def main() -> None:
    args = parse_args()

    extra_args = args.training_args

    overall_status = 0
    for idx, job in enumerate(args.jobs, start=1):
        job_paths = [segment.strip() for segment in job.split(",") if segment.strip()]
        if not job_paths:
            print(f"[job {idx}] skipped (no valid paths specified)")
            continue

        return_code = run_job(
            idx,
            job_paths,
            python_exec=args.python,
            timeout=args.timeout,
            extra_args=extra_args,
        )
        if return_code != 0 and overall_status == 0:
            overall_status = return_code

    sys.exit(overall_status)


if __name__ == "__main__":
    main()
