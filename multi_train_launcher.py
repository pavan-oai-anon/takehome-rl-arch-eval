"""Multi-launcher for running training.py across multiple Codex runs."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

DEFAULT_TIMEOUT_SECONDS = 600  # 10 minutes
DEFAULT_MAX_CONCURRENT = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch training.py in separate subprocesses with a fixed timeout. "
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
        "--parallel",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help="Maximum concurrent training jobs (default: 1; set 2 to run two at a time).",
    )
    parser.add_argument(
        "--training-args",
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


def main() -> None:
    args = parse_args()

    extra_args = args.training_args or []
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    jobs: list[tuple[int, list[str]]] = []
    for idx, job in enumerate(args.jobs, start=1):
        job_paths = [segment.strip() for segment in job.split(",") if segment.strip()]
        if not job_paths:
            print(f"[job {idx}] skipped (no valid paths specified)")
            continue
        jobs.append((idx, job_paths))

    overall_status = 0
    active: list[dict[str, object]] = []
    job_cursor = 0
    max_concurrent = max(1, args.parallel)

    while job_cursor < len(jobs) or active:
        while job_cursor < len(jobs) and len(active) < max_concurrent:
            idx, job_paths = jobs[job_cursor]
            job_cursor += 1

            command = build_command(args.python, job_paths, extra_args)
            display_command = " ".join(shlex.quote(part) for part in command)
            print(f"[job {idx}] starting: {display_command}")

            process = subprocess.Popen(command, cwd=Path.cwd())
            active.append(
                {
                    "index": idx,
                    "paths": job_paths,
                    "process": process,
                    "display": display_command,
                    "start": time.time(),
                }
            )

        if not active:
            break

        time.sleep(1)
        for job in list(active):
            process: subprocess.Popen = job["process"]  # type: ignore[assignment]
            idx = job["index"]  # type: ignore[assignment]
            display_command = job["display"]  # type: ignore[assignment]
            start_time = job["start"]  # type: ignore[assignment]
            elapsed = time.time() - start_time

            retcode = process.poll()
            if retcode is None:
                if elapsed > args.timeout:
                    print(
                        f"[job {idx}] timed out after {elapsed:.1f}s (limit {args.timeout}s). Command was: {display_command}"
                    )
                    process.kill()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                    active.remove(job)
                    if overall_status == 0:
                        overall_status = 124
                continue

            if retcode == 0:
                print(f"[job {idx}] completed successfully in {elapsed:.1f}s")
            else:
                print(f"[job {idx}] failed in {elapsed:.1f}s with exit code {retcode}")
                if overall_status == 0:
                    overall_status = retcode
            active.remove(job)

    sys.exit(overall_status)


if __name__ == "__main__":
    main()
