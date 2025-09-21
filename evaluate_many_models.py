#!/usr/bin/env python3
"""Run evaluate_model.py on multiple codex runs concurrently."""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import List

DEFAULT_PARALLEL = 2
DEFAULT_TIMEOUT = None  # seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluator for multiple codex run folders")
    parser.add_argument(
        "runs",
        nargs="+",
        help="Codex run folders (e.g. codex_runs/task/model/timestamp) to evaluate",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL,
        help="Number of evaluate_model.py processes to run concurrently (default: 2)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Optional timeout in seconds for each evaluate_model.py invocation",
    )
    parser.add_argument(
        "--runs-per-eval",
        type=int,
        default=10,
        help="Number of evaluation loops per model (passed to evaluate_model.py --runs)",
    )
    parser.add_argument(
        "--python",
        default="python3",
        help="Python interpreter to use (default: python3)",
    )
    parser.add_argument(
        "--script",
        default="evaluate_model.py",
        help="Path to evaluate_model.py (default: evaluate_model.py in CWD)",
    )
    parser.add_argument(
        "--output-name",
        default="evaluation_report.json",
        help="Filename for the evaluation report inside each run folder (default: evaluation_report.json)",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass through to evaluate_model.py (use -- before these flags)",
    )
    return parser.parse_args()


def build_command(python: str, script: str, run_path: str, report_name: str, runs_per_eval: int, extra_args: List[str]) -> list[str]:
    command = [python, script, run_path, "--runs", str(runs_per_eval), "--output", report_name]
    if extra_args:
        if extra_args[0] == "--":
            extra_args = extra_args[1:]
        command.extend(extra_args)
    return command


def main() -> None:
    args = parse_args()
    run_paths = [Path(run).resolve() for run in args.runs]
    for run in run_paths:
        if not run.exists():
            raise FileNotFoundError(f"Run path {run} does not exist")

    extra_args = args.extra_args or []
    max_parallel = max(1, args.parallel)

    queue = list(run_paths)
    active: List[dict] = []
    overall_status = 0

    while queue or active:
        while queue and len(active) < max_parallel:
            run_path = queue.pop(0)
            output_path = run_path / args.output_name
            command = build_command(
                args.python,
                args.script,
                str(run_path),
                str(output_path),
                args.runs_per_eval,
                extra_args,
            )
            print(f"[launch] {' '.join(map(str, command))}")
            process = subprocess.Popen(command)
            active.append(
                {
                    "run": run_path,
                    "process": process,
                    "start": time.time(),
                    "command": command,
                }
            )

        if not active:
            break

        time.sleep(1)
        for job in list(active):
            process: subprocess.Popen = job["process"]  # type: ignore[assignment]
            run_path: Path = job["run"]  # type: ignore[assignment]
            command = job["command"]  # type: ignore[assignment]
            retcode = process.poll()
            if retcode is None:
                if args.timeout and (time.time() - job["start"]) > args.timeout:
                    print(f"[timeout] {run_path} exceeded {args.timeout}s, terminating")
                    process.kill()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                    retcode = 124
                else:
                    continue

            if retcode == 0:
                print(f"[done] {run_path} -> success")
            else:
                print(f"[fail] {run_path} -> exit {retcode}")
                if overall_status == 0:
                    overall_status = retcode
            active.remove(job)

    if overall_status != 0:
        print(f"[summary] One or more evaluations failed (exit {overall_status})")
    else:
        print("[summary] All evaluations completed successfully")

    raise SystemExit(overall_status)


if __name__ == "__main__":
    main()
