#!/usr/bin/env python3
"""Score Codex-generated environments against their rubrics using Codex CLI."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROMPT_TEMPLATE = """You are an automated reviewer scoring an OpenPipe ART environment.
Evaluate the provided files using the rubric and return JSON only.

Rubric (10 points total):
{rubric}

Project run: {run_path}

Respond with a JSON object containing:
- "total_points": number (0-10).
- "criteria": list of objects with keys "description", "max_points", "awarded_points", "justification".
- "notes": short string.

Do not include any markdown fences, prose, or explanations outside the JSON.

env.py:
```python
{env_code}
```

rollout.py:
```python
{rollout_code}
```
"""


def find_run_directories(root: Path) -> list[Path]:
    return [
        p
        for p in root.rglob("*")
        if p.is_dir() and (p / "env.py").exists() and (p / "rollout.py").exists()
    ]


def load_text(path: Path) -> str:
    return path.read_text().strip()


def build_prompt(run_path: Path, rubric_text: str, env_code: str, rollout_code: str) -> str:
    return PROMPT_TEMPLATE.format(
        rubric=rubric_text.strip(),
        run_path=str(run_path),
        env_code=env_code.strip(),
        rollout_code=rollout_code.strip(),
    )


def call_codex(prompt: str, model: str) -> str:
    command = [
        "codex",
        "exec",
        "--full-auto",
        "--model",
        model,
        prompt,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Codex command failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def extract_json(text: str) -> Any:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Codex output")
    return json.loads(match.group(0))


def score_run(root: Path, run_path: Path, rubric_root: Path, model: str, overwrite: bool) -> None:
    try:
        relative = run_path.relative_to(root)
        parts = relative.parts
        task_name = parts[0] if parts else run_path.name
    except ValueError:
        task_name = run_path.name
    rubric_path = rubric_root / f"{task_name}.txt_rubric"
    if not rubric_path.exists():
        print(f"[skip] No rubric for task '{task_name}' (expected {rubric_path})")
        return

    score_path = run_path / "rubric_score.json"
    raw_path = run_path / "rubric_score_raw.txt"
    if score_path.exists() and not overwrite:
        print(f"[skip] Score already exists for {run_path}")
        return

    env_code = load_text(run_path / "env.py")
    rollout_code = load_text(run_path / "rollout.py")
    rubric_text = load_text(rubric_path)

    prompt = build_prompt(run_path, rubric_text, env_code, rollout_code)
    try:
        raw_output = call_codex(prompt, model)
        data = extract_json(raw_output)
    except Exception as exc:
        print(f"[error] {run_path}: {exc}")
        raw_path.write_text(str(exc))
        return

    raw_path.write_text(raw_output)
    score_path.write_text(json.dumps(data, indent=2))
    print(f"[ok] Scored {run_path}: {data.get('total_points')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Codex runs using task rubrics via Codex CLI.")
    parser.add_argument("root", nargs="?", default="codex_runs", help="Root directory containing Codex run folders")
    parser.add_argument("--rubric-root", default="user_prompts", help="Directory holding *txt_rubric files")
    parser.add_argument("--model", default="gpt-5", help="Codex model to use (default: gpt-5)")
    parser.add_argument("--overwrite", action="store_true", help="Re-score runs even if rubric_score.json exists")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rubric_root = Path(args.rubric_root).resolve()

    if not root.exists():
        print(f"Root directory {root} does not exist", file=sys.stderr)
        sys.exit(1)

    run_dirs = find_run_directories(root)
    if not run_dirs:
        print(f"No run directories with env.py/rollout.py found under {root}")
        return

    for run_dir in sorted(run_dirs):
        score_run(root, run_dir, rubric_root, args.model, args.overwrite)


if __name__ == "__main__":
    main()
