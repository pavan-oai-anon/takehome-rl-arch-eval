"""Automate Codex runs for ART project generation."""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict

from prompt_builder import USER_PROMPT_24, compose_prompt

CODEX_FLAGS = [
    "exec",
    "--full-auto",
]

BUILT_IN_PROMPTS: Dict[str, str] = {
    "24": USER_PROMPT_24,
}

USER_PROMPTS_DIR = Path("user_prompts")


def run_codex(prompt_name: str, user_prompt: str, output_root: Path) -> Path:
    """Call Codex CLI with the composed prompt and write results to an isolated folder."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / prompt_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    final_prompt = compose_prompt(user_prompt)
    (run_dir / "prompt.txt").write_text(final_prompt)

    command = ["codex", *CODEX_FLAGS, final_prompt]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        cwd=run_dir,
    )

    (run_dir / "stdout.txt").write_text(result.stdout)
    (run_dir / "stderr.txt").write_text(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Codex command failed for prompt '{prompt_name}' with exit code {result.returncode}."
        )

    return run_dir


def load_user_prompt(prompt_name: str) -> str:
    """Fetch a prompt string from user_prompts/<name>.txt."""

    prompt_path = USER_PROMPTS_DIR / f"{prompt_name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt '{prompt_name}' not found in built-ins or {USER_PROMPTS_DIR}/"
        )
    return prompt_path.read_text().strip()


def resolve_prompt(prompt_name: str) -> str:
    """Return the prompt text for the requested task name."""

    if prompt_name in BUILT_IN_PROMPTS:
        return BUILT_IN_PROMPTS[prompt_name]
    return load_user_prompt(prompt_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Codex with a selected ART prompt")
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Task prompt name (built-in key or user_prompts/<name>.txt). Defaults to all built-ins.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("codex_runs"),
        help="Directory where Codex outputs will be stored",
    )
    return parser.parse_args()


def run_prompts(prompt_names: list[str], output_dir: Path) -> None:
    for name in prompt_names:
        prompt_text = resolve_prompt(name)
        run_dir = run_codex(name, prompt_text, output_dir)
        print(f"{name} task generated in {run_dir}")


def main() -> None:
    args = parse_args()

    if args.prompt is None:
        run_prompts(list(BUILT_IN_PROMPTS), args.output_dir)
    else:
        run_prompts([args.prompt], args.output_dir)


if __name__ == "__main__":
    main()
