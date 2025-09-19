"""Automate Codex runs for ART project generation."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict

from prompt_builder import USER_PROMPT_24, compose_prompt

CODEX_FLAGS = [
    "exec",
    "--full-auto",
]

DEFAULT_PROMPTS: Dict[str, str] = {
    "24": USER_PROMPT_24,
}


def run_codex(prompt_name: str, user_prompt: str, output_dir: Path) -> None:
    """Call Codex CLI with the composed prompt and write results to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)

    final_prompt = compose_prompt(user_prompt)
    prompt_file = output_dir / f"{prompt_name}_prompt.txt"
    prompt_file.write_text(final_prompt)

    command = ["codex", *CODEX_FLAGS, final_prompt]
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    output_file = output_dir / f"{prompt_name}_output.txt"
    output_file.write_text(result.stdout)

    stderr_file = output_dir / f"{prompt_name}_stderr.txt"
    stderr_file.write_text(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Codex command failed for prompt '{prompt_name}' with exit code {result.returncode}."
        )


def run_default_prompts(output_dir: Path = Path("codex_runs")) -> None:
    for name, prompt in DEFAULT_PROMPTS.items():
        run_codex(name, prompt, output_dir)


if __name__ == "__main__":
    run_default_prompts()
