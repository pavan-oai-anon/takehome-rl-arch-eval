"""Automate Codex runs for ART project generation."""
from __future__ import annotations

import subprocess
from datetime import datetime
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


def run_default_prompts(output_dir: Path = Path("codex_runs")) -> None:
    for name, prompt in DEFAULT_PROMPTS.items():
        run_dir = run_codex(name, prompt, output_dir)
        print(f"{name} task generated in {run_dir}")


if __name__ == "__main__":
    run_default_prompts()
