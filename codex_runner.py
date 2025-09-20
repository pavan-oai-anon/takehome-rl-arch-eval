"""Automate Codex runs for ART project generation."""
from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

from prompt_builder import USER_PROMPT_24, compose_prompt


CODEX_FLAGS = [
    "exec",
    "--full-auto",
]

BUILT_IN_PROMPTS: Dict[str, str] = {
    "24": USER_PROMPT_24,
}

USER_PROMPTS_DIR = Path("user_prompts")


def run_codex(
    prompt_name: str,
    user_prompt: str,
    output_root: Path,
    *,
    model: str | None,
    api_key: str,
) -> Path:
    """Call Codex CLI with the composed prompt and write results to an isolated folder."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model or "default"
    run_dir = output_root / prompt_name / model_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    final_prompt = compose_prompt(user_prompt)
    (run_dir / "prompt.txt").write_text(final_prompt)

    command = ["codex", *CODEX_FLAGS]
    if model:
        command.extend(["--model", model])
    command.extend(["--config", 'preferred_auth_method="apikey"'])
    command.append(final_prompt)

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    print("OpenAI API Key: ", api_key)

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        cwd=run_dir,
        env=env,
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
    parser.add_argument(
        "--model",
        help="Codex model identifier (overrides CODEX_MODEL env variable)",
        default=os.getenv("CODEX_MODEL"),
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        help="OpenAI API key (overrides OPENAI_API_KEY env variable)",
        default=os.getenv("OPENAI_API_KEY"),
    )
    return parser.parse_args()


def run_prompts(
    prompt_names: list[str],
    output_dir: Path,
    *,
    model: str | None,
    api_key: str,
) -> None:
    for name in prompt_names:
        prompt_text = resolve_prompt(name)
        run_dir = run_codex(
            name,
            prompt_text,
            output_dir,
            model=model,
            api_key=api_key,
        )
        print(f"{name} task generated in {run_dir}")


_cached_model: str | None = None
_cached_api_key: str | None = None


def resolve_api_key() -> str:
    """Fetch the OpenAI API key from environment variables."""

    global _cached_api_key
    if _cached_api_key is not None:
        return _cached_api_key

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY must be set in the environment or .env to authenticate Codex runs",
        )
    _cached_api_key = api_key
    return api_key


def resolve_model() -> str | None:
    """Pick the Codex model to use, if specified."""

    global _cached_model
    if _cached_model is not None:
        return _cached_model

    model = os.getenv("CODEX_MODEL")
    _cached_model = model
    return model


def main() -> None:
    load_dotenv()
    args = parse_args()

    global _cached_api_key
    global _cached_model

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
        _cached_api_key = None
    if args.model:
        os.environ["CODEX_MODEL"] = args.model
        _cached_model = None

    model = resolve_model()
    api_key = resolve_api_key()

    prompts = list(BUILT_IN_PROMPTS) if args.prompt is None else [args.prompt]
    run_prompts(prompts, args.output_dir, model=model, api_key=api_key)


if __name__ == "__main__":
    main()
