"""Automate Codex runs for ART project generation."""
from __future__ import annotations

import argparse
import multiprocessing
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
        dest="models",
        action="append",
        help="Codex model identifier (repeatable). Defaults to CODEX_MODEL env or 'default'.",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        help="OpenAI API key (overrides OPENAI_API_KEY env variable)",
    )
    return parser.parse_args()


def run_prompts(
    prompt_names: list[str],
    output_dir: Path,
    *,
    model: str | None,
    model_label: str,
    api_key: str,
) -> None:
    label = model_label or (model or "default")
    for name in prompt_names:
        prompt_text = resolve_prompt(name)
        run_dir = run_codex(
            name,
            prompt_text,
            output_dir,
            model=model,
            api_key=api_key,
        )
        print(f"[{label}] {name} task generated in {run_dir}")


def _run_prompts_worker(
    *,
    model_label: str,
    model_override: str | None,
    prompt_names: list[str],
    output_dir: Path,
    api_key: str,
) -> None:
    run_prompts(
        prompt_names,
        output_dir,
        model=model_override,
        model_label=model_label,
        api_key=api_key,
    )


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY must be set in the environment or via --api-key to authenticate Codex runs",
        )

    models: list[str] = []
    if args.models:
        models.extend(args.models)
    else:
        env_models = os.getenv("CODEX_MODEL")
        if env_models:
            models.extend(m.strip() for m in env_models.split(",") if m.strip())

    if not models:
        models = ["default"]

    prompts = list(BUILT_IN_PROMPTS) if args.prompt is None else [args.prompt]

    if len(models) == 1:
        model_name = models[0]
        model_override = None if model_name == "default" else model_name
        _run_prompts_worker(
            model_label=model_name,
            model_override=model_override,
            prompt_names=prompts,
            output_dir=args.output_dir,
            api_key=api_key,
        )
        return

    multiprocessing.set_start_method("spawn", force=True)

    processes: list[tuple[str, multiprocessing.Process]] = []
    for model_name in models:
        model_override = None if model_name == "default" else model_name
        proc = multiprocessing.Process(
            target=_run_prompts_worker,
            kwargs={
                "model_label": model_name,
                "model_override": model_override,
                "prompt_names": prompts,
                "output_dir": args.output_dir,
                "api_key": api_key,
            },
        )
        proc.start()
        processes.append((model_name, proc))

    failures: list[str] = []
    for model_name, proc in processes:
        proc.join()
        if proc.exitcode != 0:
            failures.append(f"{model_name} (exit {proc.exitcode})")

    if failures:
        raise RuntimeError("Codex runs failed for: " + ", ".join(failures))


if __name__ == "__main__":
    main()
