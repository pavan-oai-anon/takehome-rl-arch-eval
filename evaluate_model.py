#!/usr/bin/env python3
"""Generate and run an automatic evaluation for a PEFT model using OpenAI."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from openai import OpenAI
from unsloth import FastLanguageModel
from dotenv import load_dotenv
from tenacity import retry
load_dotenv()

@dataclass
class EvaluationPlan:
    system_message: str
    user_message: str
    evaluation_instructions: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PEFT model using GPT-5 generated tests")
    parser.add_argument(
        "model_paths",
        nargs="+",
        help=(
            "One or more PEFT checkpoint directories or Codex run folders (e.g. codex_runs/task/model/timestamp)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the evaluation JSON report (default: codex run folder /evaluation_report.json)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to sample from the model (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the PEFT model (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling for the PEFT model (default: 0.9)",
    )
    parser.add_argument(
        "--gpt5-model",
        default="gpt-5",
        help="Model name for the GPT-5 planner (default: gpt-5)",
    )
    parser.add_argument(
        "--evaluator-model",
        default="gpt-4.1",
        help="Model name for the code-interpreter evaluation step (default: gpt-4.1)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of evaluation runs to perform (default: 10)",
    )
    return parser.parse_args()


def load_user_prompt(path: Path) -> str:
    return path.read_text().strip()


def collect_response_text(resp: Any) -> str:
    pieces: list[str] = []
    for item in getattr(resp, "output", []):
        for part in getattr(item, "content", []):
            if getattr(part, "type", None) == "text":
                pieces.append(getattr(part, "text", ""))
    if not pieces and hasattr(resp, "output_text"):
        return resp.output_text
    return "".join(pieces)


def request_evaluation_plan(client: OpenAI, gpt5_model: str, user_prompt: str) -> EvaluationPlan:
    instructions = (
        "You are designing an evaluation for a fine-tuned model. "
        "Using the provided task description, craft a single test case. "
        "Return JSON with keys system_message, user_message, evaluation_instructions, notes."
        "Note: the provided task description was the prompt that was used to create the RL environment that the model was GRPOed on."
        "You should design a question that would be in line with the task description and the model was trying to solve - not creating the RL environment itself."
        f"Also - do remember that the model that we are evaluating is relatively small, so the question should be relatively simple."
    )
    resp = client.responses.create(
        model=gpt5_model,
        instructions=instructions,
        input=user_prompt,
    )
    text = collect_response_text(resp)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"GPT-5 plan was not valid JSON: {text}") from exc

    required_keys = {"system_message", "user_message", "evaluation_instructions", "notes"}
    missing = required_keys - data.keys()
    if missing:
        raise RuntimeError(f"Evaluation plan JSON missing keys: {missing}")
    return EvaluationPlan(
        system_message=data["system_message"],
        user_message=data["user_message"],
        evaluation_instructions=data["evaluation_instructions"],
        notes=data["notes"],
    )


def load_peft_model(model_path: Path) -> tuple[Any, Any, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    peft_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=16384,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        load_in_4bit=torch.cuda.is_available(),
    )
    FastLanguageModel.for_inference(peft_model)
    return peft_model, tokenizer, device


def generate_with_model(
    model,
    tokenizer,
    device: str,
    system_message: str,
    user_message: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    generated = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)
    return generated.strip()

@retry
def evaluate_with_code_interpreter(
    client: OpenAI,
    evaluator_model: str,
    plan: EvaluationPlan,
    model_response: str,
) -> Dict[str, Any]:
    instructions = (
        "You are a meticulous evaluator. Use the python code tool when helpful to check correctness. "
        "Return a JSON object with keys score (0-1.0) non-integer scores are allowed, reasoning, passed (boolean)."
    )
    evaluation_prompt = (
        "Task description notes:\n" + plan.notes + "\n\n"
        "System message provided to model:\n" + plan.system_message + "\n\n"
        "User input sent to model:\n" + plan.user_message + "\n\n"
        "Model response:\n" + model_response + "\n\n"
        "Evaluation instructions:\n" + plan.evaluation_instructions + "\n"
    )
    resp = client.responses.create(
        model=evaluator_model,
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        instructions=instructions,
        input=evaluation_prompt,
    )
    text = collect_response_text(resp)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Evaluator did not return valid JSON: {text}") from exc
    return data


def resolve_model_checkpoint(model_input: Path) -> tuple[Path, str]:
    if (model_input / "config.json").exists():  # direct checkpoint directory
        parts = model_input.parts
        if "models" in parts:
            idx = parts.index("models")
            task = parts[idx - 1] if idx > 0 else "unknown"
        else:
            task = "unknown"
        return model_input, task

    run_parts = model_input.parts
    # Expect codex_runs/<task>/<model>/<run_timestamp>
    if len(run_parts) >= 4 and run_parts[0] == "codex_runs":
        task = run_parts[1]
        requested_model = run_parts[2]
        run_timestamp = run_parts[3]
    elif len(run_parts) >= 3:
        task, requested_model, run_timestamp = run_parts[-3:]
    else:
        raise ValueError(
            f"Cannot infer task/model/timestamp from path {model_input}. Provide a checkpoint directory or codex run path."
        )

    art_root = Path(".art")
    models_root = art_root / task / "models"
    if not models_root.exists():
        raise FileNotFoundError(f"Models directory {models_root} not found")

    # Select best matching model directory
    candidates = [d for d in models_root.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No model directories found under {models_root}")

    prioritized: list[Path] = []
    if run_timestamp:
        prioritized.extend([d for d in candidates if run_timestamp in d.name])
    if not prioritized and requested_model:
        prioritized.extend([d for d in candidates if requested_model in d.name])
    if not prioritized:
        prioritized = candidates

    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for d in prioritized:
        if d not in seen:
            ordered.append(d)
            seen.add(d)

    chosen_model_dir = ordered[0]

    checkpoint_root = chosen_model_dir / "checkpoints"
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_root} not found")

    # Determine specific checkpoint folder
    checkpoints = sorted(
        [p for p in checkpoint_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {checkpoint_root}")

    checkpoint_dir = checkpoints[0]

    return checkpoint_dir, task


def determine_output_path(run_input: Path, explicit_output: str | None) -> Path:
    if explicit_output:
        return Path(explicit_output).resolve()

    try:
        relative = run_input.relative_to(Path("codex_runs"))
        report_dir = Path("codex_runs") / relative
    except ValueError:
        report_dir = run_input

    report_dir.mkdir(parents=True, exist_ok=True)
    return (report_dir / "evaluation_report.json").resolve()


def main() -> None:
    args = parse_args()
    input_paths = [Path(p).resolve() for p in args.model_paths]
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Model path {input_path} does not exist")

    resolved_models: list[tuple[Path, str, Path]] = []

    first_task = None
    for input_path in input_paths:
        model_path, task_name = resolve_model_checkpoint(input_path)
        if first_task is None:
            first_task = task_name
        elif task_name != first_task:
            raise ValueError(
                f"All models must target the same task. Expected '{first_task}', found '{task_name}' for {input_path}."
            )
        user_prompt_path = Path("user_prompts") / f"{task_name}.txt"
        if not user_prompt_path.exists():
            raise FileNotFoundError(f"User prompt {user_prompt_path} not found for inferred task '{task_name}'")
        output_path = determine_output_path(input_path, args.output)
        resolved_models.append((model_path, task_name, output_path))

    if not resolved_models:
        raise RuntimeError("No models resolved for evaluation")

    task_name = resolved_models[0][1]
    user_prompt_path = Path("user_prompts") / f"{task_name}.txt"
    user_prompt = load_user_prompt(user_prompt_path)
    user_prompt_path = Path("user_prompts") / f"{task_name}.txt"
    if not user_prompt_path.exists():
        raise FileNotFoundError(f"User prompt {user_prompt_path} not found for inferred task '{task_name}'")

    client = OpenAI()

    peft_models = [load_peft_model(model_path) for model_path, _, _ in resolved_models]

    shared_runs: list[dict[str, Any]] = []
    for run_idx in range(1, args.runs + 1):
        plan = request_evaluation_plan(client, args.gpt5_model, user_prompt)
        run_entry = {"plan": plan.__dict__, "models": []}

        for (model_path, _task, output_path), (peft_model, tokenizer, device) in zip(
            resolved_models, peft_models
        ):
            model_response = generate_with_model(
                peft_model,
                tokenizer,
                device,
                plan.system_message,
                plan.user_message,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
            )
            evaluation = evaluate_with_code_interpreter(
                client,
                args.evaluator_model,
                plan,
                model_response,
            )

            run_entry["models"].append(
                {
                    "model_path": str(model_path),
                    "output_path": str(output_path),
                    "model_response": model_response,
                    "evaluation": evaluation,
                }
            )
            print(
                f"Run {run_idx}/{args.runs} | {model_path.name}: score={evaluation.get('score')} passed={evaluation.get('passed')}"
            )

        shared_runs.append(run_entry)

    aggregate_reports = []
    for (model_path, _task, output_path) in resolved_models:
        model_entries = []
        for run in shared_runs:
            for model_entry in run["models"]:
                if model_entry["model_path"] == str(model_path):
                    entry = {
                        "plan": run["plan"],
                        "model_response": model_entry["model_response"],
                        "evaluation": model_entry["evaluation"],
                    }
                    model_entries.append(entry)

        scores = [float(entry["evaluation"].get("score", 0.0)) for entry in model_entries]
        passes = [bool(entry["evaluation"].get("passed", False)) for entry in model_entries]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        pass_rate = sum(passes) / len(passes) if passes else 0.0
        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0

        summary = {
            "task": task_name,
            "model_path": str(model_path),
            "user_prompt_path": str(user_prompt_path),
            "runs": model_entries,
            "aggregate": {
                "runs": len(model_entries),
                "average_score": avg_score,
                "pass_rate": pass_rate,
                "max_score": max_score,
                "min_score": min_score,
            },
        }

        Path(output_path).write_text(json.dumps(summary, indent=2))
        print(
            f"Evaluation written to {output_path} | avg_score={avg_score:.3f} | pass_rate={pass_rate:.2%}"
        )


if __name__ == "__main__":
    main()
