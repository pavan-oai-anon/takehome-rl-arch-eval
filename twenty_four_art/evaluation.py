"""Evaluation helpers for the trained 24 agent."""
from __future__ import annotations

from pathlib import Path

import art
import torch
from unsloth import FastLanguageModel

from .config import TrainingConfig
from .env import (
    InvalidExpressionError,
    generate_game,
    render_puzzle,
    score_expression,
    validate_solution_expression,
)
from .rollout import SYSTEM_MESSAGE, extract_expression


def _checkpoint_path(model: art.TrainableModel, step: int) -> Path:
    return Path(f".art/{model.project}/models/{model.name}/checkpoints/{step:04d}")


async def evaluate_model(
    model: art.TrainableModel,
    config: TrainingConfig,
    *,
    samples: int = 2,
) -> None:
    """Reload LoRA weights and run inference rollouts."""

    step = await model.get_step()
    checkpoint_dir = _checkpoint_path(model, step)
    if not checkpoint_dir.exists():
        print("No checkpoints available for evaluation")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    peft_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_dir),
        max_seq_length=16384,
        dtype=torch.bfloat16,
        load_in_4bit=True,  # memory-saving tweak copied from the 2048 example
    )
    FastLanguageModel.for_inference(peft_model)
    peft_model.to(device)

    for attempt in range(samples):
        game = generate_game()
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": render_puzzle(game)},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)

        with torch.no_grad():
            outputs = peft_model.generate(
                input_ids=inputs,
                max_new_tokens=config.max_completion_tokens,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
            )

        response = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)
        print(f"\nAttempt {attempt + 1}")
        print(f"Digits: {game.digits}")
        print(f"Model response: {response}")

        try:
            expression = extract_expression(response)
            value = validate_solution_expression(expression, game.digits)
            metrics = score_expression(value, game.target)
            if metrics["difference"] == 0:
                print("Solved exactly 24!")
            else:
                print(
                    "Near miss",
                    f"value={metrics['value']:.2f}",
                    f"difference={metrics['difference']:.2f}",
                )
        except InvalidExpressionError as error:
            print(f"Invalid expression: {error}")
