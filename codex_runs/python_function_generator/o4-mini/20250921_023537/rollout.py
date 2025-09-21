"""Rollout logic for Python function generation RL task."""

from __future__ import annotations
import ast
import random
from typing import Any, Dict

import art
import weave
from openai import AsyncOpenAI

from env import RANDOM_SEED, get_scenario

# GPU memory tuning can be applied as in 2048.py example if needed

@weave.op
@art.retry(exceptions=(Exception,))
async def rollout(
    model: art.Model, step: int, config: Dict[str, Any]
) -> art.Trajectory:
    """
    Generate a Python function per scenario, validate correctness, and compute reward.
    """
    # Initialize client for chat completions
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    # Deterministic scenario selection
    random.seed(RANDOM_SEED + step)
    scenario = get_scenario(step)

    # Build initial trajectory
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "You are a Python function generation model. Given a requirement and a "
                    "function signature with docstring, return a complete Python function "
                    "adhering to the signature and PEP8 style as plain code."
                ),
            }
        ],
        metadata={
            "scenario_id": scenario["id"],
            "step": step,
        },
        reward=0.0,
    )

    # User prompt with requirement and signature
    signature = f"def {scenario['function_name']}({', '.join(scenario['arg_names'])}):"
    user_content = (
        f"Requirement: {scenario['description']}\n"
        f"Signature:\n```python\n{signature}\n    {scenario['docstring']}\n```\n"
        "Provide only the complete function implementation."
    )
    trajectory.messages_and_choices.append({"role": "user", "content": user_content})

    # Invoke the model
    chat = await client.chat.completions.create(
        model=model.name,
        max_completion_tokens=config["max_completion_tokens"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        messages=trajectory.messages(),
        stream=False,
    )
    choice = chat.choices[0]
    content = choice.message.content
    trajectory.messages_and_choices.append({"role": choice.message.role, "content": content})

    # Validation and reward computation
    # Syntax check
    try:
        module = ast.parse(content)
    except SyntaxError:
        trajectory.metadata["syntax_error"] = 1
        trajectory.metrics["invalid_syntax"] = 1.0
        trajectory.reward = 0.0
        return trajectory

    # Function definition and signature check
    funcs = [n for n in module.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        trajectory.metadata["no_function_def"] = 1
        trajectory.metrics["invalid_signature"] = 1.0
        trajectory.reward = 0.0
        return trajectory
    func = funcs[0]
    name_valid = func.name == scenario["function_name"]
    arg_names = [arg.arg for arg in func.args.args]
    args_valid = arg_names == scenario["arg_names"]
    if not name_valid:
        trajectory.metadata["name_mismatch"] = 1
    if not args_valid:
        trajectory.metadata["args_mismatch"] = 1
    if not (name_valid and args_valid):
        trajectory.metrics["invalid_signature"] = 1.0
        trajectory.reward = 0.0
        return trajectory

    # Execute generated and golden code
    gen_ns: Dict[str, Any] = {}
    gold_ns: Dict[str, Any] = {}
    exec(content, gen_ns)
    exec(scenario["golden"], gold_ns)
    gen_fn = gen_ns[scenario["function_name"]]
    gold_fn = gold_ns[scenario["function_name"]]

    # Run test cases
    total = len(scenario["test_cases"])
    passed = 0
    for inp, expected in scenario["test_cases"]:
        try:
            result = gen_fn(*inp)
            if result == expected:
                passed += 1
        except Exception:
            continue
    failures = total - passed
    trajectory.metrics["test_failures"] = float(failures)
    trajectory.metrics["test_pass_rate"] = float(passed) / total

    # Reward: smooth pass rate
    trajectory.reward = float(passed) / total
    return trajectory
