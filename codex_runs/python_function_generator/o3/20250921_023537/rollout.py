"""ART rollout logic for the *function-generation* environment.

Each rollout presents a single `FunctionTask` (see `env.py`) to the model and
evaluates the returned function body.  A shaped reward provides smooth signal:

    reward = syntax_ok*0.3 + signature_ok*0.2 + test_pass_ratio*0.5

Parsing / execution errors are captured via scalar metadata and mirrored into
`trajectory.metrics` so that downstream dashboards can aggregate them easily.
"""

from __future__ import annotations

import ast
import builtins
import textwrap
from types import ModuleType
from typing import Any

import art
import requests
import weave
from openai import AsyncOpenAI

from env import RANDOM_SEED, pick_task, run_tests, verify_signature, TRAINING_CONFIG

# Weighting constants kept here for convenience – tweak as desired.
_SYNTAX_WEIGHT = 0.3
_SIGNATURE_WEIGHT = 0.2
_TEST_WEIGHT = 0.5


def _strip_code_fences(payload: str) -> str:
    """Remove common markdown code-fence noise from *payload*."""

    if payload.strip().startswith("```"):
        payload = payload.strip().lstrip("` ").split("```", maxsplit=1)[-1]
    return textwrap.dedent(payload).strip()


# ---------------------------------------------------------------------------
# Rollout entry-point --------------------------------------------------------
# ---------------------------------------------------------------------------


@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:  # noqa: D401 – signature mandated by prompt
    # Select deterministic task for *step*.
    task = pick_task(step)

    # Build conversation.
    system_msg = {
        "role": "system",
        "content": (
            "You are a meticulous Python expert. When given a function "
            "requirement and signature, respond ONLY with the full function "
            "definition in plain Python. Do NOT wrap the code in markdown or "
            "add commentary. Adhere to PEP 8."
        ),
    }
    user_msg = {"role": "user", "content": task.rendered_prompt()}

    trajectory = art.Trajectory(
        messages_and_choices=[system_msg, user_msg],
        metadata={
            "task_name": task.name,
            "step": step,
        },
        reward=0.0,
    )

    # Run inference.
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    chat_completion = await client.chat.completions.create(
        messages=trajectory.messages(),
        model=model.name,
        max_completion_tokens=TRAINING_CONFIG["max_completion_tokens"],
        temperature=TRAINING_CONFIG["temperature"],
        top_p=TRAINING_CONFIG["top_p"],
        stream=False,
    )

    assistant_choice = chat_completion.choices[0]
    content_raw = assistant_choice.message.content or ""
    trajectory.messages_and_choices.append(assistant_choice)

    # ---------------------------------------------------------------------
    # Validation & reward shaping
    # ---------------------------------------------------------------------

    syntax_ok = 0.0
    signature_ok = 0.0
    test_pass_ratio = 0.0

    invalid_code_flag = 0.0
    signature_mismatch_flag = 0.0
    tests_failed_flag = 0.0

    # 1) Strip potential markdown.
    content = _strip_code_fences(content_raw)

    # 2) Attempt to parse.
    try:
        tree = ast.parse(content)
    except SyntaxError:
        invalid_code_flag = 1.0
    else:
        # Extract first function definition.
        func_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if func_defs:
            syntax_ok = 1.0

            # Compile & exec inside isolated module.
            module = ModuleType("agent_impl")
            module.__dict__["__builtins__"] = builtins.__dict__  # allow std built-ins.
            try:
                exec(compile(tree, filename="<agent>", mode="exec"), module.__dict__)
            except Exception:
                invalid_code_flag = 1.0
            else:
                candidate = getattr(module, func_defs[0].name, None)
                if callable(candidate):
                    # 3) Signature check.
                    if verify_signature(candidate, task):
                        signature_ok = 1.0
                    else:
                        signature_mismatch_flag = 1.0

                    # 4) Run tests.
                    passed, total = run_tests(candidate, task)
                    if total:
                        test_pass_ratio = passed / total
                        if passed < total:
                            tests_failed_flag = 1.0
                else:
                    invalid_code_flag = 1.0
        else:
            invalid_code_flag = 1.0

    # Reward calculation.
    reward = (
        syntax_ok * _SYNTAX_WEIGHT
        + signature_ok * _SIGNATURE_WEIGHT
        + test_pass_ratio * _TEST_WEIGHT
    )

    # Store scalar metadata + metrics.
    trajectory.metadata["syntax_error"] = invalid_code_flag
    trajectory.metadata["signature_error"] = signature_mismatch_flag
    trajectory.metadata["tests_failed"] = tests_failed_flag

    trajectory.metrics["syntax_ok"] = syntax_ok
    trajectory.metrics["signature_ok"] = signature_ok
    trajectory.metrics["test_pass_ratio"] = test_pass_ratio
    trajectory.reward = reward

    return trajectory
