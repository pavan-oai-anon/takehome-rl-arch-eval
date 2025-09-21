"""ART rollout for Python function generation.

This rollout samples a small algorithmic utility task, asks the model to emit
exactly one Python function matching a provided signature, validates syntax and
signature, and evaluates functional equality against a golden implementation.

Rewards are shaped to be smooth: test pass rate is the main signal with bonuses
for valid syntax and matching signature; invalid outputs record metrics and
metadata strings for introspection. LocalBackend is assumed for inference.
"""
from __future__ import annotations

import ast
import math
from typing import Any

import art
import requests
import weave
from openai import AsyncOpenAI

from env import (
    RANDOM_SEED,
    TRAINING_CONFIG,
    problem_for_step,
    render_user_prompt,
    system_prompt,
    Signature,
)


def _strip_code_fences(text: str) -> str:
    """Remove common Markdown code fences and trim whitespace."""

    t = text.strip()
    if t.startswith("```"):
        # strip first fence line
        t = t.split("\n", 1)[-1]
    if t.endswith("```"):
        t = t.rsplit("\n", 1)[0]
    # Also strip optional leading "python" language hint
    if t.lower().startswith("python\n"):
        t = t.split("\n", 1)[-1]
    return t.strip()


def _extract_func_def(src: str) -> ast.FunctionDef | None:
    """Parse source and return the first top-level function definition, if any."""

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _signature_matches(node: ast.FunctionDef, sig: Signature) -> bool:
    """Check name and positional argument names match the expected signature.

    Type annotations and defaults are ignored for flexibility. Only the order
    and names of positional args are enforced, which is usually sufficient for
    unit-test comparability in this task bank.
    """

    if node.name != sig.name:
        return False
    got = tuple(a.arg for a in node.args.args)
    want = tuple(a.split(":", 1)[0].strip() for a in sig.args)
    return got == want


def _safe_exec_and_get(src: str, func_name: str) -> tuple[Any | None, str | None]:
    """Exec the provided source in a restricted global namespace and fetch the function.

    This is not a security boundary but reduces obvious hazards. We include
    common builtins and utility modules used by typical algorithmic solutions.
    """

    allowed_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "all": all,
        "any": any,
        "sorted": sorted,
        "map": map,
        "filter": filter,
        "zip": zip,
        "round": round,
        # constructors & basic types
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "print": print,  # harmless; does not affect return values
    }
    g: dict[str, Any] = {"__builtins__": allowed_builtins}
    # Provide common modules without requiring imports in the candidate code.
    import math as _math
    import re as _re
    import itertools as _itertools

    g.update({"math": _math, "re": _re, "itertools": _itertools})

    try:
        compiled = compile(src, filename="<candidate>", mode="exec")
        exec(compiled, g, None)
    except Exception as exc:  # pragma: no cover - defensive during RL
        return None, f"exec_error: {exc.__class__.__name__}: {exc}"
    func = g.get(func_name)
    if not callable(func):
        return None, "exec_ok_but_function_missing"
    return func, None


def _equal(a: Any, b: Any) -> bool:
    """Deep-ish equality with float tolerance for RL smoothness."""

    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
        except Exception:
            return a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _reward(pass_rate: float, syntax_ok: int, sig_ok: int, length_penalty: float) -> float:
    """Shaped reward balancing validity and utility.

    - pass_rate provides smooth signal (0..1)
    - syntax_ok and sig_ok add informative bonuses
    - length_penalty gently discourages overly long outputs
    """

    r = pass_rate + 0.25 * syntax_ok + 0.25 * sig_ok - length_penalty
    # Keep rewards within a tidy range for stability
    return max(-1.0, min(1.6, r))


@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(model: art.Model, step: int, config: dict[str, Any]) -> art.Trajectory:  # noqa: D401
    """Gather one function-generation trajectory and compute shaped reward."""

    rng_step = RANDOM_SEED + (step * 9973)
    problem = problem_for_step(step, seed=rng_step)

    # Prepare conversation
    sys_prompt = system_prompt()
    user_prompt = render_user_prompt(problem)

    traj = art.Trajectory(
        messages_and_choices=[{"role": "system", "content": sys_prompt}],
        metadata={
            "task": "py-function-gen",
            "problem_id": problem.pid,
            "problem_name": problem.name,
            "step": step,
        },
        reward=0.0,
    )
    traj.messages_and_choices.append({"role": "user", "content": user_prompt})

    client = AsyncOpenAI(base_url=model.inference_base_url, api_key=model.inference_api_key)
    completion = await client.chat.completions.create(
        model=model.name,
        messages=traj.messages(),
        max_completion_tokens=int(config.get("max_completion_tokens", TRAINING_CONFIG["max_completion_tokens"])),
        temperature=float(config.get("temperature", TRAINING_CONFIG["temperature"])),
        top_p=float(config.get("top_p", TRAINING_CONFIG["top_p"])),
        stream=False,
    )
    choice = completion.choices[0]
    content = choice.message.content or ""
    traj.messages_and_choices.append(choice)

    # Validation & evaluation
    raw = content
    cleaned = _strip_code_fences(raw)
    loc = cleaned.count("\n") + 1 if cleaned else 0
    chars = len(cleaned)
    too_long = 1 if loc > 240 or chars > 20000 else 0
    length_penalty = 0.05 if too_long else 0.0

    node = _extract_func_def(cleaned)
    syntax_ok = 1 if node is not None else 0
    sig_ok = 0
    err: str | None = None

    cand_func = None
    if node is not None:
        sig_ok = 1 if _signature_matches(node, problem.signature) else 0
        cand_func, err = _safe_exec_and_get(cleaned, problem.signature.name)
    else:
        err = "syntax_error"

    # Evaluate functional correctness
    total = len(problem.cases)
    passed = 0
    runtime_error = 0
    if cand_func is not None:
        for args in problem.cases:
            try:
                expected = problem.golden(*args)
                got = cand_func(*args)
                if _equal(got, expected):
                    passed += 1
            except Exception:
                runtime_error = 1
                # continue checking remaining cases for better gradient
    pass_rate = (passed / total) if total else 0.0

    invalid_solution = 1 if (syntax_ok == 0 or sig_ok == 0 or cand_func is None) else 0

    # Metrics and reward
    traj.metrics["syntax_ok"] = float(syntax_ok)
    traj.metrics["signature_ok"] = float(sig_ok)
    traj.metrics["tests_passed"] = float(passed)
    traj.metrics["tests_total"] = float(total)
    traj.metrics["pass_rate"] = float(pass_rate)
    traj.metrics["invalid_solution"] = float(invalid_solution)
    traj.metrics["runtime_error"] = float(runtime_error)
    traj.metrics["lines_of_code"] = float(loc)
    traj.metrics["chars"] = float(chars)

    traj.metadata["error_message"] = (err or "ok")[:240]
    traj.metadata["response_chars"] = chars
    traj.metadata["response_lines"] = loc

    traj.reward = _reward(pass_rate, syntax_ok, sig_ok, length_penalty)
    return traj


__all__ = ["rollout"]

