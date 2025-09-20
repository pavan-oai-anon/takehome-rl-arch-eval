"""Environment helpers for a minimal JSON-extraction ART task.

This package defines:
- constants and training hyperparameters surfaced via TRAINING_CONFIG
- a small bank of unstructured examples with ground-truth labels
- utilities to build prompts, parse/validate model output, and score rewards

Notes
- Inference and training are assumed to run on ART's LocalBackend.
- If you need to install dependencies locally, prefer uv, e.g.:
  `uv pip install openpipe-art weave openai requests`.
"""
from __future__ import annotations

import json
import random
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


# ----------------------------
# Task constants (easy to tweak)
# ----------------------------
RANDOM_SEED: int = 1337

# Minimal training config consumed by the host training loop.
# Keep values small for fast iteration; adjust as needed.
TRAINING_CONFIG: dict[str, Any] = {
    "project": "json-extract-mini",
    "model_name": "agent-jsonex-001",
    "base_model": "Qwen/Qwen2.5-1.5B",
    "steps": 24,
    "trajectories_per_group": 16,
    "groups_per_step": 1,
    "learning_rate": 1e-5,
    "max_completion_tokens": 160,
    "temperature": 0.3,
    "top_p": 0.9,
    "max_exceptions": 32,
    # Local cleanup behavior mirrors the 2048 example ergonomics.
    "cleanup_keep_last": True,
}


# ----------------------------
# Schema and seeded examples
# ----------------------------
SCHEMA_DESCRIPTION: str = (
    "Return only a valid minified JSON object with fields: "
    '{"invoice_id": string, "customer": string, "date": "YYYY-MM-DD", '
    '"total": number, "currency": string, "items_count": integer}. '
    "Do not include markdown or commentary."
)

Field = Dict[str, Any]
Example = Dict[str, Any]


EXAMPLES: list[Example] = [
    {
        "id": "ex-001",
        "text": (
            "Invoice INV-1001 for Acme Corp was issued on March 5, 2024. "
            "Amount due: USD $1,234.50 for 3 items."
        ),
        "label": {
            "invoice_id": "INV-1001",
            "customer": "Acme Corp",
            "date": "2024-03-05",
            "total": 1234.50,
            "currency": "USD",
            "items_count": 3,
        },
    },
    {
        "id": "ex-002",
        "text": (
            "ACME Billing Notice — ID=INV-1002 (EUR). Issued 2024-04-17 to Globex. "
            "Total €2,999.00 with 5 line items."
        ),
        "label": {
            "invoice_id": "INV-1002",
            "customer": "Globex",
            "date": "2024-04-17",
            "total": 2999.00,
            "currency": "EUR",
            "items_count": 5,
        },
    },
    {
        "id": "ex-003",
        "text": (
            "Receipt ref INV-1003 to Initech on 17 Apr 2024. GBP 420.75 due; "
            "items count: 2."
        ),
        "label": {
            "invoice_id": "INV-1003",
            "customer": "Initech",
            "date": "2024-04-17",
            "total": 420.75,
            "currency": "GBP",
            "items_count": 2,
        },
    },
    {
        "id": "ex-004",
        "text": (
            "Invoice# INV-1004 | Customer: Umbrella Co. | 2024/05/02 | JPY ¥58,000 | "
            "4 items"
        ),
        "label": {
            "invoice_id": "INV-1004",
            "customer": "Umbrella Co.",
            "date": "2024-05-02",
            "total": 58000.0,
            "currency": "JPY",
            "items_count": 4,
        },
    },
    {
        "id": "ex-005",
        "text": (
            "Reminder: Payment for INV-1005 (Contoso Ltd). Issued 05/10/2024. Amount: $75.00 USD. "
            "Items: 1."
        ),
        "label": {
            "invoice_id": "INV-1005",
            "customer": "Contoso Ltd",
            "date": "2024-05-10",
            "total": 75.00,
            "currency": "USD",
            "items_count": 1,
        },
    },
    {
        "id": "ex-006",
        "text": (
            "Statement INV-1006 for Wayne Enterprises dated 10-06-2024. Total due is AUD 1,050.00; "
            "there are 6 items listed."
        ),
        "label": {
            "invoice_id": "INV-1006",
            "customer": "Wayne Enterprises",
            "date": "2024-06-10",
            "total": 1050.0,
            "currency": "AUD",
            "items_count": 6,
        },
    },
    {
        "id": "ex-007",
        "text": (
            "Billing doc INV-1007 sent to Stark Industries on June 11, 2024. "
            "Amount: CAD 12,345.67 for 8 items."
        ),
        "label": {
            "invoice_id": "INV-1007",
            "customer": "Stark Industries",
            "date": "2024-06-11",
            "total": 12345.67,
            "currency": "CAD",
            "items_count": 8,
        },
    },
    {
        "id": "ex-008",
        "text": (
            "INV-1008 | MegaCorp | 2024-06-15 | CHF 980.40 | items=2"
        ),
        "label": {
            "invoice_id": "INV-1008",
            "customer": "MegaCorp",
            "date": "2024-06-15",
            "total": 980.40,
            "currency": "CHF",
            "items_count": 2,
        },
    },
    {
        "id": "ex-009",
        "text": (
            "Customer: Soylent Corp — invoice INV-1009 dated 15/07/2024. "
            "Total payable: 2,250.00 MXN for 5 items."
        ),
        "label": {
            "invoice_id": "INV-1009",
            "customer": "Soylent Corp",
            "date": "2024-07-15",
            "total": 2250.0,
            "currency": "MXN",
            "items_count": 5,
        },
    },
    {
        "id": "ex-010",
        "text": (
            "Invoice INV-1010 for Black Mesa Research Facility (ZAR). Issued Jul 20, 2024. "
            "Amount ZAR 7,700.00; 7 items."
        ),
        "label": {
            "invoice_id": "INV-1010",
            "customer": "Black Mesa Research Facility",
            "date": "2024-07-20",
            "total": 7700.0,
            "currency": "ZAR",
            "items_count": 7,
        },
    },
    {
        "id": "ex-011",
        "text": (
            "Notice: INV-1011 to Aperture Science on 2024.08.01 with total of NOK 1 240,50. Items: 3."
        ),
        "label": {
            "invoice_id": "INV-1011",
            "customer": "Aperture Science",
            "date": "2024-08-01",
            "total": 1240.50,
            "currency": "NOK",
            "items_count": 3,
        },
    },
    {
        "id": "ex-012",
        "text": (
            "Doc INV-1012; buyer: Oceanic Airlines; date=2024/08/15; amount=SGD 2,100.00; items: 4"
        ),
        "label": {
            "invoice_id": "INV-1012",
            "customer": "Oceanic Airlines",
            "date": "2024-08-15",
            "total": 2100.0,
            "currency": "SGD",
            "items_count": 4,
        },
    },
    {
        "id": "ex-013",
        "text": (
            "Invoice INV-1013 (BRL) to Hooli on 09-01-2024, totaling R$ 3.300,00 for 9 items."
        ),
        "label": {
            "invoice_id": "INV-1013",
            "customer": "Hooli",
            "date": "2024-09-01",
            "total": 3300.0,
            "currency": "BRL",
            "items_count": 9,
        },
    },
    {
        "id": "ex-014",
        "text": (
            "INV-1014 issued to Tyrell Corporation on September 2, 2024. "
            "Total due: 1,500.25 USD; items=2."
        ),
        "label": {
            "invoice_id": "INV-1014",
            "customer": "Tyrell Corporation",
            "date": "2024-09-02",
            "total": 1500.25,
            "currency": "USD",
            "items_count": 2,
        },
    },
    {
        "id": "ex-015",
        "text": (
            "Record INV-1015 to Cyberdyne Systems — 2024-09-20 — KRW ₩850,000 — 10 items."
        ),
        "label": {
            "invoice_id": "INV-1015",
            "customer": "Cyberdyne Systems",
            "date": "2024-09-20",
            "total": 850000.0,
            "currency": "KRW",
            "items_count": 10,
        },
    },
]


# ----------------------------
# Helper functions
# ----------------------------
def get_rng(step: int) -> random.Random:
    """Deterministic RNG per step for stable sampling."""
    return random.Random(RANDOM_SEED + int(step))


def pick_example(step: int, rng: random.Random) -> tuple[int, Example]:
    """Pick an example index deterministically from `EXAMPLES`.

    Uses step to create a stable cycle with minor randomness.
    """
    if not EXAMPLES:
        raise RuntimeError("No seeded examples available")
    base = step % len(EXAMPLES)
    jitter = rng.randint(0, max(0, len(EXAMPLES) - 1))
    idx = (base + jitter) % len(EXAMPLES)
    return idx, EXAMPLES[idx]


def build_user_prompt(example: Example) -> str:
    """Construct a concise user prompt for the extraction task."""
    return (
        "Extract the requested fields from the text.\n"
        f"Schema: {SCHEMA_DESCRIPTION}\n"
        "Text:\n" + example["text"]
    )


_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*|```$", re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text).strip()


def extract_json_object(text: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Attempt to parse a JSON object from arbitrary model output.

    Returns (payload, error). `payload` is None when parsing fails.
    """
    raw = _strip_code_fences(text)
    # First try: direct parse
    try:
        val = json.loads(raw)
        if isinstance(val, dict):
            return val, None
        return None, "not_a_object"
    except Exception:
        pass

    # Second try: bracket slicing from first '{' to last '}'
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            val = json.loads(snippet)
            if isinstance(val, dict):
                return val, None
            return None, "not_a_object"
        except Exception as exc:  # pragma: no cover - defensive
            return None, f"json_parse_error:{type(exc).__name__}"

    return None, "no_braces_found"


_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y.%m.%d",
    "%d-%m-%Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d %b %Y",
    "%d %B %Y",
    "%Y.%m.%d",
)


def normalize_date(value: Any) -> Optional[str]:
    """Normalize many date formats to YYYY-MM-DD; return None on failure."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    # Remove ordinal suffixes: 1st, 2nd, 3rd, 4th
    s = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", s)
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        # Remove currency symbols and thousands separators
        s = s.replace(",", "").replace(" ", "")
        s = re.sub(r"[A-Za-z$€£¥₩R$]", "", s)
        try:
            return float(s)
        except Exception:
            return None
    return None


def _to_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):  # avoid True/False as ints
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        s = value.strip()
        s = re.sub(r"[^0-9-]", "", s)
        try:
            return int(s)
        except Exception:
            return None
    return None


def compute_reward(pred: Dict[str, Any], truth: Dict[str, Any]) -> Tuple[float, Dict[str, float], Optional[str]]:
    """Score a prediction against ground truth with dense shaping.

    Reward components (0..1 each):
    - valid_json: 1 if object present (caller should set); included here for completeness
    - coverage: fraction of required fields present
    - type_score: fraction of fields coercible to the right type
    - value_score: average per-field correctness; total uses smooth relative error

    Final reward = 0.2*valid_json + 0.3*coverage + 0.2*type_score + 0.3*value_score
    """
    required = ("invoice_id", "customer", "date", "total", "currency", "items_count")

    present = {k: (k in pred) for k in required}
    coverage = sum(1.0 for k in required if present[k]) / len(required)

    # Coerce and normalize
    invoice_id = pred.get("invoice_id") if present["invoice_id"] else None
    customer = pred.get("customer") if present["customer"] else None
    date_norm = normalize_date(pred.get("date")) if present["date"] else None
    total_val = _to_float(pred.get("total")) if present["total"] else None
    currency = str(pred.get("currency")).strip().upper() if present["currency"] else None
    items_count = _to_int(pred.get("items_count")) if present["items_count"] else None

    type_ok = {
        "invoice_id": isinstance(invoice_id, str) and bool(invoice_id.strip()),
        "customer": isinstance(customer, str) and bool(customer.strip()),
        "date": isinstance(date_norm, str),
        "total": isinstance(total_val, float),
        "currency": isinstance(currency, str) and len(currency) >= 3,
        "items_count": isinstance(items_count, int),
    }
    type_score = sum(1.0 for v in type_ok.values() if v) / len(required)

    # Truth values
    t_id = truth["invoice_id"]
    t_customer = truth["customer"]
    t_date = truth["date"]
    t_total = float(truth["total"])  # already canonical
    t_curr = truth["currency"].upper()
    t_items = int(truth["items_count"])

    # Per-field correctness (0/1), except total which is smooth
    eq_id = 1.0 if (type_ok["invoice_id"] and str(invoice_id).strip() == t_id) else 0.0
    eq_customer = 1.0 if (type_ok["customer"] and str(customer).strip() == t_customer) else 0.0
    eq_date = 1.0 if (type_ok["date"] and date_norm == t_date) else 0.0
    eq_curr = 1.0 if (type_ok["currency"] and currency == t_curr) else 0.0
    eq_items = 1.0 if (type_ok["items_count"] and items_count == t_items) else 0.0

    # Smooth score for total based on relative error
    if type_ok["total"]:
        rel_err = abs(total_val - t_total) / max(1.0, abs(t_total))  # 0 good, >0 bad
        total_score = max(0.0, 1.0 - min(1.0, rel_err * 3.0))  # tolerant but informative
    else:
        rel_err = 1.0
        total_score = 0.0

    value_score = (eq_id + eq_customer + eq_date + total_score + eq_curr + eq_items) / 6.0

    # valid_json should be set by caller; we infer from coverage>0 as a weak proxy
    valid_json = 1.0 if coverage > 0 else 0.0
    exact_match = 1.0 if (eq_id == eq_customer == eq_date == eq_curr == eq_items == 1.0 and total_score == 1.0) else 0.0

    reward = 0.2 * valid_json + 0.3 * coverage + 0.2 * type_score + 0.3 * value_score

    metrics = {
        "valid_json": float(valid_json),
        "field_coverage": float(coverage),
        "type_score": float(type_score),
        "value_score": float(value_score),
        "exact_match": float(exact_match),
        "total_rel_error": float(rel_err),
    }

    # Validation error summary string for metadata (scalar-only constraint)
    missing = [k for k in required if not present[k]]
    wrong_types = [k for k, ok in type_ok.items() if not ok and present.get(k, False)]
    valerr: Optional[str]
    if missing:
        valerr = f"missing:{','.join(missing)[:80]}"
    elif wrong_types:
        valerr = f"types:{','.join(wrong_types)[:80]}"
    else:
        valerr = None

    return reward, metrics, valerr


# ----------------------------
# System prompt (concise)
# ----------------------------
SYSTEM_PROMPT: str = (
    "You are a precise information extraction model. "
    "Extract the requested fields and answer with only a valid JSON object. "
    "Use exact schema and types; no extra keys, no commentary."
)

