from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

RANDOM_SEED = 137

TRAINING_CONFIG: dict[str, Any] = {
    "project": "json-extraction",
    "model_name": "structured-json-agent",
    "base_model": "meta-llama/Llama-3-8b",
    "steps": 60,
    "trajectories_per_group": 4,
    "groups_per_step": 3,
    "learning_rate": 2e-5,
    "max_completion_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.9,
    "max_exceptions": 2,
    "cleanup_keep_last": 2,
}

REQUIRED_FIELDS: tuple[str, ...] = (
    "title",
    "author",
    "published_date",
    "category",
    "body",
)

SYSTEM_PROMPT = (
    "You transform noisy notes into clean JSON. Return only strict JSON with the "
    "keys title, author, published_date, category, and body. All values must be strings."
)


@dataclass(frozen=True)
class ExtractionExample:
    """Static container describing a single document extraction scenario."""

    example_id: str
    raw_text: str
    target: Mapping[str, str]


@dataclass(frozen=True)
class ValidationOutcome:
    """Normalized validation feedback used by the rollout policy."""

    reward: float
    invalid: bool
    error: str
    field_accuracy: float


_examples: tuple[ExtractionExample, ...] = (
    ExtractionExample(
        example_id="alpha1",
        raw_text=(
            "Sender: Maya Hernandez (Marketing)\n"
            "Slack thread captured on March 18, 2024.\n"
            "Subject line states: Spring Campaign Kickoff.\n"
            "Metadata summary says Title=Spring Campaign Kickoff | Author=Maya Hernandez | Date=2024-03-18 | Category=marketing.\n"
            "Body section mentions: Outlines timeline for social ads and partner webinars."
        ),
        target={
            "title": "Spring Campaign Kickoff",
            "author": "Maya Hernandez",
            "published_date": "2024-03-18",
            "category": "marketing",
            "body": "Outlines timeline for social ads and partner webinars.",
        },
    ),
    ExtractionExample(
        example_id="bravo2",
        raw_text=(
            "Weekly engineering recap by Priya Shah.\n"
            "Header shows title 'Service Latency Investigation'.\n"
            "Log line: written 2024-02-27.\n"
            "Categorized as engineering in the inbox tagger.\n"
            "Detailed notes read: Root cause traced to cache stampede mitigation backlog."
        ),
        target={
            "title": "Service Latency Investigation",
            "author": "Priya Shah",
            "published_date": "2024-02-27",
            "category": "engineering",
            "body": "Root cause traced to cache stampede mitigation backlog.",
        },
    ),
    ExtractionExample(
        example_id="charlie3",
        raw_text=(
            "Finance memo stored under quarterly updates.\n"
            "Document title listed as 'Budget Reforecast Q2'.\n"
            "Prepared by Owen Brooks.\n"
            "Timestamp: 2024-04-05.\n"
            "Body copy: Adjusting spend toward customer education webinars.\n"
            "Tag set in archive: finance."
        ),
        target={
            "title": "Budget Reforecast Q2",
            "author": "Owen Brooks",
            "published_date": "2024-04-05",
            "category": "finance",
            "body": "Adjusting spend toward customer education webinars.",
        },
    ),
    ExtractionExample(
        example_id="delta4",
        raw_text=(
            "Operations checklist captured from Confluence.\n"
            "Page heading reads 'Warehouse Intake SOP Refresh'.\n"
            "Compiled by Taylor Kim on 2024-01-22.\n"
            "Ops label applied.\n"
            "Narrative paragraph: Emphasizes barcode validation before pallet staging."
        ),
        target={
            "title": "Warehouse Intake SOP Refresh",
            "author": "Taylor Kim",
            "published_date": "2024-01-22",
            "category": "operations",
            "body": "Emphasizes barcode validation before pallet staging.",
        },
    ),
    ExtractionExample(
        example_id="echo5",
        raw_text=(
            "HR onboarding digest compiled by Lena Ortiz.\n"
            "Summary label indicates title Onboarding Satisfaction Survey.\n"
            "Completion date logged as 2024-03-01.\n"
            "Topic bucket: hr.\n"
            "Survey findings line: Highlights mentorship pairing as top satisfaction driver."
        ),
        target={
            "title": "Onboarding Satisfaction Survey",
            "author": "Lena Ortiz",
            "published_date": "2024-03-01",
            "category": "hr",
            "body": "Highlights mentorship pairing as top satisfaction driver.",
        },
    ),
    ExtractionExample(
        example_id="foxtrot6",
        raw_text=(
            "Product discovery note.\n"
            "Headline: Beta Feedback Rollup.\n"
            "Reporter: Amir Saleh with date 2024-04-15.\n"
            "Filed under category product.\n"
            "Body snippet says: Customers want clearer onboarding videos and contextual tips."
        ),
        target={
            "title": "Beta Feedback Rollup",
            "author": "Amir Saleh",
            "published_date": "2024-04-15",
            "category": "product",
            "body": "Customers want clearer onboarding videos and contextual tips.",
        },
    ),
    ExtractionExample(
        example_id="golf7",
        raw_text=(
            "Support center escalation note.\n"
            "Note header: Priority Ticket Audit.\n"
            "Analyst: Noor Bennett recorded on 2024-02-14.\n"
            "Tag displayed: support.\n"
            "Body text indicates: Audit found response scripts missing refund scenarios."
        ),
        target={
            "title": "Priority Ticket Audit",
            "author": "Noor Bennett",
            "published_date": "2024-02-14",
            "category": "support",
            "body": "Audit found response scripts missing refund scenarios.",
        },
    ),
    ExtractionExample(
        example_id="hotel8",
        raw_text=(
            "Executive briefing recorded for leadership.\n"
            "Briefing title: Strategic Partnerships Outlook.\n"
            "Drafted by CEO Morgan Lee with date 2024-05-02.\n"
            "Labelled under executive.\n"
            "Body summary captures: Expecting two new fintech alliances pending compliance review."
        ),
        target={
            "title": "Strategic Partnerships Outlook",
            "author": "Morgan Lee",
            "published_date": "2024-05-02",
            "category": "executive",
            "body": "Expecting two new fintech alliances pending compliance review.",
        },
    ),
    ExtractionExample(
        example_id="india9",
        raw_text=(
            "Compliance bulletin uploaded to shared drive.\n"
            "Reference headline 'Policy Audit Findings'.\n"
            "Prepared by Dana Cooper, dated 2024-01-30.\n"
            "Classification: compliance.\n"
            "Bullet point states: Identified outdated retention language in vendor contracts."
        ),
        target={
            "title": "Policy Audit Findings",
            "author": "Dana Cooper",
            "published_date": "2024-01-30",
            "category": "compliance",
            "body": "Identified outdated retention language in vendor contracts.",
        },
    ),
    ExtractionExample(
        example_id="juliet10",
        raw_text=(
            "Lab notebook entry with header 'Prototype Thermal Test'.\n"
            "Scientist: Rui Nakamura recorded results on 2024-03-25.\n"
            "Tagged research.\n"
            "Narrative segment: Cooling loop maintained stability under 85C stress."
        ),
        target={
            "title": "Prototype Thermal Test",
            "author": "Rui Nakamura",
            "published_date": "2024-03-25",
            "category": "research",
            "body": "Cooling loop maintained stability under 85C stress.",
        },
    ),
    ExtractionExample(
        example_id="kilo11",
        raw_text=(
            "Logistics update email referencing 'Freight Consolidation Trial'.\n"
            "Coordinator Billie Adams posted on 2024-02-08.\n"
            "Categorized under logistics.\n"
            "Message body: Trial reduced partial loads and improved dock utilization."
        ),
        target={
            "title": "Freight Consolidation Trial",
            "author": "Billie Adams",
            "published_date": "2024-02-08",
            "category": "logistics",
            "body": "Trial reduced partial loads and improved dock utilization.",
        },
    ),
    ExtractionExample(
        example_id="lima12",
        raw_text=(
            "Community report delivered in town hall.\n"
            "Display title says 'Volunteer Impact Recap'.\n"
            "Narrated by Erin Blake, dated 2024-04-02.\n"
            "Label indicates community.\n"
            "Body paragraph reads: Volunteers logged 420 hours mentoring STEM students."
        ),
        target={
            "title": "Volunteer Impact Recap",
            "author": "Erin Blake",
            "published_date": "2024-04-02",
            "category": "community",
            "body": "Volunteers logged 420 hours mentoring STEM students.",
        },
    ),
)


def get_examples() -> Sequence[ExtractionExample]:
    """Return the full immutable example suite."""

    return _examples


def sample_example(rng: random.Random | None = None) -> ExtractionExample:
    """Select a deterministic example when seeded."""

    if rng is None:
        rng = random.Random(RANDOM_SEED)
    return rng.choice(_examples)


def build_user_prompt(example: ExtractionExample) -> str:
    """Compose the user-facing prompt clarifying schema expectations."""

    schema_hint = json.dumps({field: "string" for field in REQUIRED_FIELDS}, indent=2)
    return (
        "Extract the required metadata from the raw document below."\
        " Ensure the reply is valid JSON with double-quoted keys and values."\
        " Do not add explanations or code fences.\n\n"
        f"Schema:\n{schema_hint}\n\n"
        "Document:\n<<<\n"
        f"{example.raw_text}\n"
        ">>>"
    )


def validate_agent_response(raw_response: str, example: ExtractionExample) -> ValidationOutcome:
    """Score the agent response using strict JSON validation."""

    raw_response = raw_response.strip()
    if not raw_response:
        return ValidationOutcome(reward=-1.0, invalid=True, error="empty_response", field_accuracy=0.0)

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return ValidationOutcome(reward=-1.0, invalid=True, error="json_parse_error", field_accuracy=0.0)

    if not isinstance(parsed, dict):
        return ValidationOutcome(reward=-1.0, invalid=True, error="not_object", field_accuracy=0.0)

    total_fields = len(REQUIRED_FIELDS)
    correct_fields = 0

    for field in REQUIRED_FIELDS:
        if field not in parsed:
            continue
        value = parsed[field]
        if not isinstance(value, str):
            continue
        if value.strip() == example.target[field]:
            correct_fields += 1

    field_accuracy = correct_fields / total_fields

    missing = [field for field in REQUIRED_FIELDS if field not in parsed]
    non_string = [field for field in REQUIRED_FIELDS if field in parsed and not isinstance(parsed[field], str)]

    if missing:
        return ValidationOutcome(
            reward=field_accuracy - 1.0,
            invalid=True,
            error=f"missing:{','.join(sorted(missing))}",
            field_accuracy=field_accuracy,
        )

    if non_string:
        return ValidationOutcome(
            reward=field_accuracy - 1.0,
            invalid=True,
            error=f"non_string:{','.join(sorted(non_string))}",
            field_accuracy=field_accuracy,
        )

    if correct_fields != total_fields:
        return ValidationOutcome(
            reward=max(field_accuracy - 0.2, -0.5),
            invalid=False,
            error="value_mismatch",
            field_accuracy=field_accuracy,
        )

    return ValidationOutcome(
        reward=1.0,
        invalid=False,
        error="ok",
        field_accuracy=1.0,
    )


__all__ = [
    "RANDOM_SEED",
    "TRAINING_CONFIG",
    "REQUIRED_FIELDS",
    "SYSTEM_PROMPT",
    "ExtractionExample",
    "ValidationOutcome",
    "get_examples",
    "sample_example",
    "build_user_prompt",
    "validate_agent_response",
]
