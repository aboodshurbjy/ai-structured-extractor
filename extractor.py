"""
AI Structured Extractor
========================
A lightweight pipeline that takes unstructured text, sends it to an LLM API
with engineered prompts, and returns validated structured output.

Supports multiple extraction modes:
  - summarize:  Executive summary + key points
  - entities:   Named entity extraction with types and context
  - actions:    Action items with owners, deadlines, and priority
  - classify:   Multi-label classification with confidence scores

Works with both Anthropic (Claude) and OpenAI (GPT) APIs.
"""

import json
import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

def _call_anthropic(model: str, system: str, user: str, max_tokens: int) -> str:
    """Call Anthropic Messages API and return the assistant text."""
    from anthropic import Anthropic

    client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return _text_from_anthropic_response(response)


def _text_from_anthropic_response(response) -> str:
    """Join the text blocks of an Anthropic response.

    The content list can start with non-text blocks (current models return a
    thinking block first), so the text is not always at index 0.
    """
    if getattr(response, "stop_reason", None) == "max_tokens":
        raise TruncatedResponseError(_TRUNCATED_MESSAGE)
    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    if not text:
        raise ParseError(
            "Anthropic response contained no text "
            f"(stop_reason: {getattr(response, 'stop_reason', None)})."
        )
    return text


def _call_openai(model: str, system: str, user: str, max_tokens: int) -> str:
    """Call OpenAI Chat Completions API and return the assistant text."""
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from env
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return _text_from_openai_response(response)


def _text_from_openai_response(response) -> str:
    """Return the message text, or raise if the output was cut off."""
    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        raise TruncatedResponseError(_TRUNCATED_MESSAGE)
    return choice.message.content


PROVIDERS = {
    "anthropic": {
        "call_name": "_call_anthropic",
        "default_model": "claude-sonnet-5-5",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "call_name": "_call_openai",
        "default_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
}


def _get_caller(provider_key: str):
    """Look up the provider's call function at runtime (allows mocking)."""
    import extractor as _self
    return getattr(_self, PROVIDERS[provider_key]["call_name"])

# ---------------------------------------------------------------------------
# Prompt templates  (the prompt engineering layer)
# ---------------------------------------------------------------------------

SYSTEM_PREAMBLE = (
    "You are a precise data-extraction assistant. "
    "Always respond with valid JSON matching the schema the user provides. "
    "Do not include any text outside the JSON object. "
    "Do not wrap the JSON in markdown code fences."
)

MODE_PROMPTS: dict[str, str] = {
    "summarize": """
Extract a structured summary from the following text.

Return JSON matching this schema EXACTLY:
{{
  "title": "<short descriptive title>",
  "summary": "<2-3 sentence executive summary>",
  "key_points": ["<point 1>", "<point 2>", ...],
  "word_count_original": <int>,
  "sentiment": "positive" | "neutral" | "negative"
}}

TEXT:
{text}
""",
    "entities": """
Extract all named entities from the following text.

Return JSON matching this schema EXACTLY:
{{
  "entities": [
    {{
      "name": "<entity name>",
      "type": "PERSON" | "ORG" | "LOCATION" | "DATE" | "MONEY" | "PRODUCT" | "OTHER",
      "context": "<short phrase showing how entity appears in text>"
    }}
  ],
  "entity_count": <int>
}}

TEXT:
{text}
""",
    "actions": """
Extract action items from the following text (e.g., meeting notes, emails, project updates).

Return JSON matching this schema EXACTLY:
{{
  "actions": [
    {{
      "task": "<what needs to be done>",
      "owner": "<person responsible, or 'unassigned'>",
      "deadline": "<deadline if mentioned, or 'none'>",
      "priority": "high" | "medium" | "low"
    }}
  ],
  "action_count": <int>
}}

TEXT:
{text}
""",
    "classify": """
Classify the following text into one or more categories.

Available categories:
  - technical, business, legal, financial, operations,
    marketing, hr, research, general

Return JSON matching this schema EXACTLY:
{{
  "classifications": [
    {{
      "category": "<category>",
      "confidence": <float 0-1>
    }}
  ],
  "primary_category": "<highest confidence category>",
  "language": "<detected language>"
}}

TEXT:
{text}
""",
}

# ---------------------------------------------------------------------------
# Output parsing & validation
# ---------------------------------------------------------------------------

REQUIRED_KEYS: dict[str, list[str]] = {
    "summarize": ["title", "summary", "key_points", "sentiment"],
    "entities": ["entities", "entity_count"],
    "actions": ["actions", "action_count"],
    "classify": ["classifications", "primary_category"],
}


class ParseError(Exception):
    """Raised when the LLM response cannot be parsed into valid JSON."""


class TruncatedResponseError(ParseError):
    """Raised when the provider stopped at the max_tokens limit, so the JSON
    is incomplete. On models that think before answering, thinking tokens
    count against the same limit."""


_TRUNCATED_MESSAGE = (
    "The response was cut off at the max_tokens limit, so the JSON is incomplete. "
    "Retry with a higher --max-tokens value."
)


def parse_response(raw: str, mode: str) -> dict:
    """
    Parse the raw LLM text into a validated Python dict.

    Steps:
      1. Strip markdown fences if present (common LLM quirk)
      2. Parse JSON
      3. Validate required keys for the given mode
    """
    cleaned = raw.strip()

    # Strip ```json ... ``` wrappers
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ParseError(
            f"LLM returned invalid JSON.\n"
            f"  Error: {exc}\n"
            f"  Raw response (first 500 chars): {raw[:500]}"
        ) from exc

    missing = [k for k in REQUIRED_KEYS.get(mode, []) if k not in data]
    if missing:
        raise ParseError(
            f"Response JSON is missing required keys for mode '{mode}': {missing}"
        )

    return data

# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """Container for a completed extraction."""
    mode: str
    provider: str
    model: str
    data: dict
    raw_response: str = field(repr=False)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.data, indent=indent, ensure_ascii=False)


def extract(
    text: str,
    mode: str = "summarize",
    provider: str = "anthropic",
    model: Optional[str] = None,
    max_tokens: int = 1024,
) -> ExtractionResult:
    """
    Run the full extraction pipeline:
      1. Select prompt template for the given mode
      2. Call the LLM provider
      3. Parse and validate the structured output
      4. Return an ExtractionResult
    """
    if mode not in MODE_PROMPTS:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(MODE_PROMPTS)}")
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")

    prov = PROVIDERS[provider]

    # Check for API key
    if not os.environ.get(prov["env_key"]):
        raise EnvironmentError(
            f"Set the {prov['env_key']} environment variable before running."
        )

    resolved_model = model or prov["default_model"]
    user_prompt = MODE_PROMPTS[mode].format(text=text)

    call_fn = _get_caller(provider)
    raw = call_fn(resolved_model, SYSTEM_PREAMBLE, user_prompt, max_tokens)
    data = parse_response(raw, mode)

    return ExtractionResult(
        mode=mode,
        provider=provider,
        model=resolved_model,
        data=data,
        raw_response=raw,
    )

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured data from unstructured text using LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Summarize a file using Claude
  python extractor.py summarize meeting_notes.txt

  # Extract entities using OpenAI
  python extractor.py entities report.txt --provider openai

  # Pipe text directly
  echo "Schedule a review with Jane by Friday." | python extractor.py actions -
        """,
    )
    parser.add_argument(
        "mode",
        choices=list(MODE_PROMPTS),
        help="Extraction mode",
    )
    parser.add_argument(
        "input",
        help="Path to a text file, or '-' to read from stdin",
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS),
        default="anthropic",
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the default model for the chosen provider",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens for the LLM response (default: 1024)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also print the raw LLM response for debugging",
    )

    args = parser.parse_args()

    # Read input
    if args.input == "-":
        text = sys.stdin.read()
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()

    if not text.strip():
        print("Error: input text is empty.", file=sys.stderr)
        sys.exit(1)

    try:
        result = extract(
            text=text,
            mode=args.mode,
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
        )
    except (ParseError, EnvironmentError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Output
    print(result.to_json())

    if args.raw:
        print("\n--- RAW RESPONSE ---", file=sys.stderr)
        print(result.raw_response, file=sys.stderr)


if __name__ == "__main__":
    main()
