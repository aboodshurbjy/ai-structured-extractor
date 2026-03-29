# AI Structured Extractor

A lightweight Python pipeline that takes unstructured text, routes it through an LLM with engineered prompts, and returns validated structured JSON output.

Built as a clean demonstration of **API integration**, **prompt engineering**, and **structured output parsing** — the core building blocks behind any production LLM application.

## What It Does

Feed it any text — meeting notes, technical documents, emails, project briefs — and choose an extraction mode:

| Mode | Output |
|------|--------|
| `summarize` | Executive summary, key points, sentiment |
| `entities` | Named entities with types (person, org, location, date, etc.) and source context |
| `actions` | Action items with owner, deadline, and priority |
| `classify` | Multi-label classification with confidence scores |

The pipeline handles prompt construction, API calls, response parsing, schema validation, and error recovery.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────┐     ┌────────────┐
│  Raw Text   │ ──► │  Prompt Engine   │ ──► │  LLM Provider │ ──► │  Parser &  │
│  (file/     │     │  (mode-specific  │     │  (Anthropic / │     │  Validator │
│   stdin)    │     │   templates)     │     │   OpenAI)     │     │  (JSON +   │
└─────────────┘     └──────────────────┘     └───────────────┘     │  schema)   │
                                                                    └─────┬──────┘
                                                                          │
                                                                    ┌─────▼──────┐
                                                                    │ Structured │
                                                                    │   Output   │
                                                                    └────────────┘
```

**Key design decisions:**

- **Provider abstraction** — Anthropic and OpenAI share one interface, so swapping models is a flag change
- **Prompt templates with explicit schemas** — Each mode defines the exact JSON shape the LLM must return, reducing hallucinated structure
- **Defensive parsing** — Strips markdown fences, validates required keys, and surfaces clear errors when the LLM deviates
- **Dataclass result container** — `ExtractionResult` holds both parsed data and the raw response for debugging

## Quick Start

```bash
# Clone and install
git clone https://github.com/aboodshurbjy/ai-structured-extractor.git
cd ai-structured-extractor
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."

# Run an extraction
python extractor.py summarize examples/meeting_notes.txt
python extractor.py entities examples/rfi_document.txt
python extractor.py actions examples/meeting_notes.txt --provider openai
echo "Ship the v2 release by Friday. Assign to Jordan." | python extractor.py actions -
```

## Example Output

**Input:** `examples/meeting_notes.txt` (AEC project kickoff notes)

**Command:** `python extractor.py actions examples/meeting_notes.txt`

```json
{
  "actions": [
    {
      "task": "Get updated boring logs from geotech contractor",
      "owner": "Mike Torres",
      "deadline": "2026-03-22",
      "priority": "high"
    },
    {
      "task": "Run updated HVAC load calc and share options",
      "owner": "Priya Patel",
      "deadline": "2026-03-20",
      "priority": "high"
    },
    {
      "task": "Confirm B1/B2 ceiling heights for MEP coordination",
      "owner": "David Kim",
      "deadline": "2026-03-18",
      "priority": "medium"
    },
    {
      "task": "Prepare facade cost comparison (3 options)",
      "owner": "David Kim",
      "deadline": "2026-03-25",
      "priority": "medium"
    },
    {
      "task": "Set up LEED scorecard in project portal",
      "owner": "Sarah Chen",
      "deadline": "2026-03-17",
      "priority": "medium"
    }
  ],
  "action_count": 5
}
```

## Testing

Tests cover prompt rendering, JSON parsing, schema validation, and full end-to-end extraction with mocked API calls — no API key needed to run them.

```bash
pytest tests.py -v
```

## Project Structure

```
ai-structured-extractor/
├── extractor.py          # Core pipeline: prompts, API calls, parsing, CLI
├── tests.py              # Unit + integration tests (mocked providers)
├── requirements.txt
├── examples/
│   ├── meeting_notes.txt # AEC project kickoff meeting
│   └── rfi_document.txt  # Construction RFI (request for information)
└── README.md
```

## How It Works (Technical Detail)

**Prompt Engineering:** Each extraction mode uses a template with an explicit JSON schema embedded in the prompt. The system message constrains the LLM to output only valid JSON. This approach is more reliable than asking for "structured output" without specifying the shape.

**Output Parsing:** The parser handles common LLM failure modes — markdown code fences around JSON, missing keys, malformed responses — and raises descriptive errors rather than silent failures. This is critical in production: you need to know *why* an extraction failed, not just that it did.

**Provider Abstraction:** Both Anthropic and OpenAI are wrapped behind a common call signature `(model, system, user, max_tokens) → str`. Adding a new provider (e.g., Mistral, local Ollama) is ~10 lines.

## License

MIT
