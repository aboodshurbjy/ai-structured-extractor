# AI Structured Extractor

A small Python pipeline that sends unstructured text to an LLM with a mode-specific prompt and returns parsed, checked JSON. It supports Anthropic and OpenAI models behind one interface, and it comes with a labelled evaluation set and a script that scores the output.

## Modes

Give it text (meeting notes, emails, technical documents) and choose a mode:

| Mode | Output |
|------|--------|
| `summarize` | Title, summary, key points, sentiment |
| `entities` | Named entities with a type (person, org, location, date, money, product, other) and source context |
| `actions` | Action items with owner, deadline, and priority |
| `classify` | Category labels with confidence scores, a primary category, and the language |

The pipeline builds the prompt, calls the provider, parses the response, checks that the required keys are present, and raises a descriptive error when something is wrong.

## Architecture

Each mode has a prompt template with the expected JSON shape written out. The system message tells the model to return only JSON.

Anthropic and OpenAI share one call signature, `(model, system, user, max_tokens) -> str`, so switching provider is a command-line flag. Adding a provider means writing that function and adding an entry to `PROVIDERS`.

`parse_response` strips markdown code fences if present, parses the JSON, and checks the required keys for the mode. Failures raise `ParseError` with the reason and the start of the raw response. A response cut off at the token limit raises `TruncatedResponseError`, so truncation is reported separately from malformed JSON.

`ExtractionResult` holds the parsed data and the raw response for debugging.

## Quick start

```bash
git clone https://github.com/aboodshurbjy/ai-structured-extractor.git
cd ai-structured-extractor
pip install -r requirements.txt

# Set an API key
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."

# Run an extraction
python extractor.py summarize examples/meeting_notes.txt
python extractor.py entities examples/rfi_document.txt
python extractor.py actions examples/meeting_notes.txt --provider openai
echo "Ship the v2 release by Friday. Assign to Jordan." | python extractor.py actions -
```

Default models are `claude-sonnet-5-5` (Anthropic) and `gpt-4o-mini` (OpenAI). Use `--model` to override.

## Example output

Input: `examples/meeting_notes.txt` (project kickoff notes)

Command: `python extractor.py actions examples/meeting_notes.txt`

Output from `claude-sonnet-5-5` on 2026-11-10:

```json
{
  "actions": [
    {
      "task": "Get updated boring logs from geotech contractor",
      "owner": "Mike Torres",
      "deadline": "March 22, 2026",
      "priority": "high"
    },
    {
      "task": "Revise foundation plan once updated soil data is received",
      "owner": "Mike Torres",
      "deadline": "none",
      "priority": "high"
    },
    {
      "task": "Run updated HVAC load calculation and share options for expanding the Level B1 mechanical room or splitting equipment across two floors",
      "owner": "Priya Patel",
      "deadline": "March 20, 2026",
      "priority": "high"
    },
    {
      "task": "Confirm ceiling heights on B1 and B2 for Priya",
      "owner": "David Kim",
      "deadline": "March 18, 2026",
      "priority": "high"
    },
    {
      "task": "Prepare cost comparison of all three facade options and send to Lisa",
      "owner": "David Kim",
      "deadline": "March 25, 2026",
      "priority": "medium"
    },
    {
      "task": "Set up shared LEED scorecard in the project portal",
      "owner": "Sarah Chen",
      "deadline": "March 17, 2026",
      "priority": "medium"
    },
    {
      "task": "Flag any coordination conflicts in the parking garage zone (ventilation layout and seismic bracing overlap) to Sarah",
      "owner": "All attendees",
      "deadline": "ASAP",
      "priority": "high"
    },
    {
      "task": "Coordinate early on the parking garage ventilation layout, which overlaps with the seismic bracing zones",
      "owner": "Mike Torres and Priya Patel",
      "deadline": "none",
      "priority": "high"
    },
    {
      "task": "Complete construction documents to 90% for July permit submission",
      "owner": "unassigned",
      "deadline": "June 30, 2026",
      "priority": "high"
    }
  ],
  "action_count": 9
}
```

The notes end with an explicit list of 7 action items. The model returned 9: the 7 listed plus 2 it inferred from the prose above the list. Whether those 2 are wanted depends on the use case.

## Evaluation

### What was run

- The data is 24 short synthetic documents, 6 per mode, in `eval/documents/`, with gold labels in `eval/gold/`. They vary in domain and difficulty and include informal, messy text and edge cases (a document with no entities, one with no action items, a two-sentence note). The label format and the matching rules are in [eval/README.md](eval/README.md).
- The documents and gold labels were drafted with AI assistance and reviewed by me.
- The models were `claude-haiku-5-5` and `claude-sonnet-5-5`. I ran the evaluation on Anthropic models only. The OpenAI code path was not evaluated and is covered only by unit tests with mocked responses.
- Each run used the extractor's own prompts, call function, and parser, with the default `max_tokens` of 4096. One call per document per model, run sequentially on 2026-11-10.
- That is 24 documents per model. For the scored modes this is 38 gold entities across 6 documents, 18 gold action items across 6 documents (7 of them with a labelled priority), and 6 classification documents.

24 documents is a small sample. The results below show that the pipeline works on this set. They do not support claims about accuracy in general, and one more or one fewer error would move most of these percentages by several points.

### Results

Final run, raw outputs in `eval/results/final/`:

| Metric | anthropic:claude-haiku-5-5 | anthropic:claude-sonnet-5-5 |
|---|---|---|
| Documents | 24 | 24 |
| API call errors | 0 | 0 |
| Raw response is valid JSON as returned | 24/24 (100.0%) | 24/24 (100.0%) |
| Parsed by extractor (JSON + required keys) | 24/24 (100.0%) | 24/24 (100.0%) |
| Valid JSON and passes schema validation | 24/24 (100.0%) | 24/24 (100.0%) |
| ...and internally consistent | 24/24 (100.0%) | 24/24 (100.0%) |
| Entities: precision | 100.0% | 100.0% |
| Entities: recall | 100.0% | 100.0% |
| Entities: F1 | 100.0% | 100.0% |
| Entities: TP / FP / FN | 38 / 0 / 0 | 38 / 0 / 0 |
| Entities: recall on name only | 100.0% | 100.0% |
| Actions: precision | 100.0% | 100.0% |
| Actions: recall | 100.0% | 100.0% |
| Actions: F1 | 100.0% | 100.0% |
| Actions: TP / FP / FN | 18 / 0 / 0 | 18 / 0 / 0 |
| Actions: owner accuracy (matched) | 100.0% | 100.0% |
| Actions: deadline accuracy (matched) | 100.0% | 100.0% |
| Actions: priority accuracy (where labelled) | 100.0% (n=7) | 100.0% (n=7) |
| Actions: found with all fields correct | 18/18 (100.0%) | 18/18 (100.0%) |
| Classify: primary category accuracy | 6/6 (100.0%) | 6/6 (100.0%) |
| Classify: label precision, all returned labels | 60.0% (FP=4) | 54.5% (FP=5) |
| Classify: label recall, all returned labels | 100.0% | 100.0% |
| Classify: label precision, confidence >= 0.5 | 100.0% (FP=0) | 100.0% (FP=0) |
| Classify: label recall, confidence >= 0.5 | 100.0% | 100.0% |
| Classify: language accuracy | 6/6 (100.0%) | 6/6 (100.0%) |
| Latency per call, mean (s) | 2.37 | 2.57 |
| Latency per call, median (s) | 2.47 | 1.82 |
| Latency per call, min (s) | 0.77 | 1.23 |
| Latency per call, max (s) | 3.96 | 6.20 |

"Valid JSON and passes schema validation" means the response parsed with `extractor.parse_response` and also passed the stricter check in `eval/run_eval.py` (`schema_errors`): every field in the prompt's schema present with the right type, enum fields limited to the listed values, and confidence between 0 and 1. "Internally consistent" adds: `entity_count` and `action_count` equal the list lengths, `primary_category` is the highest-confidence label, and `word_count_original` is within 10% of the true word count.

Latency is wall-clock time for the provider call, measured from one machine in one run. It includes network time and the model's thinking time, and it will vary.

### Reading the results

Both models score at or near 100% on every accuracy metric. This evaluation set is too easy to tell the two models apart. I use it as a regression check for the pipeline, and it is not a benchmark. The differences that did show up were in validity and stability across runs, described under failure cases.

Each model was run once per document in each run. Outputs are not deterministic: in the second run `claude-haiku-5-5` gave a deadline for one action that did not match the gold label (17 of 18 actions fully correct), and in the final run it matched (18 of 18).

### Three runs

I ran the evaluation three times as I found and fixed bugs. Each run's raw outputs are kept.

| Run | Extractor state | claude-haiku-5-5 valid | claude-sonnet-5-5 valid |
|-----|-----------------|------------------------|-------------------------|
| `eval/results/baseline/` | As originally written, `max_tokens` 1024 | 7/24 (29.2%) | 19/24 (79.2%) |
| `eval/results/after_fix/` | After the thinking-block fix, `max_tokens` 1024 | 22/24 (91.7%) | 22/24 (91.7%) |
| `eval/results/final/` | After all fixes, `max_tokens` 4096 | 24/24 (100.0%) | 24/24 (100.0%) |

"Valid" is the "valid JSON and passes schema validation" row.

### Failure cases

1. Crash on responses that start with a thinking block (baseline run). The extractor read `response.content[0].text`. Current Anthropic models can return a thinking block first, which has no `text` attribute, so the call raised `AttributeError`. This affected 17 of 24 calls on `claude-haiku-5-5` and 5 of 24 on `claude-sonnet-5-5`. Fixed by reading the text blocks wherever they are in the list.

2. Truncated JSON at `max_tokens` 1024 (second run). For both models, `summarize_02` and `summarize_04` came back cut off mid-string and failed to parse. Thinking tokens count against `max_tokens`, so 1024 did not leave enough room for the longer summaries. The extractor also reported this as "invalid JSON", which hid the cause. Fixed in two steps: a response that stops at the token limit now raises `TruncatedResponseError` with a message that says to raise `--max-tokens`, and the default limit is now 4096.

3. Retired default model. The original default, `claude-sonnet-4-20250514`, is no longer available. The default is now `claude-sonnet-5-5`.

4. Low-confidence extra labels in classify (all runs). Both models add categories with low confidence that are neither required nor accepted as optional in the gold labels. In the final run `claude-haiku-5-5` added "operations" on 4 of 6 documents, and `claude-sonnet-5-5` added 5 such labels on 4 documents ("operations" three times, "research" once, "general" once). Counting every returned label, label precision is 60.0% and 54.5%. In all three runs the confidence on these labels was between 0.05 and 0.35, so precision is 100.0% for both models when only labels with confidence of at least 0.5 are counted. The 0.5 threshold was added to the script after seeing the first run. Anyone using the label list should filter on confidence.

5. One deadline that did not match the gold label (second run, `claude-haiku-5-5`). For "draft the methods section before the 1 July group review", the model returned "before the 1 July 2025 group review". The gold label accepts the date or "before the 1 July group review" without the year, so this was scored as an error. I think the label is too narrow here. I left it unchanged.

No other errors were observed in the final run: no missed or invented entities, no missed or spurious action items (including the document with completed and cancelled tasks), and no wrong primary category or language.

### Matching rules

Exact match is used where the field has a closed set of values. Free-text fields (entity names, action tasks, owners, deadlines) use looser rules, which are written out in [eval/README.md](eval/README.md). They are implemented in the `score_*` functions in `eval/run_eval.py` and tested in `eval/test_run_eval.py`. Summarize is not scored for accuracy.

### Reproducing

```bash
# Recompute the scores from the saved raw outputs (no API calls)
python eval/run_eval.py --label final --score-only

# Run again against the API (writes to eval/results/<label>/)
export ANTHROPIC_API_KEY="sk-ant-..."
python eval/run_eval.py --label my_run \
  --run anthropic:claude-haiku-5-5 --run anthropic:claude-sonnet-5-5

# Add an OpenAI model
python eval/run_eval.py --label my_run --run openai:gpt-4o-mini
```

A provider whose API key is not set is skipped with a message. A new API run will not reproduce the saved outputs exactly, because model outputs vary between runs.

## Testing

The unit tests use mocked provider responses, so they need no API key.

```bash
pytest tests.py eval/test_run_eval.py -v
```

`tests.py` covers prompt rendering, JSON parsing, required-key validation, Anthropic response handling, truncation detection, and end-to-end extraction with mocked providers. `eval/test_run_eval.py` covers the scoring rules and checks that the gold labels score 100% against themselves.

## Layout

| Path | Contents |
|---|---|
| `extractor.py` | prompts, provider calls, parsing, CLI |
| `tests.py` | unit and integration tests (mocked providers) |
| `eval/documents/` | 24 synthetic input documents |
| `eval/gold/` | gold labels, one JSON file per document |
| `eval/results/` | raw model outputs and score summaries per run |
| `eval/run_eval.py` | evaluation script |
| `eval/test_run_eval.py` | tests for the scoring rules |
| `eval/README.md` | gold label format and matching rules |
| `examples/meeting_notes.txt` | project kickoff meeting |
| `examples/rfi_document.txt` | construction RFI (request for information) |

## Known limitations

- `parse_response` checks that required keys exist. It does not check field types or enum values. The stricter check lives in `eval/run_eval.py` and is not applied at extraction time.
- There is no retry on a failed or truncated response.
- The OpenAI path has not been run against the live API in this evaluation.

## License

MIT
