# Evaluation set

24 short synthetic documents, 6 per extraction mode, with gold labels. The documents and gold labels were drafted with AI assistance and reviewed by me.
All names, companies, and figures are invented.

- `documents/<mode>_<nn>.txt` is the input text, one file per document.
- `gold/<mode>_<nn>.json` is the expected output for that document.

Each gold file has `id`, `mode`, `difficulty`, `domain`, `notes` (what the document is meant to test), and `expected` (the label).

## Documents

| Mode | Easy | Medium | Hard | Messy | Edge case |
|------|------|--------|------|-------|-----------|
| summarize | 01 | 02, 03 | 04 | 05 | 06 (two sentences) |
| entities | 01 | 02, 03 | 04 | 05 | 06 (no entities) |
| actions | 01 | 02, 06 | 03 | 04 | 05 (no actions) |
| classify | 01, 02 | 04 | 03, 06 | 05 | |

"Messy" documents are informal chat or draft text with typos, abbreviations, and missing capitalization.

## Gold label format

### entities

- `entities`: entities the extractor must find. Each has `name` and `type`.
  - `aliases`: other spellings of the name that count as the same entity (for example "Dr. Amara Okafor" for "Amara Okafor").
  - `type_accept`: used only where more than one type is defensible (for example a hospital as ORG or LOCATION). If absent, only `type` is accepted.
- `optional_entities`: borderline items (role words, vague dates such as "last spring"). Returning one is not rewarded and not penalized.

### actions

- `actions`: the action items that should be extracted. Each has:
  - `task`: a reference wording, for the human reader.
  - `match_keywords`: lowercase substrings that must all appear in a predicted task for it to be paired with this gold action. Task text is free-form, so this is how a prediction is aligned to a gold item.
  - `owner` and `owner_accept`: the expected owner and the lowercase strings accepted as equal to it. `unassigned` when the text names nobody.
  - `deadline` and `deadline_accept`: the expected deadline and the lowercase strings accepted as equal to it. `none` when the text gives no deadline. Full dates are written in ISO form in `deadline`.
  - `priority`: `high`, `medium`, or `low` only when the text states or clearly signals a priority ("urgent", "no rush"). `null` when it does not, in which case priority is not scored for that action.
- `not_actions` (actions_03 only): things in the text that look like tasks but are done, cancelled, or decisions. Listed for the reader.

### classify

- `primary_category`: the single expected primary category.
- `required_categories`: labels that must appear in the predicted label list.
- `optional_categories`: labels that are defensible but not required. Not rewarded and not penalized.
- `language` and `language_accept`: expected language and the lowercase strings accepted as equal to it.

### summarize

`expected` holds a reference title, summary, key points, and sentiment in the extractor's output shape. Summaries are free text, so these are a reference for reading the outputs, not a scoring key.

## Matching rules

Exact match is used where the field has a closed set of values. Free-text fields use the rules below. They are implemented in the `score_*` functions in `run_eval.py` and tested in `test_run_eval.py`.

### Entity matching

- Names are compared after lowercasing, Unicode normalization, collapsing whitespace, and stripping outer punctuation and a leading "the".
- A predicted entity matches a gold entity when its name equals the gold name or one of its listed aliases, and its type equals the gold type. A few gold entities accept two types where both are defensible (a hospital as ORG or LOCATION).
- Predictions are de-duplicated on name and type. A repeat of an already matched entity is ignored.
- Predictions that match a gold "optional" entity (vague dates, role words) are ignored: not rewarded and not penalized.
- A correct name with the wrong type counts as one false positive and one false negative.
- Precision, recall, and F1 are micro-averaged over all 6 documents. "Recall on name only" ignores the type.

### Action matching

- Task text is free-form, so a predicted action is paired with a gold action when every gold keyword appears in the predicted task (case-insensitive). Each prediction can be paired once. Unpaired predictions are false positives and unpaired gold actions are false negatives.
- Owner: normalized text must be in the gold accept list (for example "hana" or "hana suzuki"). The literal "unassigned" is required when the text names nobody.
- Deadline: normalized text must be in the gold accept list, or parse to the same calendar date as the gold date, with a missing year taken from the gold date. The literal "none" is required when there is no deadline.
- Priority: exact match, scored only for the 7 actions where the text signals a priority. For the other 11 the gold priority is null and is not scored.
- "Found with all fields correct" is the share of gold actions that were paired and had owner, deadline, and (where labelled) priority all correct.

### Classify matching

- Primary category and language: exact match after lowercasing. Language accepts the name or the ISO code ("spanish" or "es").
- Label list: each required gold label must be present. Labels marked optional in the gold file are ignored. Any other label is a false positive. Reported for all returned labels and for labels with confidence of at least 0.5.

### Summarize

Summarize is not scored for accuracy. A summary is free text with many acceptable wordings, and exact or keyword matching against a reference would mostly measure phrasing. Scoring it properly needs human rating or a model-based judge, which would add its own errors. I left that out of scope. Summarize outputs are included in the validity, consistency, and latency rows, and the gold files hold a reference summary for reading the outputs by hand.
