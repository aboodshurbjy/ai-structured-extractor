# Evaluation set

24 short synthetic documents, 6 per extraction mode, with hand-written gold labels.
All names, companies, and figures are invented.

```
eval/
├── documents/<mode>_<nn>.txt   # input text, one file per document
└── gold/<mode>_<nn>.json       # expected output for that document
```

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
