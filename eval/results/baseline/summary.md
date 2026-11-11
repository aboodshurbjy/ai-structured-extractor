| Metric | anthropic:claude-haiku-5-5 | anthropic:claude-sonnet-5-5 |
|---|---|---|
| Documents | 24 | 24 |
| API call errors | 17 | 5 |
| Raw response is valid JSON as returned | 7/24 (29.2%) | 19/24 (79.2%) |
| Parsed by extractor (JSON + required keys) | 7/24 (29.2%) | 19/24 (79.2%) |
| Valid JSON and passes schema validation | 7/24 (29.2%) | 19/24 (79.2%) |
| ...and internally consistent | 7/24 (29.2%) | 19/24 (79.2%) |
| Entities: precision | n/a | 100.0% |
| Entities: recall | n/a | 100.0% |
| Entities: F1 | n/a | 100.0% |
| Entities: TP / FP / FN | 0 / 0 / 0 | 38 / 0 / 0 |
| Entities: recall on name only | n/a | 100.0% |
| Actions: precision | 100.0% | 100.0% |
| Actions: recall | 100.0% | 100.0% |
| Actions: F1 | 100.0% | 100.0% |
| Actions: TP / FP / FN | 4 / 0 / 0 | 18 / 0 / 0 |
| Actions: owner accuracy (matched) | 100.0% | 100.0% |
| Actions: deadline accuracy (matched) | 100.0% | 100.0% |
| Actions: priority accuracy (where labelled) | 100.0% (n=3) | 100.0% (n=7) |
| Actions: found with all fields correct | 4/4 (100.0%) | 18/18 (100.0%) |
| Classify: primary category accuracy | 5/5 (100.0%) | 6/6 (100.0%) |
| Classify: label precision | 71.4% | 54.5% |
| Classify: label recall | 100.0% | 100.0% |
| Classify: language accuracy | 5/5 (100.0%) | 6/6 (100.0%) |
| Latency per call, mean (s) | 0.95 | 1.80 |
| Latency per call, median (s) | 0.87 | 1.68 |
| Latency per call, min (s) | 0.85 | 1.05 |
| Latency per call, max (s) | 1.44 | 2.89 |

### Failures: anthropic:claude-haiku-5-5

- `actions_01`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `actions_02`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `actions_03`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `actions_04`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `classify_01`
  - extra label: operations
- `classify_04`
  - extra label: operations
- `classify_06`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `entities_01`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `entities_02`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `entities_03`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `entities_04`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `entities_05`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `entities_06`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `summarize_01`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `summarize_02`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `summarize_03`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `summarize_04`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `summarize_05`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `summarize_06`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'

### Failures: anthropic:claude-sonnet-5-5

- `classify_01`
  - extra label: operations
  - extra label: research
- `classify_03`
  - extra label: general
- `classify_04`
  - extra label: operations
- `classify_06`
  - extra label: operations
- `summarize_01`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `summarize_02`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `summarize_03`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `summarize_04`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
- `summarize_05`
  - call failed: AttributeError: 'ThinkingBlock' object has no attribute 'text'
