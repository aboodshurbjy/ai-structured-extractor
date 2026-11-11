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

### Failures: anthropic:claude-haiku-5-5

- `classify_01`
  - extra label (any confidence): operations
- `classify_04`
  - extra label (any confidence): operations
- `classify_05`
  - extra label (any confidence): operations
- `classify_06`
  - extra label (any confidence): operations

### Failures: anthropic:claude-sonnet-5-5

- `classify_01`
  - extra label (any confidence): operations
  - extra label (any confidence): research
- `classify_03`
  - extra label (any confidence): general
- `classify_04`
  - extra label (any confidence): operations
- `classify_06`
  - extra label (any confidence): operations
