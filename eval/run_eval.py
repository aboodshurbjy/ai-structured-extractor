"""
Evaluate the extractor against the gold-labelled documents in eval/.

For each provider:model pair it runs every document through the same prompt,
provider call, and parser that `extractor.extract` uses, then reports:

  - share of responses that parse as JSON and pass schema validation
  - field-level accuracy against the gold labels (entities, actions, classify)
  - latency per call

Raw outputs are written to eval/results/<label>/ so the scores can be
recomputed without calling the API again:

  python eval/run_eval.py --label my_run --run anthropic:claude-haiku-5-5
  python eval/run_eval.py --label my_run --score-only

The matching rules are documented in README.md (Evaluation section) and in the
docstrings of the score_* functions below.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

import extractor  # noqa: E402

MODES = ["summarize", "entities", "actions", "classify"]
ENTITY_TYPES = {"PERSON", "ORG", "LOCATION", "DATE", "MONEY", "PRODUCT", "OTHER"}
PRIORITIES = {"high", "medium", "low"}
SENTIMENTS = {"positive", "neutral", "negative"}
CATEGORIES = {
    "technical", "business", "legal", "financial", "operations",
    "marketing", "hr", "research", "general",
}

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset() -> list[dict]:
    """Return one dict per document: id, mode, text, and the gold label file."""
    items = []
    for gold_path in sorted((EVAL_DIR / "gold").glob("*.json")):
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        text = (EVAL_DIR / "documents" / f"{gold['id']}.txt").read_text(encoding="utf-8")
        items.append({"id": gold["id"], "mode": gold["mode"], "text": text, "gold": gold})
    return items


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def norm(value) -> str:
    """Lowercase, Unicode-normalize, collapse whitespace, strip outer
    punctuation and a leading 'the'. Non-strings become ''."""
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" .,;:!?\"'`()[]")
    if text.startswith("the "):
        text = text[4:]
    return text


_MONTHS = {
    name: i + 1
    for i, names in enumerate(
        [
            ("january", "jan"), ("february", "feb"), ("march", "mar"),
            ("april", "apr"), ("may",), ("june", "jun"), ("july", "jul"),
            ("august", "aug"), ("september", "sep", "sept"), ("october", "oct"),
            ("november", "nov"), ("december", "dec"),
        ]
    )
    for name in names
}


def parse_date(value, default_year: int | None = None) -> str | None:
    """Parse '2026-02-09', 'February 9, 2026', '9 Feb', 'by 9 February' and
    similar into an ISO date string. A missing year is filled with
    default_year. Returns None if the text is not a recognizable date."""
    text = norm(value)
    text = re.sub(r"^(by|due|before|on|until)\s+", "", text)
    match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if match:
        year, month, day = (int(g) for g in match.groups())
    else:
        text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text).replace(",", " ")
        tokens = text.split()
        month = day = year = None
        for token in tokens:
            if token in _MONTHS and month is None:
                month = _MONTHS[token]
            elif re.fullmatch(r"\d{1,2}", token) and day is None:
                day = int(token)
            elif re.fullmatch(r"\d{4}", token) and year is None:
                year = int(token)
            else:
                return None
        if month is None or day is None:
            return None
        year = year or default_year
        if year is None:
            return None
    try:
        return dt.date(year, month, day).isoformat()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Schema validation (stricter than extractor.parse_response)
# ---------------------------------------------------------------------------

def schema_errors(data, mode: str) -> list[str]:
    """Check the parsed output against the schema given in the prompt:
    field types, enum values, and numeric ranges. Returns a list of problems,
    empty if the output conforms."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["top level is not a JSON object"]

    def need(obj, key, kind, where=""):
        if key not in obj:
            errors.append(f"{where}missing '{key}'")
            return None
        value = obj[key]
        if kind is int:
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif kind is float:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            ok = isinstance(value, kind)
        if not ok:
            errors.append(f"{where}'{key}' has type {type(value).__name__}")
            return None
        return value

    if mode == "summarize":
        need(data, "title", str)
        need(data, "summary", str)
        points = need(data, "key_points", list)
        if points is not None and not all(isinstance(p, str) for p in points):
            errors.append("'key_points' contains non-strings")
        need(data, "word_count_original", int)
        sentiment = need(data, "sentiment", str)
        if sentiment is not None and sentiment not in SENTIMENTS:
            errors.append(f"'sentiment' value '{sentiment}' not in schema")

    elif mode == "entities":
        entities = need(data, "entities", list)
        need(data, "entity_count", int)
        for i, entity in enumerate(entities or []):
            where = f"entities[{i}]: "
            if not isinstance(entity, dict):
                errors.append(f"{where}not an object")
                continue
            need(entity, "name", str, where)
            need(entity, "context", str, where)
            etype = need(entity, "type", str, where)
            if etype is not None and etype not in ENTITY_TYPES:
                errors.append(f"{where}'type' value '{etype}' not in schema")

    elif mode == "actions":
        actions = need(data, "actions", list)
        need(data, "action_count", int)
        for i, action in enumerate(actions or []):
            where = f"actions[{i}]: "
            if not isinstance(action, dict):
                errors.append(f"{where}not an object")
                continue
            need(action, "task", str, where)
            need(action, "owner", str, where)
            need(action, "deadline", str, where)
            priority = need(action, "priority", str, where)
            if priority is not None and priority not in PRIORITIES:
                errors.append(f"{where}'priority' value '{priority}' not in schema")

    elif mode == "classify":
        labels = need(data, "classifications", list)
        if labels is not None and not labels:
            errors.append("'classifications' is empty")
        for i, label in enumerate(labels or []):
            where = f"classifications[{i}]: "
            if not isinstance(label, dict):
                errors.append(f"{where}not an object")
                continue
            category = need(label, "category", str, where)
            if category is not None and category not in CATEGORIES:
                errors.append(f"{where}'category' value '{category}' not in schema")
            confidence = need(label, "confidence", float, where)
            if confidence is not None and not 0 <= confidence <= 1:
                errors.append(f"{where}'confidence' {confidence} outside 0-1")
        primary = need(data, "primary_category", str)
        if primary is not None and primary not in CATEGORIES:
            errors.append(f"'primary_category' value '{primary}' not in schema")
        need(data, "language", str)

    return errors


def consistency_errors(data, mode: str, text: str) -> list[str]:
    """Checks that the output agrees with itself (and, for word count, with
    the input). Reported separately from schema validity."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return errors
    if mode == "entities" and isinstance(data.get("entities"), list):
        if data.get("entity_count") != len(data["entities"]):
            errors.append(
                f"entity_count={data.get('entity_count')} but {len(data['entities'])} entities listed"
            )
    if mode == "actions" and isinstance(data.get("actions"), list):
        if data.get("action_count") != len(data["actions"]):
            errors.append(
                f"action_count={data.get('action_count')} but {len(data['actions'])} actions listed"
            )
    if mode == "classify" and isinstance(data.get("classifications"), list):
        scored = [
            c for c in data["classifications"]
            if isinstance(c, dict) and isinstance(c.get("confidence"), (int, float))
        ]
        if scored:
            top = max(c["confidence"] for c in scored)
            top_categories = {c.get("category") for c in scored if c["confidence"] == top}
            if data.get("primary_category") not in top_categories:
                errors.append("primary_category is not the highest-confidence category")
    if mode == "summarize" and isinstance(data.get("word_count_original"), int):
        actual = len(text.split())
        reported = data["word_count_original"]
        if abs(reported - actual) > max(1, round(0.10 * actual)):
            errors.append(f"word_count_original={reported} but the text has {actual} words")
    return errors


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_entities(data: dict, expected: dict) -> dict:
    """Entity scoring.

    A predicted entity matches a gold entity when its normalized name equals
    the gold name or one of its aliases AND its type is the gold type (or one
    of `type_accept`). Each gold entity can be matched once.

    - Predictions are de-duplicated on (normalized name, type) first.
    - A prediction whose name matches an `optional_entities` item is ignored.
    - A prediction that repeats an already-matched gold entity is ignored.
    - A prediction with a gold name but the wrong type counts as a false
      positive, and the gold entity as a false negative (`type_errors`).
    - Everything else predicted is a false positive.
    """
    gold = expected.get("entities", [])
    optional = expected.get("optional_entities", [])

    def names(entity):
        return {norm(entity["name"])} | {norm(a) for a in entity.get("aliases", [])}

    def types(entity):
        return set(entity.get("type_accept") or [entity["type"]])

    predictions, seen = [], set()
    for entity in data.get("entities", []) if isinstance(data.get("entities"), list) else []:
        if not isinstance(entity, dict):
            continue
        key = (norm(entity.get("name")), entity.get("type"))
        if key not in seen:
            seen.add(key)
            predictions.append(key)

    matched_gold: set[int] = set()
    used: set[int] = set()
    # pass 1: name and type both match
    for gi, g in enumerate(gold):
        for pi, (pname, ptype) in enumerate(predictions):
            if pi not in used and pname in names(g) and ptype in types(g):
                matched_gold.add(gi)
                used.add(pi)
                break

    false_positives, type_errors, ignored = [], [], []
    for pi, (pname, ptype) in enumerate(predictions):
        if pi in used:
            continue
        if any(pname in names(o) for o in optional):
            ignored.append(pname)
            continue
        name_hits = [gi for gi, g in enumerate(gold) if pname in names(g)]
        if any(gi in matched_gold and ptype in types(gold[gi]) for gi in name_hits):
            ignored.append(pname)  # duplicate mention of a matched entity
            continue
        if any(gi not in matched_gold for gi in name_hits):
            type_errors.append(f"{pname} ({ptype})")
        false_positives.append(f"{pname} ({ptype})")

    missed = [f"{g['name']} ({g['type']})" for gi, g in enumerate(gold) if gi not in matched_gold]
    all_pred_names = {p[0] for p in predictions}
    name_only = sum(1 for g in gold if names(g) & all_pred_names)
    return {
        "tp": len(matched_gold),
        "fp": len(false_positives),
        "fn": len(missed),
        "gold": len(gold),
        "name_only_found": name_only,
        "missed": missed,
        "false_positives": false_positives,
        "type_errors": type_errors,
        "ignored": ignored,
    }


def deadline_matches(predicted, gold_action: dict, default_year: int | None) -> bool:
    """A deadline is correct when its normalized text is in `deadline_accept`,
    or, for gold deadlines that are calendar dates, when it parses to the same
    date (a missing year is taken from the gold date)."""
    text = norm(predicted)
    accept = {norm(a) for a in gold_action["deadline_accept"]}
    if text in accept or re.sub(r"^(by|due|before)\s+", "", text) in accept:
        return True
    gold_date = parse_date(gold_action["deadline"])
    if gold_date:
        return parse_date(predicted, default_year or int(gold_date[:4])) == gold_date
    return False


def score_actions(data: dict, expected: dict) -> dict:
    """Action scoring.

    Task text is free-form, so a predicted action is paired with a gold action
    when every string in the gold `match_keywords` appears in the predicted
    task (case-insensitive). Gold actions are processed in file order and each
    prediction can be used once. Unpaired predictions are spurious; unpaired
    gold actions are missed.

    For paired actions the fields are scored separately:
    - owner: normalized text must be in `owner_accept`
    - deadline: see deadline_matches
    - priority: exact match, scored only when the gold priority is not null
    """
    gold = expected.get("actions", [])
    predictions = [
        a for a in (data.get("actions") if isinstance(data.get("actions"), list) else [])
        if isinstance(a, dict)
    ]
    used: set[int] = set()
    pairs = []
    for g in gold:
        for pi, p in enumerate(predictions):
            task = p.get("task").lower() if isinstance(p.get("task"), str) else ""
            if pi not in used and all(k in task for k in g["match_keywords"]):
                used.add(pi)
                pairs.append((g, p))
                break

    paired_gold = {id(g) for g, _ in pairs}
    result = {
        "tp": len(pairs),
        "fp": len(predictions) - len(used),
        "fn": len(gold) - len(pairs),
        "gold": len(gold),
        "owner_correct": 0,
        "deadline_correct": 0,
        "priority_scored": 0,
        "priority_correct": 0,
        "all_fields_correct": 0,
        "missed": [g["task"] for g in gold if id(g) not in paired_gold],
        "spurious": [predictions[i].get("task") for i in range(len(predictions)) if i not in used],
        "field_errors": [],
    }
    for g, p in pairs:
        owner_ok = norm(p.get("owner")) in {norm(o) for o in g["owner_accept"]}
        deadline_ok = deadline_matches(p.get("deadline"), g, None)
        priority_ok = True
        if g["priority"] is not None:
            result["priority_scored"] += 1
            priority_ok = p.get("priority") == g["priority"]
            result["priority_correct"] += priority_ok
            if not priority_ok:
                result["field_errors"].append(
                    f"{g['task']}: priority '{p.get('priority')}', expected '{g['priority']}'"
                )
        result["owner_correct"] += owner_ok
        result["deadline_correct"] += deadline_ok
        result["all_fields_correct"] += owner_ok and deadline_ok and priority_ok
        if not owner_ok:
            result["field_errors"].append(
                f"{g['task']}: owner '{p.get('owner')}', expected '{g['owner']}'"
            )
        if not deadline_ok:
            result["field_errors"].append(
                f"{g['task']}: deadline '{p.get('deadline')}', expected '{g['deadline']}'"
            )
    return result


def score_classify(data: dict, expected: dict) -> dict:
    """Classification scoring.

    - primary_category: exact match after lowercasing.
    - label set: every `required_categories` label should be present (false
      negative if not). Predicted labels in `optional_categories` are ignored.
      Any other predicted label is a false positive.
    - language: normalized text must be in `language_accept`.
    """
    predicted = set()
    for label in data.get("classifications", []) if isinstance(data.get("classifications"), list) else []:
        if isinstance(label, dict) and isinstance(label.get("category"), str):
            predicted.add(label["category"].strip().lower())
    required = set(expected["required_categories"])
    optional = set(expected.get("optional_categories", []))
    primary = data.get("primary_category")
    primary = primary.strip().lower() if isinstance(primary, str) else None
    extra = sorted(predicted - required - optional)
    return {
        "primary_correct": primary == expected["primary_category"],
        "primary_predicted": primary,
        "primary_expected": expected["primary_category"],
        "tp": len(required & predicted),
        "fn": len(required - predicted),
        "fp": len(extra),
        "extra_labels": extra,
        "predicted_labels": sorted(predicted),
        "language_correct": norm(data.get("language")) in {norm(a) for a in expected["language_accept"]},
        "language_predicted": data.get("language"),
    }


SCORERS = {"entities": score_entities, "actions": score_actions, "classify": score_classify}


def evaluate_record(record: dict, item: dict) -> dict:
    """Attach validity checks and scores to one raw output record."""
    mode = item["mode"]
    raw = record.get("raw_response")
    out = {
        "raw_json_ok": False,   # json.loads on the untouched response
        "parsed_ok": False,     # extractor.parse_response succeeded
        "schema_ok": False,     # parsed and conforms to the prompt schema
        "parse_error": None,
        "schema_errors": [],
        "consistency_errors": [],
        "score": None,
    }
    if isinstance(raw, str):
        try:
            json.loads(raw)
            out["raw_json_ok"] = True
        except json.JSONDecodeError:
            pass
        try:
            data = extractor.parse_response(raw, mode)
            out["parsed_ok"] = True
        except extractor.ParseError as exc:
            out["parse_error"] = str(exc).splitlines()[0]
            data = None
        if data is not None:
            out["schema_errors"] = schema_errors(data, mode)
            out["schema_ok"] = not out["schema_errors"]
            out["consistency_errors"] = consistency_errors(data, mode, item["text"])
            if mode in SCORERS and isinstance(data, dict):
                out["score"] = SCORERS[mode](data, item["gold"]["expected"])
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall else (0.0 if precision is not None and recall is not None else None)
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def summarize_run(records: list[dict], items: dict[str, dict]) -> dict:
    """Aggregate per-document results for one provider:model pair."""
    n = len(records)
    evaluated = [(r, evaluate_record(r, items[r["id"]])) for r in records]
    summary: dict = {
        "n_documents": n,
        "api_errors": sum(1 for r in records if r.get("error")),
        "raw_json_ok": sum(e["raw_json_ok"] for _, e in evaluated),
        "parsed_ok": sum(e["parsed_ok"] for _, e in evaluated),
        "schema_ok": sum(e["schema_ok"] for _, e in evaluated),
        "consistent": sum(e["schema_ok"] and not e["consistency_errors"] for _, e in evaluated),
        "by_mode_schema_ok": {},
    }
    for mode in MODES:
        subset = [e for r, e in evaluated if items[r["id"]]["mode"] == mode]
        summary["by_mode_schema_ok"][mode] = [sum(e["schema_ok"] for e in subset), len(subset)]

    latencies = [r["latency_s"] for r in records if r.get("latency_s") is not None and not r.get("error")]
    if latencies:
        summary["latency_s"] = {
            "n": len(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
        }

    def scores(mode):
        return [
            e["score"] for r, e in evaluated
            if items[r["id"]]["mode"] == mode and e["score"] is not None
        ]

    def total(rows, key):
        return sum(row[key] for row in rows)

    ent = scores("entities")
    summary["entities"] = {
        "documents_scored": len(ent),
        "gold_entities": total(ent, "gold"),
        "tp": total(ent, "tp"), "fp": total(ent, "fp"), "fn": total(ent, "fn"),
        **prf(total(ent, "tp"), total(ent, "fp"), total(ent, "fn")),
        "name_only_recall": total(ent, "name_only_found") / total(ent, "gold") if ent and total(ent, "gold") else None,
    }
    act = scores("actions")
    tp = total(act, "tp")
    summary["actions"] = {
        "documents_scored": len(act),
        "gold_actions": total(act, "gold"),
        "tp": tp, "fp": total(act, "fp"), "fn": total(act, "fn"),
        **prf(tp, total(act, "fp"), total(act, "fn")),
        "owner_accuracy": total(act, "owner_correct") / tp if tp else None,
        "deadline_accuracy": total(act, "deadline_correct") / tp if tp else None,
        "priority_scored": total(act, "priority_scored"),
        "priority_accuracy": (
            total(act, "priority_correct") / total(act, "priority_scored")
            if total(act, "priority_scored") else None
        ),
        "all_fields_correct": total(act, "all_fields_correct"),
        # share of gold actions found with every scored field correct
        "exact_action_rate": total(act, "all_fields_correct") / total(act, "gold") if act and total(act, "gold") else None,
    }
    cls = scores("classify")
    summary["classify"] = {
        "documents_scored": len(cls),
        "primary_correct": total(cls, "primary_correct"),
        "primary_accuracy": total(cls, "primary_correct") / len(cls) if cls else None,
        "language_correct": total(cls, "language_correct"),
        "language_accuracy": total(cls, "language_correct") / len(cls) if cls else None,
        "label_tp": total(cls, "tp"), "label_fp": total(cls, "fp"), "label_fn": total(cls, "fn"),
        **prf(total(cls, "tp"), total(cls, "fp"), total(cls, "fn")),
    }

    failures = []
    for r, e in evaluated:
        problems = []
        if r.get("error"):
            problems.append(f"call failed: {r['error']}")
        elif not e["parsed_ok"]:
            problems.append(f"parse failed: {e['parse_error']}")
        problems += [f"schema: {s}" for s in e["schema_errors"]]
        problems += [f"consistency: {s}" for s in e["consistency_errors"]]
        score = e["score"] or {}
        problems += [f"missed: {m}" for m in score.get("missed", [])]
        problems += [f"false positive: {m}" for m in score.get("false_positives", [])]
        problems += [f"spurious action: {m}" for m in score.get("spurious", [])]
        problems += [f"field: {m}" for m in score.get("field_errors", [])]
        if score.get("primary_correct") is False:
            problems.append(
                f"primary_category '{score['primary_predicted']}', expected '{score['primary_expected']}'"
            )
        problems += [f"extra label: {m}" for m in score.get("extra_labels", [])]
        if score.get("language_correct") is False:
            problems.append(f"language '{score['language_predicted']}' not accepted")
        if problems:
            failures.append({"id": r["id"], "problems": problems})
    summary["failures"] = failures
    return summary


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_model(provider: str, model: str, dataset: list[dict], max_tokens: int) -> list[dict]:
    """Call the provider once per document using the extractor's own prompt
    templates and call function. Returns raw output records."""
    call_fn = extractor._get_caller(provider)
    records = []
    for item in dataset:
        prompt = extractor.MODE_PROMPTS[item["mode"]].format(text=item["text"])
        record = {
            "id": item["id"], "mode": item["mode"], "provider": provider, "model": model,
            "max_tokens": max_tokens, "raw_response": None, "error": None, "latency_s": None,
        }
        start = time.perf_counter()
        try:
            record["raw_response"] = call_fn(model, extractor.SYSTEM_PREAMBLE, prompt, max_tokens)
        except Exception as exc:  # recorded as a failed call, the run continues
            record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        record["latency_s"] = round(time.perf_counter() - start, 3)
        records.append(record)
        status = "error" if record["error"] else "ok"
        print(f"  {provider}:{model}  {item['id']:<14} {record['latency_s']:>6.2f}s  {status}", flush=True)
    return records


def output_path(out_dir: Path, provider: str, model: str) -> Path:
    return out_dir / f"{provider}__{re.sub(r'[^A-Za-z0-9._-]', '_', model)}.jsonl"


def pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def render_markdown(summaries: dict[str, dict]) -> str:
    """Side-by-side results table, one column per provider:model."""
    names = list(summaries)
    rows = [("Metric", *names)]

    def add(label, fn):
        rows.append((label, *[fn(summaries[name]) for name in names]))

    def frac(key):
        return lambda s: f"{s[key]}/{s['n_documents']} ({pct(s[key] / s['n_documents'])})"

    add("Documents", lambda s: str(s["n_documents"]))
    add("API call errors", lambda s: str(s["api_errors"]))
    add("Raw response is valid JSON as returned", frac("raw_json_ok"))
    add("Parsed by extractor (JSON + required keys)", frac("parsed_ok"))
    add("Valid JSON and passes schema validation", frac("schema_ok"))
    add("...and internally consistent", frac("consistent"))
    add("Entities: precision", lambda s: pct(s["entities"]["precision"]))
    add("Entities: recall", lambda s: pct(s["entities"]["recall"]))
    add("Entities: F1", lambda s: pct(s["entities"]["f1"]))
    add("Entities: TP / FP / FN", lambda s: "{tp} / {fp} / {fn}".format(**s["entities"]))
    add("Entities: recall on name only", lambda s: pct(s["entities"]["name_only_recall"]))
    add("Actions: precision", lambda s: pct(s["actions"]["precision"]))
    add("Actions: recall", lambda s: pct(s["actions"]["recall"]))
    add("Actions: F1", lambda s: pct(s["actions"]["f1"]))
    add("Actions: TP / FP / FN", lambda s: "{tp} / {fp} / {fn}".format(**s["actions"]))
    add("Actions: owner accuracy (matched)", lambda s: pct(s["actions"]["owner_accuracy"]))
    add("Actions: deadline accuracy (matched)", lambda s: pct(s["actions"]["deadline_accuracy"]))
    add("Actions: priority accuracy (where labelled)",
        lambda s: f"{pct(s['actions']['priority_accuracy'])} (n={s['actions']['priority_scored']})")
    add("Actions: found with all fields correct",
        lambda s: f"{s['actions']['all_fields_correct']}/{s['actions']['gold_actions']} ({pct(s['actions']['exact_action_rate'])})")
    add("Classify: primary category accuracy",
        lambda s: f"{s['classify']['primary_correct']}/{s['classify']['documents_scored']} ({pct(s['classify']['primary_accuracy'])})")
    add("Classify: label precision", lambda s: pct(s["classify"]["precision"]))
    add("Classify: label recall", lambda s: pct(s["classify"]["recall"]))
    add("Classify: language accuracy",
        lambda s: f"{s['classify']['language_correct']}/{s['classify']['documents_scored']} ({pct(s['classify']['language_accuracy'])})")
    for stat in ("mean", "median", "min", "max"):
        add(f"Latency per call, {stat} (s)",
            lambda s, stat=stat: f"{s['latency_s'][stat]:.2f}" if "latency_s" in s else "n/a")

    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    out = "\n".join(lines) + "\n"
    for name in names:
        out += f"\n### Failures: {name}\n\n"
        if not summaries[name]["failures"]:
            out += "None.\n"
        for failure in summaries[name]["failures"]:
            out += f"- `{failure['id']}`\n"
            out += "".join(f"  - {p}\n" for p in failure["problems"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", metavar="PROVIDER:MODEL",
                        help="provider and model to evaluate; repeat for several. "
                             "Default: each provider's default model from extractor.py")
    parser.add_argument("--label", default="latest", help="results go to eval/results/<label>/")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="max_tokens passed to the provider (default: 1024, the extractor default)")
    parser.add_argument("--score-only", action="store_true",
                        help="recompute scores from saved raw outputs, without calling any API")
    args = parser.parse_args()

    dataset = load_dataset()
    items = {item["id"]: item for item in dataset}
    out_dir = EVAL_DIR / "results" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.score_only:
        runs = args.run or [f"{p}:{cfg['default_model']}" for p, cfg in extractor.PROVIDERS.items()]
        for spec in runs:
            provider, _, model = spec.partition(":")
            if provider not in extractor.PROVIDERS or not model:
                parser.error(f"--run expects PROVIDER:MODEL with a known provider, got '{spec}'")
            env_key = extractor.PROVIDERS[provider]["env_key"]
            if not os.environ.get(env_key):
                print(f"Skipping {spec}: {env_key} is not set.")
                continue
            print(f"Running {spec} on {len(dataset)} documents")
            records = run_model(provider, model, dataset, args.max_tokens)
            with open(output_path(out_dir, provider, model), "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summaries = {}
    for path in sorted(out_dir.glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if records:
            summaries[f"{records[0]['provider']}:{records[0]['model']}"] = summarize_run(records, items)
    if not summaries:
        print("No results to score.")
        return

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = render_markdown(summaries)
    (out_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"Raw outputs and summaries are in {out_dir}")


if __name__ == "__main__":
    main()
