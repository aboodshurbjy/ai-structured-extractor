"""Tests for the scoring rules in eval/run_eval.py. No API calls.

Run:  pytest eval/test_run_eval.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_eval  # noqa: E402


# -- normalization ------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    ("2026-02-09", "2026-02-09"),
    ("February 9, 2026", "2026-02-09"),
    ("9 Feb 2026", "2026-02-09"),
    ("by 9 February", "2026-02-09"),
    ("Feb 9th", "2026-02-09"),
    ("Friday", None),
    ("none", None),
    ("30 February 2026", None),
])
def test_parse_date(value, expected):
    assert run_eval.parse_date(value, default_year=2026) == expected


def test_norm_strips_case_punctuation_and_article():
    assert run_eval.norm("  The Tri-County  Water Authority. ") == "tri-county water authority"
    assert run_eval.norm(None) == ""


# -- gold labels score perfectly against themselves ---------------------------

def gold_as_prediction(item):
    expected = item["gold"]["expected"]
    if item["mode"] == "entities":
        entities = [{"name": e["name"], "type": e["type"], "context": ""} for e in expected["entities"]]
        return {"entities": entities, "entity_count": len(entities)}
    if item["mode"] == "actions":
        actions = [
            {"task": a["task"], "owner": a["owner"], "deadline": a["deadline"], "priority": a["priority"] or "medium"}
            for a in expected["actions"]
        ]
        return {"actions": actions, "action_count": len(actions)}
    if item["mode"] == "classify":
        return {
            "classifications": [{"category": c, "confidence": 0.9} for c in expected["required_categories"]],
            "primary_category": expected["primary_category"],
            "language": expected["language"],
        }
    return {**expected, "word_count_original": len(item["text"].split())}


def test_dataset_has_24_documents_six_per_mode():
    dataset = run_eval.load_dataset()
    assert len(dataset) == 24
    for mode in run_eval.MODES:
        assert sum(item["mode"] == mode for item in dataset) == 6


def test_gold_labels_score_perfectly_against_themselves():
    dataset = run_eval.load_dataset()
    items = {item["id"]: item for item in dataset}
    records = [
        {"id": item["id"], "mode": item["mode"], "provider": "gold", "model": "gold",
         "raw_response": json.dumps(gold_as_prediction(item)), "error": None, "latency_s": 0.1}
        for item in dataset
    ]
    summary = run_eval.summarize_run(records, items)
    assert summary["schema_ok"] == 24
    assert summary["failures"] == []
    assert summary["entities"]["f1"] == 1.0
    assert summary["actions"]["f1"] == 1.0
    assert summary["actions"]["exact_action_rate"] == 1.0
    assert summary["classify"]["primary_accuracy"] == 1.0


# -- entities -----------------------------------------------------------------

ENTITY_GOLD = {
    "entities": [
        {"name": "Amara Okafor", "type": "PERSON", "aliases": ["Dr. Amara Okafor"]},
        {"name": "St. Brigid Hospital", "type": "ORG", "type_accept": ["ORG", "LOCATION"]},
        {"name": "Austin", "type": "PERSON"},
        {"name": "Austin", "type": "LOCATION", "aliases": ["Austin, Texas"]},
    ],
    "optional_entities": [{"name": "last spring", "type": "DATE"}],
}


def entity_output(*pairs):
    return {"entities": [{"name": n, "type": t, "context": ""} for n, t in pairs]}


def test_entities_alias_and_alternate_type_count_as_matches():
    score = run_eval.score_entities(
        entity_output(("dr. amara okafor", "PERSON"), ("St. Brigid Hospital", "LOCATION"),
                      ("Austin", "PERSON"), ("Austin, Texas", "LOCATION")),
        ENTITY_GOLD,
    )
    assert (score["tp"], score["fp"], score["fn"]) == (4, 0, 0)


def test_entities_wrong_type_is_false_positive_and_false_negative():
    score = run_eval.score_entities(entity_output(("Amara Okafor", "ORG")), ENTITY_GOLD)
    assert score["tp"] == 0
    assert score["fp"] == 1
    assert score["type_errors"] == ["amara okafor (ORG)"]
    assert "Amara Okafor (PERSON)" in score["missed"]
    assert score["name_only_found"] == 1


def test_entities_optional_and_duplicates_are_ignored_and_extras_penalized():
    score = run_eval.score_entities(
        entity_output(("Amara Okafor", "PERSON"), ("Dr. Amara Okafor", "PERSON"),
                      ("last spring", "DATE"), ("diabetes", "OTHER")),
        ENTITY_GOLD,
    )
    assert score["tp"] == 1
    assert score["false_positives"] == ["diabetes (OTHER)"]
    assert sorted(score["ignored"]) == ["dr. amara okafor", "last spring"]


def test_entities_empty_gold_counts_only_false_positives():
    score = run_eval.score_entities(entity_output(("back door", "LOCATION")), {"entities": []})
    assert (score["tp"], score["fp"], score["fn"]) == (0, 1, 0)


# -- actions ------------------------------------------------------------------

ACTION_GOLD = {
    "actions": [
        {"task": "Write the migration script", "match_keywords": ["migration"],
         "owner": "Hana Suzuki", "owner_accept": ["hana", "hana suzuki"],
         "deadline": "2026-02-09", "deadline_accept": ["2026-02-09"], "priority": "high"},
        {"task": "Order banner stands", "match_keywords": ["banner"],
         "owner": "unassigned", "owner_accept": ["unassigned"],
         "deadline": "none", "deadline_accept": ["none"], "priority": None},
    ]
}


def test_actions_pairing_and_field_scoring():
    output = {"actions": [
        {"task": "Write DB migration script for orders", "owner": "Hana", "deadline": "Feb 9", "priority": "high"},
        {"task": "Order new banner stands", "owner": "Omar", "deadline": "none", "priority": "low"},
        {"task": "Celebrate", "owner": "unassigned", "deadline": "none", "priority": "low"},
    ]}
    score = run_eval.score_actions(output, ACTION_GOLD)
    assert (score["tp"], score["fp"], score["fn"]) == (2, 1, 0)
    assert score["owner_correct"] == 1          # "Omar" is wrong for the unassigned task
    assert score["deadline_correct"] == 2       # "Feb 9" parses to the gold date
    assert (score["priority_scored"], score["priority_correct"]) == (1, 1)  # null gold priority is not scored
    assert score["all_fields_correct"] == 1
    assert score["spurious"] == ["Celebrate"]


def test_actions_missing_action_is_false_negative():
    score = run_eval.score_actions({"actions": []}, ACTION_GOLD)
    assert (score["tp"], score["fp"], score["fn"]) == (0, 0, 2)


def test_actions_null_deadline_is_not_accepted_as_none():
    output = {"actions": [{"task": "Order banner stands", "owner": "unassigned", "deadline": None, "priority": "low"}]}
    score = run_eval.score_actions(output, ACTION_GOLD)
    assert score["deadline_correct"] == 0


# -- classify -----------------------------------------------------------------

CLASSIFY_GOLD = {
    "primary_category": "marketing", "required_categories": ["marketing"],
    "optional_categories": ["financial"], "language": "English", "language_accept": ["english", "en"],
}


def test_classify_optional_label_ignored_and_extra_label_penalized():
    output = {
        "classifications": [
            {"category": "marketing", "confidence": 0.9},
            {"category": "financial", "confidence": 0.4},
            {"category": "hr", "confidence": 0.2},
        ],
        "primary_category": "Marketing", "language": "en",
    }
    score = run_eval.score_classify(output, CLASSIFY_GOLD)
    assert score["primary_correct"] and score["language_correct"]
    assert (score["tp"], score["fp"], score["fn"]) == (1, 1, 0)
    assert score["extra_labels"] == ["hr"]
    # only marketing (0.9) clears the 0.5 confidence threshold
    assert (score["tp_confident"], score["fp_confident"], score["fn_confident"]) == (1, 0, 0)


# -- schema and consistency ---------------------------------------------------

def test_schema_errors_flag_bad_enum_type_and_missing_field():
    data = {"actions": [{"task": "x", "owner": None, "deadline": "none", "priority": "urgent"}]}
    errors = run_eval.schema_errors(data, "actions")
    assert any("action_count" in e for e in errors)
    assert any("'owner' has type NoneType" in e for e in errors)
    assert any("'priority' value 'urgent'" in e for e in errors)


def test_schema_accepts_conforming_classify_output():
    data = {"classifications": [{"category": "legal", "confidence": 1}], "primary_category": "legal", "language": "English"}
    assert run_eval.schema_errors(data, "classify") == []


def test_consistency_flags_wrong_count():
    data = {"entities": [{"name": "A", "type": "ORG", "context": ""}], "entity_count": 3}
    assert run_eval.consistency_errors(data, "entities", "text") != []


def test_evaluate_record_handles_failed_call_and_bad_json():
    item = {"id": "x", "mode": "entities", "text": "t", "gold": {"expected": {"entities": []}}}
    failed = run_eval.evaluate_record({"raw_response": None, "error": "boom"}, item)
    assert not failed["parsed_ok"] and failed["score"] is None
    bad = run_eval.evaluate_record({"raw_response": "not json", "error": None}, item)
    assert not bad["parsed_ok"] and "invalid JSON" in bad["parse_error"]
