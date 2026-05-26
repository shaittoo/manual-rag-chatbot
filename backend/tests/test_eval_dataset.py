"""
Validation tests for the evaluation dataset (eval/questions.json).

These are the automatable half of the "eval test cases": they don't measure
answer quality (that needs a live model via run_eval.py), but they guarantee the
dataset itself is well-formed, so a malformed entry can't silently corrupt an
eval run or skew reported accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

# run_eval.py looks in PROJECT_ROOT/eval first, then BACKEND_DIR/eval.
_candidates = [PROJECT_ROOT / "eval" / "questions.json", BACKEND_DIR / "eval" / "questions.json"]
QUESTIONS_PATH = next((p for p in _candidates if p.exists()), _candidates[0])

REQUIRED_KEYS = {"id", "manual_filename", "question", "reference_answer", "type"}
ALLOWED_TYPES = {"factual", "procedural", "tricky"}


@pytest.fixture(scope="module")
def questions():
    assert QUESTIONS_PATH.exists(), f"questions.json not found at {QUESTIONS_PATH}"
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "questions.json must be a JSON array"
    return data


def test_dataset_is_non_empty(questions):
    assert len(questions) > 0


def test_every_question_has_required_keys(questions):
    for i, q in enumerate(questions):
        missing = REQUIRED_KEYS - set(q)
        assert not missing, f"question #{i} ({q.get('id')}) missing keys: {missing}"


def test_ids_are_unique(questions):
    ids = [q["id"] for q in questions]
    dupes = {x for x in ids if ids.count(x) > 1}
    assert not dupes, f"duplicate question ids: {dupes}"


def test_types_are_in_allowed_set(questions):
    for q in questions:
        assert q["type"] in ALLOWED_TYPES, f"{q['id']} has bad type {q['type']!r}"


def test_questions_and_references_are_substantive(questions):
    for q in questions:
        assert q["question"].strip().endswith("?") or len(q["question"]) > 10
        # Reference answers should be real, gradeable text, not a stub.
        assert len(q["reference_answer"].strip()) >= 40, f"{q['id']} reference too short"


def test_manual_filenames_look_like_pdfs(questions):
    for q in questions:
        assert q["manual_filename"].lower().endswith(".pdf"), q["id"]


def test_each_manual_has_coverage(questions):
    # The eval set is designed as several questions per manual; make sure no
    # manual is represented by only a single question (weak coverage).
    counts = {}
    for q in questions:
        counts[q["manual_filename"]] = counts.get(q["manual_filename"], 0) + 1
    thin = {m: c for m, c in counts.items() if c < 2}
    assert not thin, f"manuals with thin coverage (<2 questions): {thin}"
