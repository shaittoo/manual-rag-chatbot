"""
Unit tests for rag_pipeline.py:

- _strip_model_citations : safety net that removes (file.pdf, p. N) the model
  may emit despite being told not to. Page numbers from a small local model are
  not trustworthy, so this must reliably strip them.
- _to_source            : turns a retrieved chunk into the citation object the
  frontend renders (snippet truncation + score rounding).
- _build_retrieval_query: enriches a follow-up question with recent history so
  retrieval understands pronouns like "that".
- ask                   : end-to-end glue, with retrieve/generate monkeypatched
  so we test orchestration without loading any model.
"""

from __future__ import annotations

import rag_pipeline
from rag_pipeline import (
    Answer,
    _build_retrieval_query,
    _strip_model_citations,
    _to_source,
    ask,
)
from retriever import RetrievedChunk


# --- _strip_model_citations ---------------------------------------------

def test_strips_basic_citation():
    text = "Open the valve to drain the antifreeze (db05a9.pdf, p. 12)."
    assert _strip_model_citations(text) == "Open the valve to drain the antifreeze."


def test_strips_citation_case_insensitively_and_loose_spacing():
    text = "Do the thing (C06184015.PDF ,p.3 ) now."
    assert "PDF" not in _strip_model_citations(text)
    assert "p.3" not in _strip_model_citations(text)


def test_strips_multiple_citations():
    text = "Step one (a.pdf, p. 1) then step two (b.pdf, p. 22)."
    cleaned = _strip_model_citations(text)
    assert ".pdf" not in cleaned
    assert cleaned.startswith("Step one")
    assert "step two" in cleaned


def test_leaves_normal_parentheses_untouched():
    text = "Turn the dial clockwise (to the right) until it clicks."
    assert _strip_model_citations(text) == text


# --- _to_source ----------------------------------------------------------

def test_to_source_rounds_score_and_preserves_metadata():
    chunk = RetrievedChunk(text="hello world", source="db05a9.pdf", page=7, score=0.123456)
    src = _to_source(chunk)
    assert src.source == "db05a9.pdf"
    assert src.page == 7
    assert src.score == 0.1235  # rounded to 4 dp
    assert src.snippet == "hello world"


def test_to_source_truncates_long_snippet_with_ellipsis():
    long_text = "word " * 400  # 2000 chars
    src = _to_source(chunk_with_text(long_text), )
    assert src.snippet.endswith("…")
    assert len(src.snippet) <= 601  # 600 cap + ellipsis char


def chunk_with_text(text: str) -> RetrievedChunk:
    return RetrievedChunk(text=text, source="x.pdf", page=1, score=0.5)


# --- _build_retrieval_query ---------------------------------------------

def test_retrieval_query_without_history_is_just_the_query():
    assert _build_retrieval_query("How do I drain it?", None) == "How do I drain it?"


def test_retrieval_query_with_history_includes_prior_turns():
    history = [
        {"role": "user", "content": "How do I drain antifreeze from my washer?"},
        {"role": "assistant", "content": "Open the drain valve..."},
    ]
    out = _build_retrieval_query("Can I add laundry during that?", history)
    assert "Conversation so far:" in out
    assert "drain antifreeze" in out
    assert out.strip().endswith("Can I add laundry during that?")


def test_retrieval_query_truncates_very_long_history_turn():
    history = [{"role": "user", "content": "x" * 1000}]
    out = _build_retrieval_query("now what?", history)
    assert "..." in out  # long turn was truncated


# --- ask (orchestration) -------------------------------------------------

def test_ask_empty_query_returns_prompt_to_ask():
    result = ask("   ")
    assert isinstance(result, Answer)
    assert result.answer == "Please ask a question."
    assert result.sources == []


def test_ask_with_no_retrieval_results_returns_no_manuals_message(monkeypatch):
    monkeypatch.setattr(rag_pipeline, "search", lambda *a, **k: [])
    result = ask("anything")
    assert "indexed manuals" in result.answer.lower()
    assert result.sources == []


def test_ask_with_source_and_no_results_names_the_source(monkeypatch):
    monkeypatch.setattr(rag_pipeline, "search", lambda *a, **k: [])
    result = ask("anything", source="db05a9.pdf")
    assert "db05a9.pdf" in result.answer
    assert result.sources == []


def test_ask_happy_path_strips_citations_and_returns_sources(monkeypatch):
    fake_chunks = [
        RetrievedChunk(text="Open the drain valve.", source="db05a9.pdf", page=3, score=0.9),
    ]
    monkeypatch.setattr(rag_pipeline, "search", lambda *a, **k: fake_chunks)
    monkeypatch.setattr(
        rag_pipeline,
        "generate",
        lambda *a, **k: "Open the drain valve (db05a9.pdf, p. 3).",
    )

    result = ask("How do I drain it?", source="db05a9.pdf")

    # Citation must be stripped from the model output.
    assert ".pdf" not in result.answer
    assert result.answer == "Open the drain valve."
    # Sources are surfaced separately, derived from the retrieved chunks.
    assert len(result.sources) == 1
    assert result.sources[0].source == "db05a9.pdf"
    assert result.sources[0].page == 3
