"""
Unit tests for retriever._chunk_text — the text-windowing logic that turns a
page of manual text into overlapping retrievable chunks.

These tests pin the V1 baseline behaviour (chunk_size=800, overlap=120) so a
future tweak to the chunker is a deliberate, visible change rather than a silent
regression that quietly degrades retrieval.
"""

from __future__ import annotations

from retriever import _chunk_text


def test_empty_text_returns_no_chunks():
    assert _chunk_text("") == []


def test_short_text_is_a_single_chunk():
    text = "Drain the antifreeze before storing the washer for winter."
    chunks = _chunk_text(text, chunk_size=800, overlap=120)
    assert chunks == [text]


def test_long_text_is_split_into_multiple_chunks():
    text = "word " * 1000  # ~5000 chars, well over one window
    chunks = _chunk_text(text, chunk_size=800, overlap=120)
    assert len(chunks) > 1


def test_no_chunk_exceeds_window_by_much():
    # The boundary-snapping logic can push an end slightly past chunk_size,
    # but never beyond the next breakpoint, so allow a small slack.
    text = "sentence. " * 500
    chunks = _chunk_text(text, chunk_size=800, overlap=120)
    assert all(len(c) <= 900 for c in chunks)


def test_consecutive_chunks_overlap():
    # With no sentence/newline breaks, windowing is purely length-based, so the
    # tail of one chunk must reappear at the head of the next (overlap).
    text = "abcdefghij" * 200  # 2000 chars, no breakpoints
    chunks = _chunk_text(text, chunk_size=800, overlap=120)
    assert len(chunks) >= 2
    tail = chunks[0][-120:]
    assert tail in chunks[1]


def test_no_empty_chunks_emitted():
    text = "First sentence. \n\n   \n Second sentence. " * 50
    chunks = _chunk_text(text)
    assert all(c.strip() for c in chunks)


def test_prefers_sentence_boundary_when_available():
    # Build text where a sentence break sits comfortably inside the back half of
    # the first window; the chunker should end the chunk on that break.
    first = "A" * 500 + ". "
    rest = "B" * 800
    chunks = _chunk_text(first + rest, chunk_size=800, overlap=120)
    assert chunks[0].endswith(".")
