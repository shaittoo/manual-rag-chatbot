"""
Unit tests for generator.py prompt-assembly helpers and backend selection.

These exercise the parts of the generator that do NOT require a loaded model:
- _format_context        : numbered context block from retrieved chunks
- _format_history        : recent-turn formatting + role normalization + caps
- _build_messages        : the shared system/user message structure
- _messages_to_plain_prompt : Ollama plain-text rendering of those messages
- get_generator          : factory validation
- OllamaGenerator.generate when the server is unreachable -> graceful fallback
"""

from __future__ import annotations

import pytest

import generator
from generator import (
    OllamaGenerator,
    _build_messages,
    _format_context,
    _format_history,
    _messages_to_plain_prompt,
    get_generator,
)
from retriever import RetrievedChunk


def _chunks():
    return [
        RetrievedChunk(text="First passage.", source="a.pdf", page=1, score=0.9),
        RetrievedChunk(text="Second passage.", source="b.pdf", page=5, score=0.8),
    ]


# --- _format_context -----------------------------------------------------

def test_format_context_empty_returns_placeholder():
    assert _format_context([]) == "(no context retrieved)"


def test_format_context_numbers_each_chunk_with_source_and_page():
    out = _format_context(_chunks())
    assert "[1] a.pdf, p. 1" in out
    assert "[2] b.pdf, p. 5" in out
    assert "First passage." in out
    assert "Second passage." in out


# --- _format_history -----------------------------------------------------

def test_format_history_empty_returns_placeholder():
    assert _format_history(None) == "(no previous conversation)"
    assert _format_history([]) == "(no previous conversation)"


def test_format_history_normalizes_unknown_role_to_user():
    history = [{"role": "system", "content": "ignore me"}]
    out = _format_history(history)
    assert out.startswith("user:")


def test_format_history_truncates_long_turns():
    history = [{"role": "user", "content": "z" * 1000}]
    out = _format_history(history)
    assert out.endswith("...")


def test_format_history_keeps_only_recent_turns():
    history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
    out = _format_history(history)
    # Only the last 6 turns are retained.
    assert "msg 19" in out
    assert "msg 0" not in out


# --- _build_messages / plain prompt --------------------------------------

def test_build_messages_has_system_and_user_roles():
    msgs = _build_messages("How do I reset it?", _chunks())
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user"]
    assert "How do I reset it?" in msgs[1]["content"]


def test_plain_prompt_includes_system_user_and_assistant_markers():
    msgs = _build_messages("Question here", _chunks())
    prompt = _messages_to_plain_prompt(msgs)
    assert "System:" in prompt
    assert "User:" in prompt
    assert prompt.rstrip().endswith("Assistant:")


# --- get_generator factory ----------------------------------------------

def test_get_generator_rejects_unknown_backend():
    with pytest.raises(ValueError):
        get_generator("not-a-real-backend")


def test_get_generator_returns_ollama_without_loading_a_model():
    # OllamaGenerator does no model loading at construction, so this is safe.
    gen = get_generator("ollama")
    assert isinstance(gen, OllamaGenerator)


# --- Ollama offline behaviour --------------------------------------------

def test_ollama_generate_returns_friendly_message_when_unreachable():
    # Point at a port nothing is listening on so urlopen raises URLError, which
    # the backend is supposed to catch and turn into a user-facing message.
    gen = OllamaGenerator(url="http://127.0.0.1:9/api/generate", timeout=1)
    out = gen.generate("Why won't it cool?", _chunks())
    assert "Ollama" in out
    assert "could not" in out.lower()
