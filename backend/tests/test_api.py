"""
API-level tests for main.py using FastAPI's TestClient.

We never load a real model here. Endpoints that would call the RAG pipeline or
the classifier are monkeypatched, so these tests verify the HTTP contract:
request validation, status codes, and response shaping — fast and deterministic.
"""

from __future__ import annotations

import main
from fastapi.testclient import TestClient

client = TestClient(main.app)


# --- /health -------------------------------------------------------------

def test_health_ok():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


# --- /ask request validation (no model needed) --------------------------

def test_ask_rejects_empty_query():
    res = client.post("/ask", json={"query": ""})
    assert res.status_code == 422  # min_length=1


def test_ask_rejects_out_of_range_top_k():
    res = client.post("/ask", json={"query": "hi", "top_k": 21})
    assert res.status_code == 422  # le=20


def test_ask_rejects_unknown_generator_backend():
    res = client.post(
        "/ask",
        json={"query": "hi", "generator_backend": "gpt5"},
    )
    assert res.status_code == 422  # Literal["transformers","ollama"]


# --- /ask happy path (pipeline monkeypatched) ---------------------------

def test_ask_returns_answer_and_sources(monkeypatch):
    def fake_ask_dict(query, top_k=4, source=None, generator_backend="transformers", history=None):
        assert query == "How do I drain it?"
        return {
            "answer": "Open the drain valve.",
            "sources": [{"source": "db05a9.pdf", "page": 3, "score": 0.9, "snippet": "..."}],
        }

    monkeypatch.setattr(main, "ask_dict", fake_ask_dict)

    res = client.post("/ask", json={"query": "How do I drain it?", "source": "db05a9.pdf"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "Open the drain valve."
    assert body["sources"][0]["source"] == "db05a9.pdf"


# --- /sources ------------------------------------------------------------

def test_sources_lists_indexed_filenames(monkeypatch):
    monkeypatch.setattr(main, "list_sources", lambda: ["a.pdf", "b.pdf"])
    res = client.get("/sources")
    assert res.status_code == 200
    assert res.json() == {"sources": ["a.pdf", "b.pdf"]}


# --- /classify + /ask_auto (classifier monkeypatched) -------------------

def test_classify_returns_routing(monkeypatch):
    def fake_predict_full(query):
        return {
            "predicted_source": "db05a9.pdf",
            "confidence": 0.87,
            "distribution": [{"source": "db05a9.pdf", "prob": 0.87}],
        }

    monkeypatch.setattr(main, "_load_classifier_or_fail", lambda: fake_predict_full)

    res = client.post("/classify", json={"query": "washer won't fill"})
    assert res.status_code == 200
    body = res.json()
    assert body["predicted_source"] == "db05a9.pdf"
    assert body["confidence"] == 0.87


def test_ask_auto_routes_then_answers(monkeypatch):
    def fake_predict_full(query):
        return {"predicted_source": "db05a9.pdf", "confidence": 0.91, "distribution": []}

    def fake_ask_dict(query, top_k=4, source=None, generator_backend="transformers", history=None):
        # ask_auto must forward the routed source into the pipeline.
        assert source == "db05a9.pdf"
        return {"answer": "Check the inlet hoses.", "sources": []}

    monkeypatch.setattr(main, "_load_classifier_or_fail", lambda: fake_predict_full)
    monkeypatch.setattr(main, "ask_dict", fake_ask_dict)

    res = client.post("/ask_auto", json={"query": "no water coming in"})
    assert res.status_code == 200
    body = res.json()
    assert body["routed_to"] == "db05a9.pdf"
    assert body["routing_confidence"] == 0.91
    assert body["answer"] == "Check the inlet hoses."
