"""
rag_pipeline.py
---------------
Glue layer: question -> retrieve -> generate -> structured answer.

Keeping orchestration here (not in main.py) means main.py stays as a thin HTTP
shell, and you can call ask() directly from a script or notebook for evals.

Supports:
- model switching via generator_backend
- conversational / follow-up questions via history
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import List, Optional

from generator import generate
from retriever import RetrievedChunk, search


@dataclass
class Source:
    """What we expose to the frontend for citation links."""
    source: str
    page: int
    score: float
    snippet: str  # short preview of the chunk text


@dataclass
class Answer:
    answer: str
    sources: List[Source]


def _to_source(chunk: RetrievedChunk, snippet_chars: int = 600) -> Source:
    snippet = chunk.text.strip().replace("\n", " ")

    if len(snippet) > snippet_chars:
        snippet = snippet[:snippet_chars].rsplit(" ", 1)[0] + "…"

    return Source(
        source=chunk.source,
        page=chunk.page,
        score=round(chunk.score, 4),
        snippet=snippet,
    )


# Regex: find any "(filename.pdf, p. 12)" that the model might still emit despite
# being told not to. We strip these because page numbers are not trustworthy
# from a small local language model.
_INLINE_CITATION = re.compile(
    r"\s*\(\s*[A-Za-z0-9._\-]+\.pdf\s*,\s*p\.?\s*\d+\s*\)",
    flags=re.IGNORECASE,
)


def _strip_model_citations(text: str) -> str:
    """Remove any (filename.pdf, p. N) the model snuck in."""
    return _INLINE_CITATION.sub("", text).strip()


def _build_retrieval_query(
    query: str,
    history: Optional[List[dict]] = None,
) -> str:
    """
    Make retrieval aware of follow-up questions without requiring another LLM call.

    Example:
        Previous: "How do I drain antifreeze from my washer?"
        Current:  "Can I add laundry during that?"

    If we retrieve using only "Can I add laundry during that?",
    Chroma may not know what "that" refers to.

    So we enrich the retrieval query with recent chat history.
    """
    query = (query or "").strip()

    if not history:
        return query

    lines = []

    for msg in history[-6:]:
        role = msg.get("role", "")
        content = " ".join((msg.get("content") or "").split())

        if not content:
            continue

        if len(content) > 500:
            content = content[:500] + "..."

        lines.append(f"{role}: {content}")

    if not lines:
        return query

    return (
        "Conversation so far:\n"
        + "\n".join(lines)
        + "\n\nCurrent question:\n"
        + query
    )


def ask(
    query: str,
    top_k: int = 4,
    source: Optional[str] = None,
    generator_backend: str = "transformers",
    history: Optional[List[dict]] = None,
) -> Answer:
    """
    Run the full RAG loop.

    If `source` is provided, retrieval is restricted to chunks from that filename.

    If `history` is provided, retrieval and generation become conversation-aware,
    which helps with follow-up questions like:
        "What about after that?"
        "Can I do that again?"
        "What if it still does not work?"

    If retrieval comes up empty, we return early with a clear message rather
    than letting the LLM hallucinate without context.
    """
    query = (query or "").strip()

    if not query:
        return Answer(answer="Please ask a question.", sources=[])

    retrieval_query = _build_retrieval_query(query, history)

    chunks = search(
        retrieval_query,
        top_k=top_k,
        source=source,
    )

    if not chunks:
        msg = (
            f"No matching content found in '{source}'."
            if source
            else (
                "I don't have any indexed manuals yet. "
                "Drop PDFs into backend/manuals/ and POST to /ingest to build the index."
            )
        )

        return Answer(answer=msg, sources=[])

    raw_answer = generate(
        query,
        chunks,
        generator_backend=generator_backend,
        history=history,
    )

    clean_answer = _strip_model_citations(raw_answer)

    return Answer(
        answer=clean_answer,
        sources=[_to_source(c) for c in chunks],
    )


def ask_dict(
    query: str,
    top_k: int = 4,
    source: Optional[str] = None,
    generator_backend: str = "transformers",
    history: Optional[List[dict]] = None,
) -> dict:
    """Same as ask(), but returns a plain dict for FastAPI / JSON."""
    a = ask(
        query,
        top_k=top_k,
        source=source,
        generator_backend=generator_backend,
        history=history,
    )

    return {
        "answer": a.answer,
        "sources": [asdict(s) for s in a.sources],
    }