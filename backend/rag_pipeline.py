"""
rag_pipeline.py
---------------
Glue layer: question -> retrieve -> generate -> structured answer.

Keeping orchestration here (not in main.py) means main.py stays as a thin HTTP
shell, and you can call ask() directly from a script or notebook for evals.
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


def _to_source(chunk: RetrievedChunk, snippet_chars: int = 240) -> Source:
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
# from a 3.8B-parameter model.
_INLINE_CITATION = re.compile(
    r"\s*\(\s*[A-Za-z0-9._\-]+\.pdf\s*,\s*p\.?\s*\d+\s*\)",
    flags=re.IGNORECASE,
)


def _strip_model_citations(text: str) -> str:
    """Remove any (filename.pdf, p. N) the model snuck in. Belt-and-braces with the prompt."""
    return _INLINE_CITATION.sub("", text).strip()


def ask(query: str, top_k: int = 4, source: Optional[str] = None) -> Answer:
    """
    Run the full RAG loop.

    If `source` is provided, retrieval is restricted to chunks from that filename.

    If retrieval comes up empty (e.g. user hasn't run /ingest yet) we return early
    with a clear message rather than letting the LLM hallucinate without context.
    """
    query = (query or "").strip()
    if not query:
        return Answer(answer="Please ask a question.", sources=[])

    chunks = search(query, top_k=top_k, source=source)
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

    raw_answer = generate(query, chunks)
    clean_answer = _strip_model_citations(raw_answer)
    return Answer(
        answer=clean_answer,
        sources=[_to_source(c) for c in chunks],
    )


def ask_dict(query: str, top_k: int = 4, source: Optional[str] = None) -> dict:
    """Same as ask(), but returns a plain dict (handy for FastAPI / JSON)."""
    a = ask(query, top_k=top_k, source=source)
    return {
        "answer": a.answer,
        "sources": [asdict(s) for s in a.sources],
    }
