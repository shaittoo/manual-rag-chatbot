"""
retriever.py
------------
PDF ingestion + ChromaDB-backed similarity search.

Pipeline:
    PDF files in manuals/  --(pypdf)-->  page text
                          --(chunker)-->  overlapping chunks
                          --(embedder)-->  vectors
                          --(Chroma)-->  persistent collection on disk

At query time we embed the question with the SAME model and ask Chroma for the
top-k nearest chunks (cosine similarity, since we normalized embeddings).

Design notes:
- We use a *persistent* Chroma client so the index survives restarts.
- Chunking is character-based with overlap. Token-based would be more accurate
  for the LLM context budget, but character-based is good enough for a mini-project
  and avoids dragging in a tokenizer here.
- We store metadata (source file, page number) so we can cite sources in the UI.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

from embedder import embed_query, embed_texts


# --- Paths ---------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
MANUALS_DIR = BACKEND_DIR / "manuals"
CHROMA_DIR = BACKEND_DIR / "chroma_db"
COLLECTION_NAME = "manuals"


# --- Data types ----------------------------------------------------------

@dataclass
class Chunk:
    """One retrievable unit of text + where it came from."""
    text: str
    source: str    # filename, e.g. "router_x500_manual.pdf"
    page: int      # 1-indexed page number


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float   # similarity score (higher = closer); Chroma returns distance, we invert


# --- PDF -> text ---------------------------------------------------------

def _read_pdf(path: Path) -> List[tuple[int, str]]:
    """Return [(page_number, text), ...] for a single PDF. Page numbers are 1-indexed."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # pypdf sometimes leaves weird whitespace and ligature artifacts.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages.append((i, text))
    return pages


# --- Chunking ------------------------------------------------------------

def _chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> List[str]:
    """
    Split text into overlapping windows.

    Why overlap? Important context (e.g. a step number and its description) often
    spans a paragraph boundary. Overlap lets at least one chunk see the full context.
    """
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # Try to end on a sentence/whitespace boundary for cleaner chunks.
        if end < n:
            last_break = max(
                text.rfind(". ", start, end),
                text.rfind("\n", start, end),
            )
            if last_break > start + chunk_size // 2:
                end = last_break + 1
        chunks.append(text[start:end].strip())
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def _build_chunks(manuals_dir: Path) -> List[Chunk]:
    """Walk the manuals folder, parse every PDF, and produce Chunk objects."""
    chunks: List[Chunk] = []
    pdfs = sorted(manuals_dir.glob("*.pdf"))
    for pdf_path in pdfs:
        for page_num, page_text in _read_pdf(pdf_path):
            for c in _chunk_text(page_text):
                chunks.append(Chunk(text=c, source=pdf_path.name, page=page_num))
    return chunks


# --- Chroma --------------------------------------------------------------

def _get_client() -> chromadb.api.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def _get_collection(client: chromadb.api.ClientAPI):
    # cosine works because we normalize embeddings in embedder.py
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# --- Public API ----------------------------------------------------------

def reset_index() -> None:
    """Drop the collection so the next ingest starts clean."""
    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # Collection may not exist yet — that's fine.
        pass


def ingest(manuals_dir: Optional[Path] = None) -> dict:
    """
    Re-index every PDF in `manuals_dir` (default: ./manuals).

    This wipes the existing collection and rebuilds it. For a mini-project that's
    simpler than incremental updates — and PDF manuals don't change often.

    Returns a small report dict: {"files": N, "chunks": M}.
    """
    manuals_dir = manuals_dir or MANUALS_DIR
    if not manuals_dir.exists():
        raise FileNotFoundError(f"manuals directory not found: {manuals_dir}")

    chunks = _build_chunks(manuals_dir)
    if not chunks:
        # Don't silently succeed — the caller almost certainly wants to know.
        raise RuntimeError(
            f"No text extracted from PDFs in {manuals_dir}. "
            "Are the PDFs scanned images? You'd need OCR for that."
        )

    reset_index()
    client = _get_client()
    coll = _get_collection(client)

    texts = [c.text for c in chunks]
    metadatas = [{"source": c.source, "page": c.page} for c in chunks]
    ids = [f"{c.source}::p{c.page}::i{i}" for i, c in enumerate(chunks)]
    embeddings = embed_texts(texts)

    # Chroma can blow up with very large single batches; chunk the inserts.
    BATCH = 256
    for i in range(0, len(ids), BATCH):
        coll.add(
            ids=ids[i:i + BATCH],
            documents=texts[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH],
        )

    return {
        "files": len({c.source for c in chunks}),
        "chunks": len(chunks),
    }


def search(
    query: str,
    top_k: int = 4,
    source: Optional[str] = None,
) -> List[RetrievedChunk]:
    """
    Return the top_k most similar chunks to `query`.

    If `source` is provided, restrict retrieval to chunks from that exact filename
    (e.g. source="db05a9.pdf"). This is the cheapest way to handle a multi-appliance
    corpus: instead of mixing chunks from a washer manual and a printer manual,
    the user picks which appliance they're asking about.
    """
    client = _get_client()
    coll = _get_collection(client)
    if coll.count() == 0:
        return []

    query_vec = embed_query(query)
    where = {"source": source} if source else None
    res = coll.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out: List[RetrievedChunk] = []
    for doc, meta, dist in zip(docs, metas, dists):
        # cosine distance in Chroma is (1 - cosine_similarity); convert back to similarity.
        score = 1.0 - float(dist)
        out.append(
            RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                page=int(meta.get("page", 0)),
                score=score,
            )
        )
    return out


def list_sources() -> List[str]:
    """
    Return all unique source filenames currently in the index.

    Used by the frontend to populate a "which manual?" dropdown so the user can
    scope their question to one appliance.
    """
    client = _get_client()
    coll = _get_collection(client)
    if coll.count() == 0:
        return []
    # Pull metadata only — cheaper than pulling documents/embeddings.
    got = coll.get(include=["metadatas"])
    metas = got.get("metadatas", []) or []
    sources = sorted({m.get("source") for m in metas if m and m.get("source")})
    return sources
