"""
retriever.py
------------
PDF ingestion + ChromaDB-backed similarity search.

Final V2 retrieval setup:
- Embedder: BAAI/bge-small-en-v1.5
- Chunk size / overlap: 500 / 80
- Initial retrieval: ChromaDB cosine similarity
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2

Pipeline:
    PDF files in manuals/  --(pypdf)-->  page text
                          --(chunker)-->  overlapping chunks
                          --(embedder)-->  vectors
                          --(Chroma)-->  persistent collection on disk

At query time:
    user query
        -> query embedding
        -> retrieve top candidate chunks from Chroma
        -> rerank candidates using cross-encoder
        -> return final top_k chunks
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import chromadb
import torch
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import CrossEncoder

from embedder import embed_query, embed_texts


# ---------------------------------------------------------------------
# FINAL V2 SETTINGS
# ---------------------------------------------------------------------

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVE_CANDIDATES = 20


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
MANUALS_DIR = BACKEND_DIR / "manuals"
CHROMA_DIR = BACKEND_DIR / "chroma_db"
COLLECTION_NAME = "manuals"


# ---------------------------------------------------------------------
# DATA TYPES
# ---------------------------------------------------------------------

@dataclass
class Chunk:
    """One retrievable unit of text and its source location."""
    text: str
    source: str
    page: int


@dataclass
class RetrievedChunk:
    """One retrieved chunk returned to the RAG pipeline."""
    text: str
    source: str
    page: int
    score: float


# ---------------------------------------------------------------------
# RERANKER
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    """
    Load the cross-encoder reranker once per process.

    The reranker scores query-chunk pairs and helps reorder the initially
    retrieved Chroma results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Reranker model: {RERANKER_MODEL}", flush=True)
    print(f"Reranker running on: {device}", flush=True)

    return CrossEncoder(
        RERANKER_MODEL,
        device=device,
    )


# ---------------------------------------------------------------------
# PDF READING
# ---------------------------------------------------------------------

def _read_pdf(path: Path) -> List[tuple[int, str]]:
    """
    Return [(page_number, text), ...] for a single PDF.

    Page numbers are 1-indexed.
    """
    reader = PdfReader(str(path))
    pages: List[tuple[int, str]] = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        # Normalize common PDF extraction whitespace issues.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if text:
            pages.append((i, text))

    return pages


# ---------------------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------------------

def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping windows.

    Final V2:
    - chunk_size = 500
    - overlap = 80
    """
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        # Try to end on a sentence or line boundary for cleaner chunks.
        if end < n:
            last_break = max(
                text.rfind(". ", start, end),
                text.rfind("\n", start, end),
            )

            if last_break > start + chunk_size // 2:
                end = last_break + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        start = max(end - overlap, start + 1)

    return chunks


def _build_chunks(manuals_dir: Path) -> List[Chunk]:
    """
    Walk the manuals folder, parse every PDF, and produce Chunk objects.
    """
    chunks: List[Chunk] = []
    pdfs = sorted(manuals_dir.glob("*.pdf"))

    for pdf_path in pdfs:
        for page_num, page_text in _read_pdf(pdf_path):
            for chunk_text in _chunk_text(page_text):
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source=pdf_path.name,
                        page=page_num,
                    )
                )

    return chunks


# ---------------------------------------------------------------------
# CHROMA
# ---------------------------------------------------------------------

def _get_client() -> chromadb.api.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        ),
    )


def _get_collection(client: chromadb.api.ClientAPI):
    """
    Use cosine distance because embeddings are normalized in embedder.py.

    Chroma returns cosine distance, where lower distance means more similar.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------

def reset_index() -> None:
    """
    Drop the Chroma collection so the next ingest starts clean.
    """
    client = _get_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # Collection may not exist yet.
        pass


def ingest(manuals_dir: Optional[Path] = None) -> dict:
    """
    Re-index every PDF in manuals_dir.

    This wipes the existing Chroma collection and rebuilds it using the
    current embedder and chunking settings.
    """
    manuals_dir = manuals_dir or MANUALS_DIR

    if not manuals_dir.exists():
        raise FileNotFoundError(f"manuals directory not found: {manuals_dir}")

    chunks = _build_chunks(manuals_dir)

    if not chunks:
        raise RuntimeError(
            f"No text extracted from PDFs in {manuals_dir}. "
            "Are the PDFs scanned images? You would need OCR for that."
        )

    reset_index()

    client = _get_client()
    coll = _get_collection(client)

    texts = [c.text for c in chunks]
    metadatas = [{"source": c.source, "page": c.page} for c in chunks]
    ids = [f"{c.source}::p{c.page}::i{i}" for i, c in enumerate(chunks)]

    embeddings = embed_texts(texts)

    # Chroma can fail with very large single batches, so insert in batches.
    batch_size = 256

    for i in range(0, len(ids), batch_size):
        coll.add(
            ids=ids[i:i + batch_size],
            documents=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
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
    Return the top_k most relevant chunks to query.

    Final V2 behavior:
    1. Embed the query.
    2. Retrieve top candidate chunks from Chroma.
    3. Rerank those candidates with MS-MARCO MiniLM cross-encoder.
    4. Return top_k reranked chunks.
    """
    client = _get_client()
    coll = _get_collection(client)

    if coll.count() == 0:
        return []

    query_vec = embed_query(query)
    where = {"source": source} if source else None

    n_candidates = max(top_k, RETRIEVE_CANDIDATES)

    res = coll.query(
        query_embeddings=[query_vec],
        n_results=n_candidates,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    if not docs:
        return []

    reranker = _get_reranker()

    pairs = [(query, doc) for doc in docs]
    rerank_scores = reranker.predict(pairs)

    indexed = sorted(
        zip(docs, metas, rerank_scores),
        key=lambda x: float(x[2]),
        reverse=True,
    )[:top_k]

    out: List[RetrievedChunk] = []

    for doc, meta, ce_score in indexed:
        out.append(
            RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                page=int(meta.get("page", 0)),
                score=round(float(ce_score), 4),
            )
        )

    return out


def list_sources() -> List[str]:
    """
    Return all unique source filenames currently in the index.

    Used by the frontend to populate a manual/source dropdown.
    """
    client = _get_client()
    coll = _get_collection(client)

    if coll.count() == 0:
        return []

    got = coll.get(include=["metadatas"])
    metas = got.get("metadatas", []) or []

    sources = sorted(
        {
            m.get("source")
            for m in metas
            if m and m.get("source")
        }
    )

    return sources