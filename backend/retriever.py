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
top-k nearest chunks.

V1 baseline settings:
- Embedder: sentence-transformers/all-MiniLM-L6-v2
- Chunk size / overlap: 800 / 120
- Reranker: none
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

from embedder import embed_query, embed_texts


# ---------------------------------------------------------------------
# V2 RERANKER SETTINGS — COMMENTED OUT FOR V1 BASELINE
# ---------------------------------------------------------------------
#
# V2 used a cross-encoder reranker:
#
# import torch
# from functools import lru_cache
# from sentence_transformers import CrossEncoder
#
# RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# RETRIEVE_CANDIDATES = 20
#
#
# @lru_cache(maxsize=1)
# def _get_reranker() -> CrossEncoder:
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Reranker running on: {device}", flush=True)
#     return CrossEncoder(RERANKER_MODEL, device=device)


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
# PDF READING
# ---------------------------------------------------------------------

def _read_pdf(path: Path) -> List[tuple[int, str]]:
    """
    Return [(page_number, text), ...] for a single PDF.
    Page numbers are 1-indexed.
    """
    reader = PdfReader(str(path))
    pages = []

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
    chunk_size: int = 800,
    overlap: int = 120,
) -> List[str]:
    """
    Split text into overlapping windows.

    V1 baseline:
    - chunk_size = 800
    - overlap = 120

    V2 changed this to:
    - chunk_size = 500
    - overlap = 80

    Keep V1 here for the baseline comparison.
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
    cosine works because embeddings are normalized in embedder.py.
    Chroma returns cosine distance, so lower distance means more similar.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------

def reset_index() -> None:
    """Drop the collection so the next ingest starts clean."""
    client = _get_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # Collection may not exist yet.
        pass


def ingest(manuals_dir: Optional[Path] = None) -> dict:
    """
    Re-index every PDF in `manuals_dir`.

    This wipes the existing Chroma collection and rebuilds it.
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

    # Chroma can blow up with very large single batches, so insert in chunks.
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
    Return the top_k most relevant chunks to `query`.

    V1 baseline:
    - Query is embedded with the same embedding model used during ingestion.
    - Chroma returns top_k chunks directly.
    - No cross-encoder reranking is applied.

    V2 behavior, currently commented out:
    - Retrieve top 20 candidates.
    - Rerank candidates using MS-MARCO MiniLM cross-encoder.
    - Return top_k reranked chunks.
    """
    client = _get_client()
    coll = _get_collection(client)

    if coll.count() == 0:
        return []

    query_vec = embed_query(query)
    where = {"source": source} if source else None

    # -----------------------------------------------------------------
    # V1 BASELINE RETRIEVAL
    # -----------------------------------------------------------------

    res = coll.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    if not docs:
        return []

    out: List[RetrievedChunk] = []

    for doc, meta, distance in zip(docs, metas, distances):
        # Chroma returns cosine distance. Convert to similarity-like score.
        similarity = 1.0 - float(distance)

        out.append(
            RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                page=int(meta.get("page", 0)),
                score=round(similarity, 4),
            )
        )

    return out

    # -----------------------------------------------------------------
    # V2 RERANKER RETRIEVAL — COMMENTED OUT FOR V1 BASELINE
    # -----------------------------------------------------------------
    #
    # n_candidates = max(top_k, RETRIEVE_CANDIDATES)
    #
    # res = coll.query(
    #     query_embeddings=[query_vec],
    #     n_results=n_candidates,
    #     where=where,
    #     include=["documents", "metadatas", "distances"],
    # )
    #
    # docs = res.get("documents", [[]])[0]
    # metas = res.get("metadatas", [[]])[0]
    #
    # if not docs:
    #     return []
    #
    # reranker = _get_reranker()
    # pairs = [(query, doc) for doc in docs]
    # rerank_scores = reranker.predict(pairs)
    #
    # indexed = sorted(
    #     zip(docs, metas, rerank_scores),
    #     key=lambda x: float(x[2]),
    #     reverse=True,
    # )[:top_k]
    #
    # out: List[RetrievedChunk] = []
    #
    # for doc, meta, ce_score in indexed:
    #     out.append(
    #         RetrievedChunk(
    #             text=doc,
    #             source=meta.get("source", "unknown"),
    #             page=int(meta.get("page", 0)),
    #             score=round(float(ce_score), 4),
    #         )
    #     )
    #
    # return out


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