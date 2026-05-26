"""
embedder.py
-----------
Thin wrapper around sentence-transformers.

Final retrieval setup:
- Model: BAAI/bge-small-en-v1.5
- Output vector size: 384 dimensions
- Embeddings are normalized for cosine similarity in ChromaDB.

Important:
- ChromaDB must be re-ingested whenever the embedding model changes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import torch
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Final V2 embedder
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# V1 baseline embedder, kept only for reference:
# DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------

def get_device() -> str:
    """Use CUDA if available; otherwise fall back to CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Cache the embedding model in memory so it only loads once per process.
    """
    device = get_device()
    print(f"Embedder model: {model_name}", flush=True)
    print(f"Embedder running on: {device}", flush=True)

    return SentenceTransformer(
        model_name,
        device=device,
    )


# ---------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------

def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
) -> List[List[float]]:
    """
    Embed a batch of manual chunks.

    Returns a list of float lists because Chroma accepts this format directly.
    """
    if not texts:
        return []

    device = get_device()
    model = _get_model(model_name)

    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device,
    )

    return vectors.tolist()


def embed_query(
    text: str,
    model_name: str = DEFAULT_MODEL,
) -> List[float]:
    """
    Embed one user query using the same model and normalization as the chunks.
    """
    return embed_texts([text], model_name=model_name)[0]