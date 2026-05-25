"""
embedder.py
-----------
Thin wrapper around sentence-transformers.

Why a wrapper?
- Keeps the embedding model name in one place so we can swap it later
  without touching retriever logic.
- Lazy-loads the model: the first call downloads/loads weights; later calls reuse
  the cached instance. This matters because FastAPI workers reuse the module.

V1 baseline:
- Model: sentence-transformers/all-MiniLM-L6-v2
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

# V1 baseline model
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# V2 model, kept here for easy switching later:
# DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


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
    Cache the model in memory so we only pay the load cost once per process.
    """
    device = get_device()
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
    Embed a batch of documents/chunks.

    Returns a list of float lists because Chroma accepts this directly.
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
    Embed a single user question.

    Uses the same model and normalization as document chunk embeddings.
    """
    return embed_texts([text], model_name=model_name)[0]