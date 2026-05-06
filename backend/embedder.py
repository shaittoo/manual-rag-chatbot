"""
embedder.py
-----------
Thin wrapper around sentence-transformers.

Why a wrapper?
- Keeps the embedding model name in one place so we can swap it later
  (e.g. all-MiniLM-L6-v2 -> bge-small-en-v1.5) without touching retriever logic.
- Lazy-loads the model: the first call downloads/loads weights; later calls reuse
  the cached instance. This matters because FastAPI workers reuse the module.
"""

from __future__ import annotations

from functools import lru_cache
import torch
from typing import List

from sentence_transformers import SentenceTransformer

# V2 change (was: sentence-transformers/all-MiniLM-L6-v2):
# bge-small-en-v1.5 is also 384-dim and ~130MB, but trained with a more recent
# contrastive objective and consistently outperforms MiniLM on retrieval
# benchmarks (e.g. MTEB). Vector dimensionality matches MiniLM, but the vector
# space is different — ChromaDB MUST be re-ingested after this change.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def _get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedder running on: {device}", flush=True)
    return SentenceTransformer(model_name, device=device)


def embed_texts(texts: List[str], model_name: str = DEFAULT_MODEL) -> List[List[float]]:
    """
    Embed a batch of documents/chunks.

    Returns a list of float lists (Chroma accepts this directly). We don't return
    numpy arrays to keep the boundary with Chroma simple and JSON-friendly.
    """
    if not texts:
        return []
    model = _get_model(model_name)
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine similarity becomes a dot product
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(text: str, model_name: str = DEFAULT_MODEL) -> List[float]:
    """Embed a single user question. Same model, same normalization, for fair similarity."""
    return embed_texts([text], model_name=model_name)[0]
