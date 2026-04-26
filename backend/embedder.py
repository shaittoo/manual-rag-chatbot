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
from typing import List

from sentence_transformers import SentenceTransformer

# 384-dim, ~80MB, fast on CPU. Good default for English manuals.
# If your manuals are technical/code-heavy, consider "BAAI/bge-small-en-v1.5".
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Cache the model in memory so we only pay the load cost once per process."""
    return SentenceTransformer(model_name)


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
