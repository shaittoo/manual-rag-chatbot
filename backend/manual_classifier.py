"""
manual_classifier.py
--------------------
Small FFN that predicts which appliance manual a user's query is about.

Architecture (frozen MiniLM embedder + trainable head):

    query (str)
        │
        ▼
    [sentence-transformers/all-MiniLM-L6-v2]   <-- frozen, no gradient
        │  384-dim vector
        ▼
    Linear(384 -> 128)
    ReLU
    Dropout(0.2)
    Linear(128 -> 5)          <-- one logit per manual
        │
        ▼
    softmax -> P(manual_i | query)

Why this design:
- Frozen MiniLM means we don't backprop through 22M params we can't afford to
  retrain on 25 questions. We only train the small head (~50K params).
- 5 classes = 5 manuals. Output index maps to the filename via LABELS.
- Cross-entropy loss + Adam, standard supervised classification.

Inference path:
    ManualClassifier.predict("how do I drain antifreeze?")
        -> ("db05a9.pdf", 0.92)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from embedder import _get_model as _get_embedder_model

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_WEIGHTS = BACKEND_DIR / "manual_classifier.pt"
LABELS_FILE = BACKEND_DIR / "manual_classifier_labels.json"

# Model hyperparameters (kept here so they're identical between train + inference).
EMBEDDING_DIM = 384      # MiniLM-L6-v2 output dim
HIDDEN_DIM = 128
DROPOUT = 0.2
N_CLASSES = 5            # 5 appliance manuals


class ManualClassifier(nn.Module):
    """Tiny feed-forward classifier head over frozen MiniLM embeddings."""

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_classes: int = N_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (apply softmax for probabilities)."""
        return self.net(x)


# --- Inference helpers ---------------------------------------------------

@lru_cache(maxsize=1)
def _load_labels() -> List[str]:
    """Load the ordered list of manual filenames. Index -> filename."""
    if not LABELS_FILE.exists():
        raise FileNotFoundError(
            f"Label map not found at {LABELS_FILE}. "
            "Run train_classifier.py first."
        )
    with LABELS_FILE.open("r", encoding="utf-8") as f:
        return list(json.load(f))


@lru_cache(maxsize=1)
def _load_classifier() -> ManualClassifier:
    """Load trained weights into a fresh ManualClassifier instance."""
    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Trained weights not found at {MODEL_WEIGHTS}. "
            "Run train_classifier.py first."
        )
    model = ManualClassifier()
    state = torch.load(MODEL_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _embed_query(query: str) -> torch.Tensor:
    """Embed a query with the same MiniLM the classifier was trained on."""
    embedder = _get_embedder_model("sentence-transformers/all-MiniLM-L6-v2")
    vec = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return torch.tensor(vec, dtype=torch.float32)


def predict(query: str) -> Tuple[str, float]:
    """
    Predict which manual a query is about.

    Returns (filename, confidence) where confidence is the softmax probability
    assigned to the predicted class.
    """
    classifier = _load_classifier()
    labels = _load_labels()

    with torch.no_grad():
        x = _embed_query(query)
        logits = classifier(x)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        idx = int(torch.argmax(probs).item())

    return labels[idx], float(probs[idx].item())


def predict_full(query: str) -> dict:
    """
    Predict + return the full probability distribution.
    Useful for debugging and for showing 'top-3 candidates' in the UI.
    """
    classifier = _load_classifier()
    labels = _load_labels()

    with torch.no_grad():
        x = _embed_query(query)
        logits = classifier(x)
        probs = F.softmax(logits, dim=-1).squeeze(0).tolist()

    distribution = sorted(
        [{"source": labels[i], "probability": round(probs[i], 4)}
         for i in range(len(labels))],
        key=lambda d: -d["probability"],
    )
    return {
        "predicted_source": distribution[0]["source"],
        "confidence": distribution[0]["probability"],
        "distribution": distribution,
    }
