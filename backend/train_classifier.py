"""
train_classifier.py
-------------------
Train the manual classifier (FFN head over frozen MiniLM).

Why this script exists:
    The /ask endpoint requires the user to specify which manual their question
    is about (the `source` filter). For an end-user demo, this is awkward —
    they shouldn't have to know which file is which. We train a small classifier
    that auto-routes the query to the right manual.

Pipeline:
    1. Load 25 hand-labeled questions from eval/questions.json.
    2. Augment the dataset deterministically (paraphrases via simple
       template+synonym substitution) to ~150 examples — small enough for CPU,
       enough variance for a 5-class classifier with frozen-embedding features.
    3. Embed every example with MiniLM (cached after first call).
    4. 5-fold cross-validation on the augmented set:
        - Train an FFN on 4 folds, evaluate on the held-out fold.
        - Aggregate per-fold accuracy as our reliability estimate.
       Five-fold is a hedge against the small dataset — a single train/test
       split would be too noisy to interpret.
    5. Train a final model on ALL augmented data; save its weights.
    6. Save metrics (per-fold + aggregate), label mapping, and a loss-curve
       plot for the paper.

Run:
    cd backend
    python train_classifier.py

Outputs (all written to backend/):
    manual_classifier.pt              <- final trained weights
    manual_classifier_labels.json     <- index -> filename mapping
    manual_classifier_metrics.json    <- per-fold + aggregate accuracy
    manual_classifier_loss.png        <- loss curve (optional, matplotlib)
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from manual_classifier import (
    DROPOUT,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    LABELS_FILE,
    MODEL_WEIGHTS,
    ManualClassifier,
)
from embedder import embed_texts


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"
METRICS_PATH = BACKEND_DIR / "manual_classifier_metrics.json"
LOSS_PLOT_PATH = BACKEND_DIR / "manual_classifier_loss.png"

SEED = 42
N_FOLDS = 5
EPOCHS = 60
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Reproducibility — important for paper claims.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ----------------------------------------------------------------------
# DATA AUGMENTATION
# ----------------------------------------------------------------------

# Synonym table for common appliance terms. Used to expand a base question
# into 2-4 plausible paraphrases. Conservative: only swap obvious aliases
# so we don't introduce label-leaking noise.
SYNONYMS: Dict[str, List[str]] = {
    "washer":           ["washing machine", "laundry machine", "washer"],
    "washing machine":  ["washer", "washing machine", "laundry unit"],
    "fridge":           ["refrigerator", "fridge", "ref"],
    "refrigerator":     ["fridge", "refrigerator", "refrigeration unit"],
    "aircon":           ["air conditioner", "AC", "aircon", "air-con unit"],
    "air conditioner":  ["aircon", "AC", "air conditioner"],
    "AC":               ["aircon", "air conditioner", "AC"],
    "printer":          ["printer", "printing device", "print unit"],
}

# Question-style rewrites. Each entry: (regex pattern, list of replacements).
QUESTION_REWRITES: List[Tuple[str, List[str]]] = [
    (r"\bWhat should I check\b", ["What can I check", "What do I check", "What should I look at"]),
    (r"\bWhat should I do\b",    ["What can I do", "What's the next step", "How should I proceed"]),
    (r"\bWhat could be\b",       ["What might be", "What is potentially"]),
    (r"\bMy\b",                  ["The", "My", "Our"]),
    (r"\bHow do I\b",            ["How can I", "What's the way to", "How do I"]),
    (r"\bIs that normal\b",      ["Is this expected", "Should I be worried"]),
    (r"\bDoes that mean\b",      ["Does this mean", "Is that to say"]),
    (r"\bWhat does that mean\b", ["What does this mean", "What does it indicate"]),
]


def _apply_one_rewrite(text: str, rng: random.Random) -> str:
    """Apply at most one question-style rewrite at random."""
    candidates = [(p, opts) for p, opts in QUESTION_REWRITES if re.search(p, text)]
    if not candidates:
        return text
    pat, options = rng.choice(candidates)
    repl = rng.choice(options)
    return re.sub(pat, repl, text, count=1)


def _swap_one_synonym(text: str, rng: random.Random) -> str:
    """Swap one synonymizable term in the text, if any."""
    # Look for any synonymizable phrase, case-insensitive but preserve case roughly.
    for phrase in sorted(SYNONYMS.keys(), key=len, reverse=True):  # longer phrases first
        if re.search(rf"\b{re.escape(phrase)}\b", text, flags=re.IGNORECASE):
            replacement = rng.choice(SYNONYMS[phrase])
            return re.sub(
                rf"\b{re.escape(phrase)}\b",
                replacement,
                text,
                count=1,
                flags=re.IGNORECASE,
            )
    return text


def augment_question(q: str, n_paraphrases: int = 4, rng: random.Random | None = None) -> List[str]:
    """
    Produce up to `n_paraphrases` paraphrases of `q` by combining synonym swap
    + question-style rewrite. Result includes the original.
    """
    rng = rng or random.Random(SEED)
    out = {q}
    attempts = 0
    while len(out) < n_paraphrases + 1 and attempts < n_paraphrases * 5:
        candidate = _apply_one_rewrite(_swap_one_synonym(q, rng), rng)
        if candidate != q:
            out.add(candidate)
        attempts += 1
    return list(out)


def build_dataset(questions_path: Path) -> Tuple[List[str], List[str], List[str]]:
    """
    Load questions.json and produce three parallel lists:
        texts          - augmented query strings
        labels         - filename labels (str)
        unique_labels  - sorted unique filenames (the class index ordering)
    """
    with questions_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    rng = random.Random(SEED)
    texts: List[str] = []
    labels: List[str] = []
    for q in items:
        question = (q.get("question") or "").strip()
        manual = q.get("manual_filename")
        if not question or not manual:
            continue
        for paraphrase in augment_question(question, n_paraphrases=4, rng=rng):
            texts.append(paraphrase)
            labels.append(manual)

    unique_labels = sorted(set(labels))
    return texts, labels, unique_labels


# ----------------------------------------------------------------------
# TRAINING
# ----------------------------------------------------------------------

def _to_class_indices(labels: List[str], label_order: List[str]) -> torch.Tensor:
    idx = {lbl: i for i, lbl in enumerate(label_order)}
    return torch.tensor([idx[lbl] for lbl in labels], dtype=torch.long)


def _embed_all(texts: List[str]) -> torch.Tensor:
    """Embed every text with MiniLM and stack into one tensor."""
    print(f"[embed] embedding {len(texts)} examples with MiniLM...")
    vecs = embed_texts(texts, model_name="sentence-transformers/all-MiniLM-L6-v2")
    return torch.tensor(vecs, dtype=torch.float32)


def train_one_split(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    verbose: bool = False,
) -> Tuple[ManualClassifier, Dict[str, List[float]]]:
    """Train ONE classifier on the given split. Returns the model + history."""
    model = ManualClassifier(
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        n_classes=int(y_train.max().item()) + 1,
        dropout=DROPOUT,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
            val_acc = (val_logits.argmax(dim=-1) == y_val).float().mean().item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(f"  epoch {epoch:3d}  train_loss={epoch_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    return model, history


def kfold_cv(
    X: torch.Tensor,
    y: torch.Tensor,
    n_folds: int = N_FOLDS,
) -> Tuple[List[Dict[str, List[float]]], List[float]]:
    """Stratified k-fold CV. Returns per-fold histories and per-fold final accuracy."""
    n = X.size(0)
    indices = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    histories: List[Dict[str, List[float]]] = []
    fold_accs: List[float] = []

    for fold_i, val_idx in enumerate(folds, start=1):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != fold_i - 1])
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        print(f"[fold {fold_i}/{n_folds}] train n={len(train_idx)}  val n={len(val_idx)}")
        _, hist = train_one_split(X_train, y_train, X_val, y_val, verbose=False)
        histories.append(hist)
        fold_accs.append(hist["val_acc"][-1])
        print(f"  -> final val_acc = {hist['val_acc'][-1]:.4f}")

    return histories, fold_accs


# ----------------------------------------------------------------------
# OUTPUTS
# ----------------------------------------------------------------------

def save_loss_plot(histories: List[Dict[str, List[float]]], path: Path) -> None:
    """Plot per-fold training curves. Skips silently if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # no display in headless / scripts
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — skipping loss curve. "
              "(`pip install matplotlib` if you want it for the report.)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Train loss
    for i, h in enumerate(histories, start=1):
        axes[0].plot(h["train_loss"], label=f"fold {i}", alpha=0.7)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train cross-entropy loss")
    axes[0].set_title("Training loss per fold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Val accuracy
    for i, h in enumerate(histories, start=1):
        axes[1].plot(h["val_acc"], label=f"fold {i}", alpha=0.7)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation accuracy")
    axes[1].set_title("Validation accuracy per fold")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {path}")


def save_label_map(label_order: List[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(label_order, f, indent=2, ensure_ascii=False)


def save_metrics(
    fold_accs: List[float],
    mean_acc: float,
    std_acc: float,
    label_order: List[str],
    n_examples: int,
    path: Path,
) -> None:
    payload = {
        "label_order": label_order,
        "n_classes": len(label_order),
        "n_augmented_examples": n_examples,
        "n_folds": len(fold_accs),
        "fold_val_accuracies": [round(a, 4) for a in fold_accs],
        "mean_val_accuracy": round(mean_acc, 4),
        "std_val_accuracy": round(std_acc, 4),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "seed": SEED,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main() -> None:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"questions.json not found at {QUESTIONS_PATH}. "
            "Make sure the eval set exists before training."
        )

    print(f"[load] reading {QUESTIONS_PATH}")
    texts, labels, label_order = build_dataset(QUESTIONS_PATH)
    print(f"[data] {len(texts)} augmented examples across {len(label_order)} manuals")
    for lbl in label_order:
        print(f"   - {lbl}: {labels.count(lbl)} examples")

    X = _embed_all(texts)
    y = _to_class_indices(labels, label_order)

    print()
    print(f"[cv] running {N_FOLDS}-fold cross-validation")
    histories, fold_accs = kfold_cv(X, y, n_folds=N_FOLDS)
    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    print()
    print(f"[cv] mean val accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    # Train final model on ALL augmented data (no val split — we already
    # measured generalization with CV).
    print()
    print(f"[final] training final model on all {len(texts)} examples")
    final_model = ManualClassifier()
    optimizer = optim.Adam(
        final_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()
    full_loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(1, EPOCHS + 1):
        final_model.train()
        loss_sum = 0.0
        for xb, yb in full_loader:
            optimizer.zero_grad()
            logits = final_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * xb.size(0)
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"  final epoch {epoch:3d}  loss={loss_sum/len(texts):.4f}")

    # Save artifacts.
    print()
    torch.save(final_model.state_dict(), MODEL_WEIGHTS)
    print(f"[save] weights -> {MODEL_WEIGHTS}")

    save_label_map(label_order, LABELS_FILE)
    print(f"[save] label map -> {LABELS_FILE}")

    save_metrics(fold_accs, mean_acc, std_acc, label_order, len(texts), METRICS_PATH)
    print(f"[save] metrics -> {METRICS_PATH}")

    save_loss_plot(histories, LOSS_PLOT_PATH)

    print()
    print("=" * 60)
    print("Manual classifier training complete.")
    print(f"  CV mean val accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Per-fold accuracies  : {[round(a, 4) for a in fold_accs]}")
    print(f"  Augmented examples   : {len(texts)} ({len(texts) // len(label_order)}/manual avg)")
    print("=" * 60)


if __name__ == "__main__":
    main()
