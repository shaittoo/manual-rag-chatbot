"""
generator.py
------------
Wraps Phi-3-mini-4k-instruct (Hugging Face Transformers) behind a single
generate(question, context_chunks) function.

Why Phi-3-mini and not TinyLlama?
- ~3.8B parameters vs 1.1B; far better at instruction-following on RAG-style tasks.
- 4k context window is enough for ~10-15 retrieved chunks.
- Works on CPU (slow but functional, ~1-3s/token) and CUDA (fast).

Honesty note for the project writeup:
- We are NOT fine-tuning Phi-3 here. This is *retrieval-augmented* generation:
  the "deep learning" part is the embedder + the pretrained LLM. If your assignment
  requires actual training, swap embedder.py for a small model you fine-tune yourself,
  or fine-tune the generator with LoRA on a Q/A dataset built from the manuals.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from retriever import RetrievedChunk

MODEL_NAME = os.environ.get("GENERATOR_MODEL", "microsoft/Phi-3-mini-4k-instruct")


# --- Model loading -------------------------------------------------------

def _device_and_dtype():
    if torch.cuda.is_available():
        print("USING CUDA:", torch.cuda.get_device_name(0), flush=True)
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("USING MPS", flush=True)
        return "mps", torch.float16
    print("USING CPU", flush=True)
    return "cpu", torch.float32


@lru_cache(maxsize=1)
def _load():
    """Load tokenizer + model once per process. ~7-8GB RAM at fp32, ~4GB at fp16."""
    device, dtype = _device_and_dtype()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device if device != "cpu" else None,
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        model.to("cpu")
    model.eval()
    return tokenizer, model, device


# --- Prompting -----------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant for product manuals. "
    "Answer the user's question using ONLY the provided context. "
    "If the context does not contain the answer, say you don't know — do not invent steps, "
    "part numbers, or model details. "
    "Do NOT include filenames, page numbers, or parenthetical citations in your answer — "
    "the system appends sources automatically. Just write a clean, direct answer."
)


def _format_context(chunks: List[RetrievedChunk]) -> str:
    """
    Render retrieved chunks as a numbered context block.

    Including the source filename inline gives the model something to cite from
    rather than hallucinating sources. This is much more reliable than asking
    the model to "remember" sources via a separate field.
    """
    if not chunks:
        return "(no context retrieved)"
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] {c.source}, p. {c.page}"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


def _build_messages(question: str, chunks: List[RetrievedChunk]):
    """Phi-3 expects chat-formatted messages. apply_chat_template handles the special tokens."""
    context = _format_context(chunks)
    user_content = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely in plain prose. Do not include filenames, page numbers, "
        "or parenthetical citations — those are added by the system."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# --- Public API ----------------------------------------------------------

def generate(
    question: str,
    chunks: List[RetrievedChunk],
    # max_new_tokens: int = 350,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
) -> str:
    """
    Generate an answer grounded in the retrieved chunks.

    temperature=0.0 (greedy decoding) is intentional: for factual lookup we want
    the most likely token at every step, which empirically reduces fabrication
    (e.g. invented page numbers, made-up part numbers). Raise to 0.5+ only for
    brainstorming / explanation tasks.
    """
    tokenizer, model, device = _load()
    messages = _build_messages(question, chunks)

    # apply_chat_template returns a tensor of input_ids when tokenize=True.
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Strip the prompt tokens; keep only what the model produced.
    new_tokens = output_ids[0, input_ids.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text
