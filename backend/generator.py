"""
generator.py
------------
Pluggable generator backend for the manual RAG chatbot.

Backends:
1. TransformersGenerator
   - Current Phi-3-mini-4k-instruct via Hugging Face Transformers.

2. OllamaGenerator
   - Qwen via local Ollama HTTP API.
   - Default Ollama URL: http://localhost:11434/api/generate

Switch backend using environment variables:

    GENERATOR_BACKEND=transformers
    GENERATOR_BACKEND=ollama

Default:
    transformers
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from functools import lru_cache
from typing import List, Protocol

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from retriever import RetrievedChunk


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

GENERATOR_BACKEND = os.environ.get("GENERATOR_BACKEND", "transformers").lower().strip()

TRANSFORMERS_MODEL_NAME = os.environ.get(
    "GENERATOR_MODEL",
    "microsoft/Phi-3-mini-4k-instruct",
)

OLLAMA_URL = os.environ.get(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)

OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "qwen2.5:3b",
)

OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "600"))


# ---------------------------------------------------------------------
# SHARED PROMPTING
# ---------------------------------------------------------------------

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

    Including the source filename inline gives the model grounding context.
    The final API sources are still handled outside this file by rag_pipeline.py.
    """
    if not chunks:
        return "(no context retrieved)"

    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] {c.source}, p. {c.page}"
        blocks.append(f"{header}\n{c.text}")

    return "\n\n".join(blocks)


def _build_messages(question: str, chunks: List[RetrievedChunk]) -> list[dict[str, str]]:
    """
    Build one shared prompt structure for both backends.
    Transformers uses this as chat messages.
    Ollama converts this same structure into a plain prompt.
    """
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


def _messages_to_plain_prompt(messages: list[dict[str, str]]) -> str:
    """
    Convert the shared message structure into a plain prompt for Ollama /api/generate.
    """
    system_content = ""
    user_content = ""

    for message in messages:
        if message["role"] == "system":
            system_content = message["content"]
        elif message["role"] == "user":
            user_content = message["content"]

    return (
        f"System:\n{system_content}\n\n"
        f"User:\n{user_content}\n\n"
        "Assistant:"
    )


# ---------------------------------------------------------------------
# GENERATOR INTERFACE
# ---------------------------------------------------------------------

class GeneratorBackend(Protocol):
    def generate(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        ...


# ---------------------------------------------------------------------
# TRANSFORMERS BACKEND
# ---------------------------------------------------------------------

class TransformersGenerator:
    def __init__(self, model_name: str = TRANSFORMERS_MODEL_NAME):
        self.model_name = model_name
        self.tokenizer, self.model, self.device = self._load()

    def _device_and_dtype(self) -> tuple[str, torch.dtype]:
        if torch.cuda.is_available():
            print("Generator backend: transformers", flush=True)
            print("USING CUDA:", torch.cuda.get_device_name(0), flush=True)
            return "cuda", torch.float16

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("Generator backend: transformers", flush=True)
            print("USING MPS", flush=True)
            return "mps", torch.float16

        print("Generator backend: transformers", flush=True)
        print("USING CPU", flush=True)
        return "cpu", torch.float32

    def _load(self):
        """Load tokenizer + model once per process."""
        device, dtype = self._device_and_dtype()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=device if device != "cpu" else None,
            low_cpu_mem_usage=True,
        )

        if device == "cpu":
            model.to("cpu")

        model.eval()

        return tokenizer, model, device

    def generate(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        messages = _build_messages(question, chunks)

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0, input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------
# OLLAMA BACKEND
# ---------------------------------------------------------------------

class OllamaGenerator:
    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        url: str = OLLAMA_URL,
        timeout: int = OLLAMA_TIMEOUT,
    ):
        self.model = model
        self.url = url
        self.timeout = timeout

        print("Generator backend: ollama", flush=True)
        print(f"Ollama model: {self.model}", flush=True)
        print(f"Ollama URL: {self.url}", flush=True)

    def generate(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        messages = _build_messages(question, chunks)
        prompt = _messages_to_plain_prompt(messages)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens,
                "top_p": 0.9,
            },
        }

        body = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("response", "").strip()

        except urllib.error.URLError:
            return (
                "I could not connect to the local Ollama server. "
                "Please make sure Ollama is running and the selected model is installed."
            )

        except Exception as e:
            return f"I could not generate an answer using Ollama: {e}"


# ---------------------------------------------------------------------
# BACKEND FACTORY
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_generator() -> GeneratorBackend:
    if GENERATOR_BACKEND == "transformers":
        return TransformersGenerator()

    if GENERATOR_BACKEND == "ollama":
        return OllamaGenerator()

    raise ValueError(
        f"Invalid GENERATOR_BACKEND='{GENERATOR_BACKEND}'. "
        "Use GENERATOR_BACKEND=transformers or GENERATOR_BACKEND=ollama."
    )


# ---------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------

def generate(
    question: str,
    chunks: List[RetrievedChunk],
    max_new_tokens: int = 100,
    temperature: float = 0.0,
) -> str:
    """
    Generate an answer grounded in the retrieved chunks.

    This function is intentionally unchanged as the public interface, so
    rag_pipeline.py can continue calling generate(query, chunks).
    """
    generator = _get_generator()

    return generator.generate(
        question=question,
        chunks=chunks,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )