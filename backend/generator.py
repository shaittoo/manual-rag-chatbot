"""
generator.py
------------
Pluggable generator backend for the manual RAG chatbot.

Backends:
1. TransformersGenerator
   - Phi-3-mini-4k-instruct via Hugging Face Transformers.

2. OllamaGenerator
   - Qwen via local Ollama HTTP API.
   - Default Ollama URL: http://localhost:11434/api/generate

Supports:
- frontend dropdown model switching
- conversational / follow-up questions through chat history
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from functools import lru_cache
from typing import List, Optional, Protocol

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from retriever import RetrievedChunk


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DEFAULT_GENERATOR_BACKEND = (
    os.environ.get("GENERATOR_BACKEND", "transformers").lower().strip()
)

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
    "Answer the user's current question using ONLY the provided manual context. "
    "Use the conversation history only to understand what the user is referring to. "
    "If the context does not contain the answer, say you don't know — do not invent steps, "
    "part numbers, or model details. "
    "\n\n"
    "IMPORTANT — relevance discipline: "
    "Answer ONLY the specific current question asked. Be precise about the user's actual problem. "
    "Even if the retrieved context contains additional troubleshooting steps, settings, or features "
    "that are mentioned alongside the relevant material, do NOT include them in your answer "
    "unless they directly address the user's current question. "
    "Prefer a shorter focused answer over a longer comprehensive one. "
    "\n\n"
    "Do NOT include filenames, page numbers, or parenthetical citations in your answer — "
    "the system appends sources automatically. Just write a clean, direct answer."
)


def _format_context(chunks: List[RetrievedChunk]) -> str:
    """
    Render retrieved chunks as a numbered context block.
    """
    if not chunks:
        return "(no context retrieved)"

    blocks = []

    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] {c.source}, p. {c.page}"
        blocks.append(f"{header}\n{c.text}")

    return "\n\n".join(blocks)


def _format_history(history: Optional[List[dict]]) -> str:
    """
    Format recent chat history for follow-up questions.

    Expected shape:
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    """
    if not history:
        return "(no previous conversation)"

    lines = []

    for msg in history[-6:]:
        role = msg.get("role", "")
        content = " ".join((msg.get("content") or "").split())

        if not content:
            continue

        if len(content) > 700:
            content = content[:700] + "..."

        if role not in {"user", "assistant"}:
            role = "user"

        lines.append(f"{role}: {content}")

    return "\n".join(lines) if lines else "(no previous conversation)"


def _build_messages(
    question: str,
    chunks: List[RetrievedChunk],
    history: Optional[List[dict]] = None,
) -> list[dict[str, str]]:
    """
    Build one shared prompt structure for both backends.

    Transformers uses this as chat messages.
    Ollama converts this same structure into a plain prompt.
    """
    context = _format_context(chunks)
    conversation = _format_history(history)

    user_content = (
        f"Conversation so far:\n{conversation}\n\n"
        f"Context from manuals:\n{context}\n\n"
        f"Current question: {question}\n\n"
        "Answer the current question using the manual context. "
        "Use the conversation history only to understand references like "
        "'that', 'it', 'after that', or 'what if it still happens'. "
        "Do not repeat previous answers unless needed. "
        "If the manual context does not contain the answer, say you don't know. "
        "Do not include filenames, page numbers, or parenthetical citations — those are "
        "added by the system."
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
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        history: Optional[List[dict]] = None,
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
        """
        Load tokenizer + model once per process.
        """
        device, dtype = self._device_and_dtype()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        # On CUDA, let Transformers/Accelerate decide placement.
        # This helps on smaller GPUs by offloading some weights if needed.
        if device == "cuda":
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        if device in {"cpu", "mps"}:
            model.to(device)

        model.eval()

        return tokenizer, model, device

    def generate(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        history: Optional[List[dict]] = None,
    ) -> str:
        messages = _build_messages(
            question=question,
            chunks=chunks,
            history=history,
        )

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Fixes warning:
        # "The attention mask is not set and cannot be inferred..."
        attention_mask = torch.ones_like(input_ids).to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "attention_mask": attention_mask,
        }

        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = 0.9

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                **generation_kwargs,
            )

        new_tokens = output_ids[0, input_ids.shape[-1]:]

        return self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()


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
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        history: Optional[List[dict]] = None,
    ) -> str:
        messages = _build_messages(
            question=question,
            chunks=chunks,
            history=history,
        )

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

@lru_cache(maxsize=4)
def get_generator(
    generator_backend: str = DEFAULT_GENERATOR_BACKEND,
) -> GeneratorBackend:
    """
    Return the requested generator backend.

    This is cached per backend, so switching from the frontend does not reload
    the same model repeatedly.

    Example cache keys:
        get_generator("transformers")
        get_generator("ollama")
    """
    backend = (generator_backend or DEFAULT_GENERATOR_BACKEND).lower().strip()

    if backend == "transformers":
        return TransformersGenerator()

    if backend == "ollama":
        return OllamaGenerator()

    raise ValueError(
        f"Invalid generator_backend='{generator_backend}'. "
        "Use 'transformers' or 'ollama'."
    )


# ---------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------

def generate(
    question: str,
    chunks: List[RetrievedChunk],
    generator_backend: str = DEFAULT_GENERATOR_BACKEND,
    max_new_tokens: int = 384,
    temperature: float = 0.0,
    history: Optional[List[dict]] = None,
) -> str:
    """
    Generate an answer grounded in the retrieved chunks.

    The frontend can switch models per request by sending:
        generator_backend="transformers"
        generator_backend="ollama"

    The frontend can also pass recent chat turns through `history`
    for conversational follow-up questions.
    """
    generator = get_generator(generator_backend)

    return generator.generate(
        question=question,
        chunks=chunks,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        history=history,
    )