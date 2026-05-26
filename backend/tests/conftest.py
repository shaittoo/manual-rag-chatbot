"""
conftest.py
-----------
Shared pytest fixtures and import-time stubs.

Why stubs?
----------
The backend modules import heavy ML libraries at module load time:
    embedder.py   -> torch, sentence_transformers
    generator.py  -> torch, transformers
    retriever.py  -> chromadb, pypdf

Those libraries are multi-gigabyte and need a GPU/model download to be useful.
The pure-logic functions we unit-test here (chunking, citation stripping,
prompt/history formatting, request validation) never actually *call* into those
libraries — they only need the `import` statements to succeed.

So: if a real library is installed (e.g. on a dev machine), we use it. If it is
NOT installed (e.g. CI or a lightweight sandbox), we register a minimal stub in
`sys.modules` so the import succeeds and the pure logic can be exercised.

This means the SAME test files run unmodified both with and without the heavy
dependencies present.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# Make the backend package importable (tests live in backend/tests/).
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _ensure_module(name: str, factory) -> None:
    """
    Register a stub module under `name` only if the real one cannot be imported.
    `factory` receives the freshly created module and populates its attributes.
    """
    try:
        importlib.import_module(name)
        return  # real library present — leave it alone
    except Exception:
        pass

    module = types.ModuleType(name)
    factory(module)
    sys.modules[name] = module


def _stub_torch(mod: types.ModuleType) -> None:
    cuda = types.SimpleNamespace(
        is_available=lambda: False,
        get_device_name=lambda i=0: "stub-gpu",
    )
    backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    mod.cuda = cuda
    mod.backends = backends
    mod.float16 = "float16"
    mod.float32 = "float32"
    mod.dtype = object
    mod.no_grad = MagicMock()
    mod.ones_like = MagicMock()
    mod.__version__ = "0.0.0-stub"


def _stub_sentence_transformers(mod: types.ModuleType) -> None:
    mod.SentenceTransformer = MagicMock(name="SentenceTransformer")
    mod.CrossEncoder = MagicMock(name="CrossEncoder")


def _stub_transformers(mod: types.ModuleType) -> None:
    mod.AutoModelForCausalLM = MagicMock(name="AutoModelForCausalLM")
    mod.AutoTokenizer = MagicMock(name="AutoTokenizer")


def _stub_chromadb(mod: types.ModuleType) -> None:
    mod.PersistentClient = MagicMock(name="PersistentClient")
    # retriever.py annotates with chromadb.api.ClientAPI, but `from __future__
    # import annotations` keeps annotations as strings, so api just needs to exist.
    api = types.ModuleType("chromadb.api")
    api.ClientAPI = object
    mod.api = api
    sys.modules["chromadb.api"] = api

    config = types.ModuleType("chromadb.config")
    config.Settings = MagicMock(name="Settings")
    mod.config = config
    sys.modules["chromadb.config"] = config


def _stub_pypdf(mod: types.ModuleType) -> None:
    mod.PdfReader = MagicMock(name="PdfReader")


# Install stubs (no-ops when the real libraries are available).
_ensure_module("torch", _stub_torch)
_ensure_module("sentence_transformers", _stub_sentence_transformers)
_ensure_module("transformers", _stub_transformers)
_ensure_module("chromadb", _stub_chromadb)
_ensure_module("pypdf", _stub_pypdf)
