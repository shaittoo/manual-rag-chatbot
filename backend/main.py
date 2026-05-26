"""
main.py
-------
FastAPI entrypoint. Endpoints:

    GET  /health             -> liveness check
    GET  /sources            -> list filenames currently in the index
    POST /ingest             -> rebuild the Chroma index from backend/manuals/*.pdf
    POST /ask {query,...}    -> RAG: retrieve + generate (optionally scoped to a source)
    POST /classify {query}   -> predict which manual a query is about (no generation)
    POST /ask_auto {query}   -> classify -> ask: auto-routes the query to the predicted manual

Run locally:
    cd manu/backend
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_pipeline import ask_dict
from retriever import ingest, list_sources


app = FastAPI(
    title="Manu — Manual RAG Chatbot",
    description="Ask natural-language questions against your product manuals.",
    version="0.1.0",
)

# CORS: open in dev so the Vite frontend on a different port can call us.
# Tighten this to specific origins before any kind of deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas -------------------------------------------------------------

GeneratorBackend = Literal["transformers", "ollama"]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")
    top_k: int = Field(4, ge=1, le=20, description="How many chunks to retrieve.")
    source: Optional[str] = Field(
        None,
        description=(
            "Optional filename to restrict retrieval to one manual "
            "(e.g. 'db05a9.pdf'). Use GET /sources to see available filenames."
        ),
    )
    generator_backend: GeneratorBackend = Field(
        "transformers",
        description=(
            "Which generator backend to use for this request: "
            "'transformers' for Phi-3-mini or 'ollama' for Qwen via Ollama."
        ),
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Recent conversation history for follow-up questions.",
    )


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


class IngestResponse(BaseModel):
    files: int
    chunks: int


class SourcesResponse(BaseModel):
    sources: list[str]


class ClassifyRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Recent conversation history for follow-up routing.",
    )


class ClassifyResponse(BaseModel):
    predicted_source: str
    confidence: float
    distribution: list[dict]


class AskAutoRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")
    top_k: int = Field(4, ge=1, le=20, description="How many chunks to retrieve.")
    generator_backend: GeneratorBackend = Field(
        "transformers",
        description=(
            "Which generator backend to use for this request: "
            "'transformers' for Phi-3-mini or 'ollama' for Qwen via Ollama."
        ),
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Recent conversation history for follow-up questions.",
    )


class AskAutoResponse(BaseModel):
    answer: str
    sources: list[dict]
    routed_to: str
    routing_confidence: float


# --- Helpers -------------------------------------------------------------

def _history_as_dicts(history: list[ChatMessage]) -> list[dict]:
    """
    Convert Pydantic chat messages into plain dicts for rag_pipeline.py.
    """
    return [
        {
            "role": msg.role,
            "content": msg.content.strip(),
        }
        for msg in history
        if msg.content and msg.content.strip()
    ]


def _conversation_aware_query(query: str, history: list[ChatMessage]) -> str:
    """
    Build a richer query for classifier routing.

    This helps when the user asks follow-up questions like:
        "What if it still does not work?"
        "Can I do that again?"
        "What about after that?"

    The classifier gets the recent conversation plus the current question,
    so it can still route to the correct manual.
    """
    query = query.strip()

    if not history:
        return query

    lines = []

    for msg in history[-6:]:
        content = " ".join(msg.content.split())

        if not content:
            continue

        if len(content) > 500:
            content = content[:500] + "..."

        lines.append(f"{msg.role}: {content}")

    if not lines:
        return query

    return (
        "Conversation so far:\n"
        + "\n".join(lines)
        + "\n\nCurrent question:\n"
        + query
    )


# --- Routes --------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint() -> IngestResponse:
    """
    Wipe the Chroma collection and rebuild from manuals/.
    Synchronous on purpose — for a mini-project the simplicity beats async complexity.
    For larger corpora you'd want a background task + progress endpoint.
    """
    try:
        report = ingest()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # e.g. "no text extracted" — almost always means scanned/image PDFs
        raise HTTPException(status_code=422, detail=str(e))

    return IngestResponse(**report)


@app.get("/sources", response_model=SourcesResponse)
def sources_endpoint() -> SourcesResponse:
    """Return all unique filenames currently indexed in Chroma."""
    return SourcesResponse(sources=list_sources())


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    result = ask_dict(
        req.query,
        top_k=req.top_k,
        source=req.source,
        generator_backend=req.generator_backend,
        history=_history_as_dicts(req.history),
    )

    return AskResponse(**result)


# --- Manual classifier (auto-routing) ------------------------------------

def _load_classifier_or_fail():
    """Lazy import + clean error if weights are missing."""
    try:
        from manual_classifier import predict_full  # noqa: WPS433

        return predict_full

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Manual classifier not available: {e}. "
                "Run `python train_classifier.py` first to produce the weights."
            ),
        )


@app.post("/classify", response_model=ClassifyResponse)
def classify_endpoint(req: ClassifyRequest) -> ClassifyResponse:
    """
    Predict which manual a query is about, without running RAG.

    For follow-up questions, we include recent conversation history in the
    classifier query so vague questions can still be routed correctly.
    """
    predict_full = _load_classifier_or_fail()

    classifier_query = _conversation_aware_query(req.query, req.history)
    result = predict_full(classifier_query)

    return ClassifyResponse(**result)


@app.post("/ask_auto", response_model=AskAutoResponse)
def ask_auto_endpoint(req: AskAutoRequest) -> AskAutoResponse:
    """
    Classify -> ask. The user doesn't have to pick a manual; we predict it.

    This also supports:
    - generator dropdown via generator_backend
    - follow-up questions via history
    """
    predict_full = _load_classifier_or_fail()

    classifier_query = _conversation_aware_query(req.query, req.history)
    routing = predict_full(classifier_query)

    result = ask_dict(
        req.query,
        top_k=req.top_k,
        source=routing["predicted_source"],
        generator_backend=req.generator_backend,
        history=_history_as_dicts(req.history),
    )

    return AskAutoResponse(
        answer=result["answer"],
        sources=result["sources"],
        routed_to=routing["predicted_source"],
        routing_confidence=routing["confidence"],
    )