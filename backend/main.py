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

from typing import Optional

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

# CORS: open in dev so the Next.js frontend on a different port can call us.
# Tighten this to specific origins before any kind of deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas -------------------------------------------------------------

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


class ClassifyResponse(BaseModel):
    predicted_source: str
    confidence: float
    distribution: list[dict]


class AskAutoResponse(BaseModel):
    answer: str
    sources: list[dict]
    routed_to: str
    routing_confidence: float


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
    result = ask_dict(req.query, top_k=req.top_k, source=req.source)
    return AskResponse(**result)


# --- Manual classifier (auto-routing) ------------------------------------
#
# These two endpoints are gated by the trained weights existing on disk.
# If `manual_classifier.pt` isn't present, both endpoints return 503 with a
# helpful message rather than crashing — useful for partial deployments where
# the classifier hasn't been trained yet (or has been deliberately disabled).

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
    Useful for the frontend to display 'Routing to X (NN%)' before the answer.
    """
    predict_full = _load_classifier_or_fail()
    result = predict_full(req.query)
    return ClassifyResponse(**result)


@app.post("/ask_auto", response_model=AskAutoResponse)
def ask_auto_endpoint(req: ClassifyRequest) -> AskAutoResponse:
    """
    Classify -> ask. The user doesn't have to pick a manual; we predict it.
    Returns the same shape as /ask plus the predicted source and its confidence.
    """
    predict_full = _load_classifier_or_fail()
    routing = predict_full(req.query)
    result = ask_dict(req.query, source=routing["predicted_source"])
    return AskAutoResponse(
        answer=result["answer"],
        sources=result["sources"],
        routed_to=routing["predicted_source"],
        routing_confidence=routing["confidence"],
    )
