# Manu — Manual RAG Chatbot

A retrieval-augmented chatbot for product manuals. Drop PDFs into `backend/manuals/`, ingest them into a local vector store, and ask natural-language questions. Backend is FastAPI + ChromaDB + sentence-transformers + Phi-3-mini. Frontend (Next.js) comes later.

## What "RAG" means here

```
PDFs -> chunks -> embeddings (MiniLM)        ┐
                                              ├──> ChromaDB (persistent, on disk)
question -> embedding (same MiniLM)           ┘
                                              │
                                              ▼
                                      top-k chunks
                                              │
                                              ▼
                            Phi-3-mini (system + context + question)
                                              │
                                              ▼
                                       grounded answer + sources
```

## Honest scope note

This project does **not** train a model. The "deep learning" is the pretrained MiniLM embedder and the pretrained Phi-3-mini generator. If your assignment requires training, options:
- Fine-tune Phi-3-mini with LoRA on Q&A pairs synthesized from your manuals.
- Train a small embedder (e.g. distilbert) on (query, passage) pairs and compare retrieval quality against MiniLM.
- Train a re-ranker (cross-encoder) on top of the retriever and ablate its contribution to answer quality.

## Setup (Windows / PowerShell)

```powershell
cd manu\backend

python -m venv .venv
.venv\Scripts\Activate.ps1

# Choose one torch install:
# CPU only:
pip install torch
# OR CUDA 12.1:
# pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

First run will download model weights (~150MB MiniLM + ~7GB Phi-3-mini) into the Hugging Face cache.

## Run

```powershell
# 1. Drop your PDFs into backend/manuals/
# 2. Start the API
uvicorn main:app --reload --port 8000
```

## API endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Liveness check. Returns `{"status": "ok"}`. |
| `GET` | `/sources` | List all filenames currently in the index. Used by the frontend to populate the "which manual?" dropdown. |
| `POST` | `/ingest` | Wipe the Chroma collection and rebuild it from every PDF in `backend/manuals/`. Returns `{files, chunks}`. Synchronous — can take minutes. |
| `POST` | `/ask` | RAG: retrieve + generate. Body: `{"query": str, "top_k": int = 4, "source": str | null}`. Returns `{answer, sources[]}`. |

The easiest way to drive the API is the auto-generated Swagger UI: http://localhost:8000/docs

## Quick test (PowerShell)

```powershell
# Liveness
Invoke-RestMethod http://localhost:8000/health

# Build the index from manuals/
Invoke-RestMethod -Method Post http://localhost:8000/ingest

# See what's indexed
Invoke-RestMethod http://localhost:8000/sources

# Ask a question, scoped to one manual
Invoke-RestMethod -Method Post http://localhost:8000/ask `
  -ContentType "application/json" `
  -Body '{"query": "How do I drain antifreeze?", "source": "db05a9.pdf"}'
```



## AI-assisted evaluation scoring with Ollama + Qwen

The main chatbot answers user questions through the RAG pipeline using Phi-3-mini. For evaluation, this project can also use a **free local AI judge** through **Ollama + Qwen** to score the chatbot's generated answers after running `run_eval.py`.

This scoring step is optional. It does **not** replace the `/ask` chatbot model. It only compares each generated answer with its expected reference answer.

The scoring flow is:

```text
eval/results.csv
        ↓
score_results.py
        ↓
Ollama local API + Qwen judge model
        ↓
eval/results_scored.csv
```

### What Qwen scores

For each row in `eval/results.csv`, the local Qwen judge receives:

```text
question
expected/reference answer
actual chatbot answer
error column
```

It assigns one of four labels:

| Label | Meaning |
|---|---|
| `Correct` | The answer contains the main required facts or steps from the reference answer. |
| `Partial` | The answer is relevant but incomplete or missing important details. |
| `Wrong` | The answer is unrelated, misleading, unsupported, contradictory, or answers a different issue. |
| `Refused` | The answer is empty, timed out, says it does not know, or gives no useful answer. |

Treat this as **AI-assisted scoring**, not fully automatic scoring. Review the suggested labels manually before reporting final accuracy.

### 1. Install Ollama on Windows

Open PowerShell and run:

```powershell
irm https://ollama.com/install.ps1 | iex
```

After installation, close and reopen PowerShell. Then check that Ollama is installed:

```powershell
ollama --version
```

If this prints an Ollama version number, installation worked.

If the PowerShell installer does not work, install Ollama manually from the official Windows installer:

```text
https://ollama.com/download/windows
```

### 2. Download the Qwen judge model

Pull the recommended local judge model:

```powershell
ollama pull qwen2.5:3b
```

This model is recommended because it is still relatively lightweight while being more reliable than very small models for judging answer quality.


If using the smaller model, update `score_results.py`:

```python
MODEL = "qwen2.5:1.5b"
```

For the recommended model, keep:

```python
MODEL = "qwen2.5:3b"
```

### 3. Test Qwen manually

Run:

```powershell
ollama run qwen2.5:3b
```

Type a simple test prompt:

```text
Classify this as Correct, Partial, Wrong, or Refused: The answer is empty.
```

To exit the Ollama chat, type:

```text
/bye
```

### 4. Check that the Ollama local API is running

Ollama normally runs a local API server at:

```text
http://localhost:11434
```

Check installed models:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

You should see the downloaded Qwen model in the response, such as:

```text
qwen2.5:3b
qwen2.5:1.5b
```

If this command fails, make sure the Ollama app/service is running, or run:

```powershell
ollama list
```

### 5. Run the chatbot evaluation first

Before scoring, generate the chatbot evaluation results:

```powershell
cd backend
python run_eval.py
```

This creates or appends to:

```text
eval/results.csv
```

For long evals, run `run_eval.py` in batches and avoid starting Uvicorn with `--reload`.

### 6. Run AI-assisted scoring

After `eval/results.csv` exists, run:

```powershell
python score_results.py
```

The scoring script reads:

```text
eval/results.csv
```

and writes:

```text
eval/results_scored.csv
```

The scored output contains the original evaluation columns plus:

```text
ai_score
ai_reason
```

Example output:

```text
ai_score: Partial
ai_reason: The answer gives relevant checks but misses the required temperature-difference threshold from the reference answer.
```

### 7. Calculate accuracy

Open `eval/results_scored.csv` in Excel or Google Sheets.

Strict accuracy counts only `Correct` answers:

```excel
=COUNTIF(K2:K26,"Correct")/COUNTA(K2:K26)
```

Lenient accuracy gives half credit to `Partial` answers:

```excel
=(COUNTIF(K2:K26,"Correct") + 0.5*COUNTIF(K2:K26,"Partial"))/COUNTA(K2:K26)
```

Adjust the column letter if `ai_score` is not in column `K`.

### Notes

- Ollama is a separate local application, so it should **not** be added to `requirements.txt`.
- The scoring script uses the local Ollama API, usually at `http://localhost:11434`.
- Empty answers, timeouts, and connection failures should normally be scored as `Refused`.
- Answers that retrieve the wrong manual section and answer a different issue should normally be scored as `Wrong`.
- Answers that contain some correct steps but miss important required details should normally be scored as `Partial`.

## Project layout

```
manu/
├── backend/
│   ├── main.py              # FastAPI app + routes
│   ├── rag_pipeline.py      # Orchestrates retrieve + generate
│   ├── embedder.py          # sentence-transformers wrapper (MiniLM)
│   ├── retriever.py         # PDF parsing, chunking, ChromaDB I/O
│   ├── generator.py         # Phi-3-mini-4k-instruct via Transformers
│   ├── requirements.txt
│   ├── manuals/             # ⬅ put PDFs here (gitignored)
│   └── chroma_db/           # auto-created, gitignored
└── frontend/                # (TBD) Next.js + ChatWidget
```

## How the code is organized (read this first if you're new to the project)

Each backend file does **one thing** and exposes a small public API. Read them in this order — top to bottom mirrors the request flow.

**`embedder.py`** — Wraps `sentence-transformers` (MiniLM). Two functions: `embed_texts(list[str])` for ingestion, `embed_query(str)` for search. Cached in memory so the model only loads once per process. Embeddings are L2-normalized so cosine similarity reduces to a dot product.

**`retriever.py`** — Owns everything related to the vector store.
- `ingest()` walks `manuals/`, parses each PDF with `pypdf`, splits each page into 800-character overlapping chunks, embeds them, and writes to a persistent ChromaDB at `backend/chroma_db/`.
- `search(query, top_k, source=None)` embeds the query, asks Chroma for the nearest chunks, optionally filtered to one filename. Returns `RetrievedChunk(text, source, page, score)`.
- `list_sources()` returns the unique filenames currently indexed.

**`generator.py`** — Loads `microsoft/Phi-3-mini-4k-instruct` via Hugging Face `transformers`. The system prompt forbids inline citations (so the model can't hallucinate page numbers) and constrains it to the retrieved context. `generate(question, chunks)` runs greedy decoding (temperature=0) and returns plain text.

**`rag_pipeline.py`** — The glue. `ask(query, top_k, source)` runs `search → generate → strip-any-leftover-citations → return Answer(answer, sources)`. This is the function to call from a notebook for evals; FastAPI is just a thin wrapper around it.

**`main.py`** — FastAPI: defines request/response schemas, mounts CORS, exposes the four endpoints above. No business logic lives here.

**Mental model:** request comes in → `main.py` validates schema → calls `rag_pipeline.ask` → which calls `retriever.search` (uses `embedder`) and then `generator.generate`. To trace a bug, follow that chain.

## Working as a pair

The project splits cleanly into two ownerships. Pick whichever fits your strengths and own it end-to-end:

**Backend / model owner**
- All five `.py` files in `backend/`
- Embeddings, retrieval quality, prompt engineering, evals
- Good extension projects (any of these is a meaty contribution to the writeup):
  - Add a cross-encoder re-ranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) between retrieval and generation.
  - Compare embedders: `all-MiniLM-L6-v2` vs `bge-small-en-v1.5` vs `e5-small-v2` on the same questions.
  - Switch `generator.py` to `llama-cpp-python` + GGUF for 5× CPU speedup; benchmark answer quality.
  - Build an eval set: 5 questions per manual with verified answers, score correct/partial/wrong.
  - LoRA fine-tune Phi-3 on synthesized Q&A from the manuals.

**Frontend / UX owner**
- `frontend/` (Next.js, not built yet — that's the next milestone)
- Wire `pages/index.js` to call `POST /ask` and `GET /sources`
- `components/ChatWidget.jsx` — chat history, input box, "which manual?" dropdown, sources panel showing snippet + page
- Good extensions:
  - Streaming responses (FastAPI `StreamingResponse` + frontend SSE/fetch streams) — Phi-3 feels much faster when tokens stream in.
  - "Show raw chunks" toggle so the user can verify any answer against what the LLM actually saw.
  - Conversation history (multi-turn). Will require rewriting the prompt in `generator.py` to include history.

**Shared work**
- The eval set (worth doing together — agree on what "correct" means).
- The final writeup / report.

## Workflow for two people

- **Branches over `main`.** Each person works on a feature branch (`backend/reranker`, `frontend/chat-widget`), opens a PR, the other reviews. Even with two people this catches bugs and keeps the history readable.
- **Don't both touch the same file at once.** The split above mostly prevents this; if it happens, communicate.
- **Commit messages should explain *why*, not *what*.** "Add reranker" is bad; "Add cross-encoder reranker — top-k retrieval was returning near-duplicates that crowded out diverse evidence" is good.
- **Run `/ingest` after pulling.** The Chroma DB is gitignored (it's regenerable), so a fresh clone needs a re-ingest before `/ask` will work.

## Knobs worth knowing

| File | Setting | Default | What it does |
|---|---|---|---|
| `embedder.py` | `DEFAULT_MODEL` | `all-MiniLM-L6-v2` | Embedding model. Swap for `bge-small-en-v1.5` if recall is weak. |
| `retriever.py` | `chunk_size`, `overlap` | 800, 120 | Char-based windows. Smaller = more precise retrieval, more chunks. |
| `retriever.py` | `top_k` (in `search`) | 4 | How many chunks fed to the LLM. More context ≠ always better. |
| `generator.py` | `temperature` | 0.0 | 0 = greedy decoding (deterministic, fewer fabrications). Raise for brainstorming. |
| `generator.py` | `MODEL_NAME` env var | Phi-3-mini-4k-instruct | Override via `GENERATOR_MODEL=...`. |

## Known limits (be ready to discuss in your writeup)

1. **Scanned PDFs won't work.** `pypdf` extracts text only. Add OCR (e.g. `ocrmypdf` or `pytesseract`) if your manuals are image-based.
2. **No re-ranking.** Top-k from cosine similarity can include near-duplicates or off-topic chunks. A cross-encoder re-ranker would help — and is a great extension for the project.
3. **No conversational memory.** Each `/ask` is independent. Adding history would mean handling follow-up questions like "and what about model X?".
4. **Phi-3-mini on CPU is slow.** Expect 5–30s per answer depending on hardware. Use CUDA if available, or swap to a GGUF + `llama-cpp-python` for ~5x CPU speedup.
