# Project Tasks — Manu (Manual RAG Chatbot)

This document tracks every task from project start to final submission. It exists so both partners can see the whole picture — what's done, what's running, what's next, and **why** each step matters.

**Status legend:**
- `[x]` Done
- `[~]` In progress
- `[ ]` Not started

**Last updated:** 2026-05-06 (V1 RAGAS complete; V2 code pushed; V2 eval handed to Duranne)

---

## Phase 0 — Project setup

- [x] **0.1 Initial planning and architecture decision**
  Decide on the system: FastAPI backend + RAG pipeline + Vite/React frontend. Pick Phi-3-mini over TinyLlama for the generator (3.8B is meaningfully better at instruction-following than 1.1B for RAG). Pick MiniLM as the embedder (small, fast, strong on English). Pick ChromaDB for vector storage (local, persistent, simple). Decision rationale lives in `README.md`.

- [x] **0.2 Repository structure**
  Create `manu/` with `backend/` and `frontend/` subdirectories. Add `.gitignore` covering venv, chroma_db, manuals/*.pdf, hf cache, node_modules, .env files. Add `manuals/.gitkeep` so the folder exists in git despite PDFs being ignored.

---

## Phase 1 — Backend foundation (Owner: Shaina)

- [x] **1.1 `requirements.txt`**
  Pin versions of fastapi, uvicorn, sentence-transformers, chromadb, pypdf, transformers, accelerate. Leave torch unpinned so users pick the right CPU/CUDA build for their machine.

- [x] **1.2 `embedder.py`**
  Wrap `sentence-transformers/all-MiniLM-L6-v2` with `embed_texts()` and `embed_query()`. Cache the model in memory with `lru_cache` so it loads once per process. L2-normalize embeddings so cosine similarity reduces to a dot product.

- [x] **1.3 `retriever.py`**
  Three jobs: parse PDFs with pypdf, split into 800-character overlapping chunks (120-char overlap), embed and store in a persistent ChromaDB at `backend/chroma_db/`. Functions: `ingest()` (rebuild index), `search(query, top_k, source)` (retrieve with optional source filter), `list_sources()` (enumerate filenames).

- [x] **1.4 `generator.py`**
  Load Phi-3-mini-4k-instruct via Hugging Face Transformers. Pick device (CUDA/MPS/CPU) and dtype automatically. System prompt forbids fabrication and inline citations. Default temperature 0 (greedy decoding) for factual lookup.

- [x] **1.5 `rag_pipeline.py`**
  Orchestrate retrieve → generate → strip-citations → return `Answer(answer, sources)`. Keeping orchestration here (not in `main.py`) means we can call `ask()` from a script for evals.

- [x] **1.6 `main.py` (FastAPI)**
  Endpoints: `GET /health`, `GET /sources`, `POST /ingest`, `POST /ask` (with optional `source` parameter). Open CORS in dev so the frontend on a different port can call it.

- [x] **1.7 Bug fix: citation hallucination**
  Phi-3 emitted parenthetical citations like `(db05a9.pdf, p. 12)` with **invented page numbers** even when instructed not to. Fix has two parts: (a) strengthen prompt to forbid citations and explain that sources are appended by the system, (b) add regex post-filter `_strip_model_citations` in `rag_pipeline.py` that deterministically removes any `(filename.pdf, p. N)` pattern. The structured `sources` field returned alongside the answer is built from retriever metadata, guaranteed accurate.

- [x] **1.8 Source filter on `/ask`**
  Multi-appliance corpus problem: an unfiltered query like *"how do I reset to factory defaults?"* mixes chunks from washer + fridge + printer manuals. Solution: optional `source` field in the request body that scopes retrieval via Chroma's `where={"source": filename}` filter. New `GET /sources` endpoint exposes available filenames so the frontend can render a dropdown.

- [x] **1.9 README + partner onboarding doc**
  Architecture diagram, per-file walkthrough, setup instructions, "Working as a pair" section explaining ownership split and extension ideas.

---

## Phase 2 — Evaluation infrastructure (Owner: Duranne)

- [x] **2.1 25-question eval set (`eval/questions.json`)**
  5 questions per manual × 5 manuals = 25 questions. Schema: `{id, manual_filename, question, reference_answer, type}`. Type breakdown: 19 procedural, 5 tricky, 1 factual. Reference answers were authored by reading each PDF and writing the verified answer in our own words. **This is the highest-leverage piece of work in the eval pipeline** — without verified references, no scoring is meaningful.

- [x] **2.2 `run_eval.py`**
  Loops through every question in `questions.json`, calls `POST /ask` with the right `source` filter, captures the response (answer + retrieved sources + latency_ms + status_code + any error). Writes to `eval/results.csv`. Run once per variant.

- [x] **2.3 `score_results.py` (Ollama + Qwen-2.5-3B as AI judge)**
  For each row in `results.csv`, the local Qwen judge sees the question, expected answer, and actual answer, and labels it `Correct / Partial / Wrong / Refused` with a brief reason. Output: `eval/results_scored.csv`. Uses Ollama's local API (no OpenAI cost). README documents the install + usage.

- [x] **2.4 First Qwen-scored baseline results**
  Initial run of V1 baseline through `run_eval.py` then `score_results.py`. Initial Qwen-judged accuracy: 4% strict / 26% lenient. **These numbers turn out to be misleading — see task 3.1.**

---

## Phase 3 — Methodology refinement (Owner: Shaina + Duranne)

- [x] **3.1 Human review of all 25 rows (`eval/human_scored.csv`)**
  Owner: Shaina. Read each `(question, expected, actual_answer)` triple and assign a Correct/Partial/Wrong/Refused label using human judgment. **Result: 56% disagreement with Qwen — and 13/14 disagreements were Qwen scoring more harshly than the human reviewer.** Pattern: Qwen penalizes answers that miss specific identifiers (error codes like "H19", exact temperatures) even when the substantive content is correct. Human-scored accuracy: 28% strict / 50% lenient — meaningfully higher than Qwen's 4%/26%. This divergence is itself a methodology finding for the report.

- [x] **3.2 Install RAGAS evaluation framework**
  `pip install ragas==0.2.10 datasets langchain-ollama langchain-community`. Pin RAGAS to 0.2.10 — the API changed substantially between 0.1 and 0.2. Configure with local Ollama judge (no API cost).

- [x] **3.3 Install Ollama on Shaina's machine**
  `irm https://ollama.com/install.ps1 | iex`. Pull `qwen2.5:3b` (~2 GB). Verify with `ollama --version` after restarting PowerShell so PATH refreshes.

- [x] **3.4 `run_ragas.py`**
  Owner: Duranne. Reads `eval/results.csv`, parses retrieved chunks from the `sources` column, builds a RAGAS `EvaluationDataset`, runs four metrics: **Faithfulness**, **Answer Relevancy**, **Context Precision (with reference)**, **Context Recall**. Uses `ChatOllama` to call local Qwen as the judge. Outputs `eval/ragas_<tag>.csv` (per-question) and `eval/ragas_<tag>_summary.json` (means).

- [x] **3.5 Smoke test + concurrency fix**
  First smoke test (2 samples) had **3 of 8 evaluation jobs hit `ReadTimeout()`** under default `max_workers=2` concurrency. Root cause: Qwen-2.5-3B on CPU is memory-bandwidth bound, so two parallel requests roughly double per-call latency and exceed the 300s timeout. Fix: drop `max_workers` from 2 to 1 (sequential), bump `LLM_TIMEOUT_SECONDS` from 300 to 600, drop `num_predict` from 512 to 384. Sequential mode has the same total wall time but reliable scores.

- [x] **3.6 Discovery: Faithfulness parser limitation**
  After concurrency fix, Faithfulness still scored ~0 for some samples. Root cause: Faithfulness asks the judge LLM to (a) decompose the answer into atomic claims as a JSON list, (b) verdict each claim. Qwen-2.5-3B cannot reliably emit parseable JSON for this prompt — RAGAS auto-retries with a `fix_output_format` prompt and still fails. The other 3 metrics (answer_relevancy, context_precision, context_recall) work reliably. **Decision (option A in our discussion):** keep Faithfulness in the run but treat it as partial signal; report all 4 metrics in the paper with an honest methodological caveat.

- [x] **3.7 REPORT.md skeleton**
  Paper-style markdown report at the project root. Sections drafted from existing materials: Architecture, Implementation, Methodology, Limitations, Contributions, References. Sections marked `[FILL IN AFTER V1/V2/V3]` need experiment results before they can be completed. Also exported to a Google Doc for collaborative editing.

---

## Phase 4 — Variant experiments (in progress)

The experiment is designed as a small matrix to isolate cause:

| Variant | Embedder | Chunk size / overlap | Reranker | Generator | Variable being studied |
|---|---|---|---|---|---|
| V1 (baseline) | all-MiniLM-L6-v2 | 800 / 120 | none | Phi-3-mini (transformers, fp32) | — |
| V2 (improved retrieval) | bge-small-en-v1.5 | 500 / 80 | cross-encoder MS-MARCO MiniLM | Phi-3-mini | retrieval pipeline |
| V3 (alternative generator) | bge-small-en-v1.5 | 500 / 80 | cross-encoder MS-MARCO MiniLM | Qwen-2.5-3B via Ollama | generator |

V1 → V2 isolates retrieval impact (controls for generator). V2 → V3 isolates generator impact (controls for retrieval). Bundling three retrieval changes in V2 is a deliberate trade-off — three variants is the realistic budget; we acknowledge the bundling as a limitation.

- [x] **4.1 V1 RAGAS full run** *(completed in 9.7 hours of compute)*
  Command: `python run_ragas.py --tag v1`. Outputs landed at `eval/ragas_v1.csv` and `eval/ragas_v1_summary.json`. **Mean V1 scores:** Faithfulness 0.4939 (over 11 of 25 samples that passed the parser), Answer Relevancy 0.7563 (25/25), Context Precision 0.8211 (25/25), Context Recall 0.8697 (22/25). About 17 of 100 metric-evaluations raised exceptions (Faithfulness JSON parser failures) but `raise_exceptions=False` kept the run going and the other metrics scored cleanly.

- [x] **4.2 Analyze V1 across all three scoring methods**
  Done — see REPORT.md §5.2 and §4.4. Headline finding: **retrieval metrics are high (CP 0.82, CR 0.87) while Faithfulness is moderate (0.49)** — the bottleneck is the generator, not the retriever. Cross-validates against the human review (28% strict / 50% lenient) and the Qwen judge (4% / 26%). Three scoring methods, three different views; the divergence is itself a paper finding.

- [x] **4.3 Implement V2 (improved retrieval)**
  All three coordinated changes pushed to main:
  1. `embedder.py`: `DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"`. Vector dim still 384, so Chroma schema is compatible — but vectors mean different things, so re-ingest is mandatory.
  2. `retriever.py`: `_chunk_text(chunk_size=500, overlap=80)` (was 800/120). After re-ingest the new corpus has **4531 chunks** (vs V1's 2901, a 56% increase — exactly what's expected when chunk size shrinks 38%).
  3. `retriever.py`: cross-encoder re-ranker added (`cross-encoder/ms-marco-MiniLM-L-6-v2`). `search()` now retrieves top-20 from Chroma with the bi-encoder, then the cross-encoder re-scores each (query, chunk) pair, and we return top-4. Reranker scores in V2 are raw cross-encoder logits (typically -10 to +15), not cosine similarity.
  4. `run_eval.py`: defaults changed to `START_INDEX=0, END_INDEX=25, APPEND_RESULTS=False` so the next run produces a clean V2 results.csv.

  V2 ingest verified: 5 files, 4531 chunks. V2 smoke test on `db05a9.pdf` antifreeze question confirms cross-encoder is active (logit range -3.49 to +2.59, antifreeze chunks ranked highest). **Observation:** the V2 smoke test answer differed from V1 — Phi-3 anchored on a lower-ranked drain-pump-filter chunk (-2.19) rather than the higher-ranked antifreeze chunks (+2.59, +2.49). The cross-encoder did its job; the small generator overrode it. This is precisely the failure mode V3 (different generator) tests.

- [~] **4.4 V2 RAGAS full run** *(currently in Duranne's hands)*
  Handed off 2026-05-06. Duranne's machine: pull V2 code, re-ingest with bge + 500-char chunks, run `run_eval.py` (~60–90 min), rename `results.csv` → `results_v2.csv`, then run `python run_ragas.py --tag v2 --results ../eval/results_v2.csv` overnight (~9 hrs). Outputs `eval/ragas_v2.csv` and `eval/ragas_v2_summary.json`.

- [ ] **4.5 Make `generator.py` pluggable**
  Refactor to support two generator backends behind a single `generate(question, chunks)` interface:
  - `TransformersGenerator` (current Phi-3-mini via Hugging Face)
  - `OllamaGenerator` (Qwen-2.5-3B via local Ollama HTTP API at `http://localhost:11434`)

  Switch via env var `GENERATOR_BACKEND=transformers|ollama` (default: transformers). Keep all existing prompt-engineering and citation-stripping logic identical between backends.

- [ ] **4.6 Implement V3 (Qwen generator)**
  Set `GENERATOR_BACKEND=ollama` and `MODEL=qwen2.5:3b`. Keep V2's improved retrieval pipeline (so the only delta from V2 is the generator). Run `run_eval.py --tag v3` and `run_ragas.py --tag v3`.

- [ ] **4.7 V3 RAGAS full run**
  Final overnight run.

- [ ] **4.8 Build experiment comparison table**
  One master CSV (`eval/comparison.csv`) and one rendered Markdown table aggregating: V1/V2/V3 × {Faithfulness, Answer Relevancy, Context Precision, Context Recall, Qwen-strict-accuracy, Qwen-lenient-accuracy, Human-strict-accuracy, Human-lenient-accuracy, Mean latency}. This is the headline data for the report.

---

## Phase 5 — Frontend (pending)

- [ ] **5.1 Scaffold Vite + React frontend**
  ```powershell
  cd manu
  npm create vite@latest frontend -- --template react
  cd frontend
  npm install
  echo "VITE_API_URL=http://localhost:8000" > .env.local
  npm run dev
  ```
  Confirm `localhost:5173` shows the default Vite + React page. Vite over Next.js for this project: simpler, no SSR complexity needed, faster reload.

- [ ] **5.2 Add streaming endpoint to backend** *(must be done BEFORE wiring frontend)*
  - `generator.py`: add `generate_stream(question, chunks)` that yields tokens as they're decoded (use `TextIteratorStreamer` from transformers).
  - `main.py`: add `POST /ask/stream` returning `fastapi.responses.StreamingResponse` of the streamed tokens. Test with `curl --no-buffer -X POST http://localhost:8000/ask/stream -H "Content-Type: application/json" -d '{"query": "..."}'` — should print tokens incrementally, not all at once.
  - **Honest warning:** streaming has sharp edges (CORS for streaming, buffering, token-boundary parsing). If this burns more than 4 hours, drop streaming and submit a non-streaming demo with streaming in "future work."

- [ ] **5.3 Build frontend components**
  Four files in `frontend/src/components/`:
  - `SourceSelector.jsx` — calls `GET /sources` on mount, renders a `<select>` dropdown, lifts the chosen filename up to `App` via a callback prop.
  - `ChatWindow.jsx` — owns the question input + submit button, sends `POST /ask/stream`, reads streaming tokens via `fetch` + `ReadableStream.getReader()`, appends to the assistant bubble as they arrive.
  - `MessageBubble.jsx` — purely presentational. Different styles for user vs assistant turns.
  - `SourcesPanel.jsx` — renders the retrieved `sources` array (filename, page, snippet) below each assistant answer. Toggleable show/hide.

  Wire all four into `App.jsx` as the state orchestrator.

- [ ] **5.4 End-to-end test + capture screenshots**
  Run uvicorn + `npm run dev` simultaneously. Ask 3-4 questions through the UI covering different manuals. Catch CORS issues, JSON parsing bugs, layout breakage. **Take screenshots** — they're going in the paper and the slides.

---

## Phase 6 — Report (partially done)

- [x] **6.1 REPORT.md skeleton**
  Created. Sections covering Abstract, Introduction, System Design, Implementation, Methodology, Results, Discussion, Limitations, Contributions, References, Appendices.

- [x] **6.2 Google Doc version**
  Drag-and-drop uploaded to Google Drive and converted via "Open with Google Docs". Useful for collaborative editing.

- [x] **6.3 Fill in V1 baseline section**
  Done. REPORT.md §3.3 has real latency numbers; §4.4 has concrete Faithfulness failure rates (11/25) and the `service_manual_18_q04` example of Qwen over-strictness; §5.1 has the V1 row of the master comparison table; §5.2 is fully written with the diagnostic interpretation ("retrieval is strong, generator is the weak link"); §5.5 has four V1-supported failure case studies including the OE-error hallucination, the water-hammer wrong-topic retrieval, the compounding-failure case, and the citation hallucination story.

- [ ] **6.4 Fill in V1 → V2 comparison section**
  Section 5.3. Did improved retrieval help? Which metric moved most? Are there per-question differences worth highlighting (e.g., a question that V1 got Wrong but V2 got Correct)?

- [ ] **6.5 Fill in V2 → V3 comparison section**
  Section 5.4. Does the generator matter once retrieval is good? Compare answer quality and latency. (Qwen via Ollama on CPU is typically faster than Phi-3 via transformers — that's a real finding for a deployable system.)

- [ ] **6.6 Write Abstract**
  Always write last. ~150 words. Problem → approach → result → implication.

- [ ] **6.7 Write Discussion sections (6.1, 6.2, 6.3)**
  What worked, what surprised us, engineering lessons. The "Qwen-vs-human disagreement" story is the strongest finding here.

- [ ] **6.8 Final pass + rewrite skeleton prose in your own voice**
  Read every section. The skeleton was written by an assistant; rewrite anything that sounds generic. Your professor will know.

- [ ] **6.9 Convert to required submission format**
  If the professor wants Word: File → Download → .docx in Google Docs. If PDF: same menu, → PDF.

---

## Phase 7 — Presentation (pending)

- [ ] **7.1 Build slide deck (~10 slides)**
  Suggested order: Title, Problem, Architecture, Demo screenshots (from task 5.4), Eval methodology, Results table (from task 4.8), Biggest finding (likely the Qwen-vs-human disagreement), Limitations, Future work, Q&A / Thank you. Use the report as your script.

- [ ] **7.2 Dry-run presentation with live demo**
  Run uvicorn + frontend live. Talk through slides while doing a demo. Time it. Find broken things now, not in front of the class. Do this **at least the day before** the real presentation, not the morning of.

- [ ] **7.3 Submit + present**
  Final code push, final report submission, deliver the talk.

---

## How we work together

- **Branches over `main`.** Each piece of meaningful work goes on a feature branch, opens a PR, the other reviews. Even with two people this catches bugs and keeps history readable.
- **Don't both touch the same file at once.** If unavoidable, communicate before merging.
- **Commit messages explain *why*.** "Add reranker" is bad; "Add cross-encoder reranker — top-k retrieval was returning near-duplicates that crowded out diverse evidence" is good.
- **`git pull` before starting work each session.** Keeps both copies in sync.
- **`POST /ingest` after pulling** if anyone changed embedding, chunking, or PDFs — the index is regenerable but stale indexes mean stale answers.

---

## Glossary

- **RAG** — Retrieval-Augmented Generation. Retrieve relevant document chunks → feed them to an LLM as context → generate an answer grounded in those chunks.
- **Embedder** — model that turns text into a fixed-size vector. Two texts with similar meaning have similar vectors.
- **ChromaDB** — local vector database. Stores embeddings + raw text + metadata, supports nearest-neighbor search.
- **Re-ranker** — model that scores candidate chunks against the query directly (cross-encoder), used to refine the initial top-k from the embedder.
- **RAGAS** — Retrieval Augmented Generation Assessment. Evaluation framework with metrics like Faithfulness, Answer Relevancy, Context Precision, Context Recall.
- **Ollama** — local LLM runtime. Runs quantized models like Qwen-2.5-3B on CPU/GPU and exposes an HTTP API.
- **Phi-3-mini-4k-instruct** — Microsoft's 3.8B-parameter instruction-tuned model. Our default generator.
- **Qwen-2.5-3B** — Alibaba's 3B-parameter model. Our judge LLM (and V3's generator).
- **Faithfulness (RAGAS)** — proportion of answer claims supported by retrieved context. Catches hallucination.
- **Context Precision** — proportion of retrieved chunks that are relevant. Diagnoses retrieval quality.
- **Context Recall** — proportion of reference-answer info present in retrieved chunks. Diagnoses what retrieval missed.

---

## Quick links

- Repository: https://github.com/shaittoo/manual-rag-chatbot
- Architecture deep-dive: `README.md` (sections "How the code is organized" and "Working as a pair")
- Report draft: `REPORT.md` (and Google Doc copy)
- Eval data:
  - `eval/questions.json` — 25 questions with verified reference answers (V1 baseline)
  - `eval/results_v1.csv` — V1 raw `/ask` outputs (renamed from `results.csv` when V2 work began)
  - `eval/results_scored.csv` — V1 Qwen-judged Correct/Partial/Wrong/Refused labels
  - `eval/human_scored.csv` — V1 human-reviewed labels with `human_score`, `agreement`, `human_reason`
  - `eval/ragas_v1.csv` — V1 per-question RAGAS scores
  - `eval/ragas_v1_summary.json` — V1 mean scores + run config
  - `eval/results_v2.csv` — V2 raw outputs *(produced when Duranne completes 4.4)*
  - `eval/ragas_v2.csv` and `eval/ragas_v2_summary.json` — V2 RAGAS *(pending)*
