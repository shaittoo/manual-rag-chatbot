# Manu — A Retrieval-Augmented Chatbot for Product Manuals

**Authors:** Shaina Talisay, Duranne B. Duran
**Course:** *(fill in: course code, semester)*
**Instructor:** *(fill in)*
**Date:** *(fill in submission date)*

---

> **NOTE TO YOURSELVES (delete before submission):**
> Sections marked **[FILL IN AFTER V1/V2/V3]** require experiment results we don't have yet. Everything else can be drafted now while RAGAS runs.
> Sections marked **[REVIEW]** have starter content drawn from existing materials — read carefully and rewrite in your own voice before submitting.

---

## Abstract

Product manuals usually contain the answer a user needs, but finding it means searching a long PDF, and keyword search fails when people describe a problem in their own words. We built Manu, a retrieval-augmented chatbot over five appliance manuals: it embeds manual text with MiniLM, retrieves the most relevant passages from a ChromaDB index (with cross-encoder re-ranking in later variants), and generates a grounded answer with cited sources using Phi-3-mini or Qwen-2.5-3B, while a small feed-forward classifier auto-routes each question to the correct manual. We evaluated three pipeline variants on a 25-question set through three independent lenses — an automatic Qwen judge, human review, and RAGAS. Retrieval was strong (context recall ≈ 0.90), but generator faithfulness was the bottleneck (0.18 at baseline, rising to 0.45 after retrieval improvements), and the automatic judge proved far stricter than humans (4% vs 28% on the baseline). Our results indicate that grounding, not retrieval, is the limiting factor — and that the evaluation method itself materially shapes the reported score.

---

## 1. Introduction

### 1.1 Problem

Modern household and office equipment ships with PDF user manuals that are often dense, long, and hard to search. A user with a specific question — *"How do I drain the antifreeze from my LG washer?"* — typically must scroll through dozens of pages and parse technical language to find the answer. Linear lookup does not scale across the multiple appliances a single household owns.

### 1.2 Why Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) addresses this by combining a vector-similarity search over chunked document text with a large language model that generates natural-language answers grounded in retrieved passages. Compared to fine-tuning a model per manual, RAG:

- generalizes across new manuals without retraining,
- preserves source attribution (we can cite the file and page),
- is substantially cheaper to develop and run.

### 1.3 Scope

This project **does not train or fine-tune any model**. The "deep learning" components are pretrained: `sentence-transformers/all-MiniLM-L6-v2` for embeddings and `microsoft/Phi-3-mini-4k-instruct` for generation. Our contribution is the system design, the evaluation methodology, and a comparative analysis of three configuration variants of the pipeline.

### 1.4 Contributions

- A working CPU-runnable RAG pipeline for multi-appliance manual question-answering, exposed via a FastAPI HTTP API.
- A 25-question evaluation set with manually verified reference answers spanning five distinct appliance categories.
- A multi-method evaluation framework combining (a) AI-assisted scoring via Ollama+Qwen, (b) human review, and (c) RAGAS metrics.
- A comparison of three pipeline variants (V1 baseline, V2 improved retrieval, V3 alternative generator) under this framework.

---

## 2. System Design

### 2.1 Architecture Overview

```
PDF files in manuals/
    │
    ├──[pypdf]──> page text
    │              │
    │              ├──[char chunker w/ overlap]──> 800-char chunks
    │              │                                  │
    │              │                                  ▼
    │              │                          [MiniLM embedder]
    │              │                                  │
    │              │                                  ▼
    │              │                          ChromaDB (persistent, cosine)
    │              │                                  ▲
    │              │                                  │
User query ─[MiniLM embedder]──> query vector  ───────┘
                                                   │
                                                   ▼
                                            top-k chunks (k=4)
                                                   │
                                                   ▼
                  [Phi-3-mini-4k-instruct + system prompt + context]
                                                   │
                                                   ▼
                                  grounded answer + structured sources
```

### 2.2 Component Summary

**Embedder.** `sentence-transformers/all-MiniLM-L6-v2`, a 384-dimensional, ~80MB English sentence encoder. We L2-normalize all embeddings so cosine similarity reduces to a dot product. The model is loaded once per process and cached.

**Retriever.** Reads PDFs with `pypdf`, splits each page into overlapping 800-character chunks (120-character overlap), embeds them, and writes the embeddings, raw text, and `(filename, page)` metadata to a persistent ChromaDB collection at `backend/chroma_db/`. At query time we embed the question with the same model and request the top-`k` (default `k=4`) nearest chunks. An optional `source` filter restricts retrieval to a single filename — necessary because our corpus contains five distinct appliance manuals and an unfiltered query can mix chunks across them.

**Generator.** Pluggable behind a `GeneratorBackend` protocol with two interchangeable implementations selected per request: `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters) loaded via Hugging Face `transformers`, or `Qwen-2.5-3B` served through a local Ollama HTTP API. Both share one prompt builder. The system prompt constrains the model to answer only from retrieved context, refuse if the context does not support an answer, and **not** emit inline citations. Decoding is greedy (`temperature=0.0`) to minimize fabrication. The generator also accepts optional conversation `history`, so follow-up questions ("what about after that?") stay grounded in the right topic.

**Pipeline orchestrator (`rag_pipeline.ask`).** Composes the three components: retrieve → generate → strip any leftover citations → return a structured `Answer` containing the cleaned text and a list of source records. It also enriches the retrieval query with recent history so pronoun-laden follow-ups still retrieve the right chunks.

**HTTP API (`main.py`).** FastAPI exposing six endpoints:
- `GET /health` — liveness check
- `GET /sources` — list of indexed filenames (for the frontend dropdown)
- `POST /ingest` — wipe and rebuild the Chroma collection from `manuals/`
- `POST /ask` — RAG query, optionally scoped to a single source; accepts `generator_backend` and `history`
- `POST /classify` — predict which manual a query is about, without retrieval or generation (see §5.6)
- `POST /ask_auto` — classify → ask in one call, auto-routing to the predicted manual (see §5.6)

### 2.3 Notable Design Decisions

**Source-filter on `/ask`.** Without filtering, a query like *"how do I reset to factory defaults?"* retrieves chunks from every manual that mentions a reset, and the generator hallucinates a Frankenstein procedure. A user-supplied `source` parameter restricts retrieval to one manual, which is the cheapest way to handle a multi-appliance corpus. We later added an automatic version — a trained manual-classifier stage exposed via `/ask_auto` — described in §5.6.

**Citation strip post-processing.** Despite being instructed *not* to cite inline, Phi-3-mini repeatedly emitted parenthetical references like `(db05a9.pdf, p. 12)` — and the page numbers were fabricated. In one observed case the model claimed evidence from page 12 when all retrieved chunks were from pages 31–32. Rather than continually re-tuning the prompt, we apply a regex post-filter (`rag_pipeline._strip_model_citations`) that removes any matching pattern. The structured `sources` field returned alongside the answer is built from retriever metadata, not the model's output, so cited sources are guaranteed accurate.

**Greedy decoding (temperature=0).** Lowered from an initial `0.2` after observing that low-but-nonzero temperatures produced occasional fabricated facts. For factual lookup the most-likely token at each step is empirically more grounded.

---

## 3. Implementation

### 3.1 Code Organization

```
manu/
├── backend/
│   ├── main.py              # FastAPI app + 6 routes
│   ├── rag_pipeline.py      # Orchestrates retrieve + generate (+ history)
│   ├── embedder.py          # sentence-transformers wrapper (MiniLM)
│   ├── retriever.py         # PDF parsing, chunking, ChromaDB I/O
│   ├── generator.py         # Pluggable: Phi-3 (Transformers) / Qwen (Ollama)
│   ├── manual_classifier.py # FFN classifier inference (auto-routing)
│   ├── train_classifier.py  # Trains the classifier (k-fold, loss curves)
│   ├── run_eval.py          # Hits /ask for each question, logs results
│   ├── score_results.py     # AI-assisted scoring via Ollama+Qwen
│   ├── run_ragas.py         # RAGAS evaluation
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── tests/               # pytest suite (see §3.5)
│   └── manuals/             # PDFs (gitignored)
├── frontend/                # Vite + React chat UI
│   └── src/
│       ├── App.jsx          # Chat orchestrator; model dropdown; calls the API
│       ├── intent.js        # Greeting / thank-you / follow-up detection (testable)
│       ├── intent.test.js   # Vitest unit tests for the intent heuristics
│       └── components/       # ChatWindow and message UI
└── eval/
    ├── questions.json       # 25 questions with reference answers
    ├── results_*.csv        # raw /ask outputs (per variant)
    ├── results_scored_*.csv # Qwen judge labels (per variant)
    ├── human_scored.csv     # human review labels
    └── ragas_v{1,2,3}.csv   # RAGAS scores (per variant)
```

### 3.2 Stack

- **FastAPI** + **Uvicorn** — HTTP API
- **ChromaDB (PersistentClient)** — local vector store
- **sentence-transformers** — embedding model
- **Hugging Face Transformers** + **PyTorch** — Phi-3-mini inference
- **pypdf** — PDF text extraction
- **RAGAS** — evaluation framework
- **Ollama** + **Qwen-2.5-3B** — local LLM judge for scoring, and an alternate generator backend
- **LangChain (Ollama / HuggingFace adapters)** — bridging Ollama and RAGAS
- **Vite + React** — frontend chat UI (model dropdown, sources panel, follow-up handling)
- **pytest** (backend) + **Vitest** (frontend) — automated test suites (see §3.5)

### 3.3 Hardware

All `/ask` generation across V1, V2, and V3 ran on **CPU-only** Windows 11 hardware. Across all variants, Phi-3-mini (or Qwen-2.5-3B in V3) on CPU produced an answer in roughly **130–170 s on average** per question, with per-question variance driven primarily by output length under greedy decoding.

RAGAS scoring is **separable from generation**, and Duranne re-ran the V1/V2/V3 judge passes on a machine with NVIDIA CUDA support. Qwen-2.5-3B as judge runs ~10× faster on CUDA via Ollama than on CPU: a full 25-question × 4-metric pass took **~24 minutes on CUDA** (V1: 1429 s, V2: 1045 s, V3: 1089 s) versus the **~9.7 hours** the original V1 CPU judge run took. We discuss the implications of judge-hardware change on metric reproducibility in §4.4 and §6.4.

### 3.4 Deep Learning Components and Concepts

This project uses deep learning at three layers of the pipeline plus one component that we trained end-to-end. Naming the underlying concepts explicitly here, since they otherwise hide behind library calls:

**(i) Embedding (transformer encoders, frozen).** `sentence-transformers/all-MiniLM-L6-v2` (V1) and `BAAI/bge-small-en-v1.5` (V2/V3) are encoder-only transformers that map a tokenized input sentence to a 384-dimensional dense vector via stacked self-attention layers. The output vector encodes semantic similarity geometrically — sentences with similar meaning lie close in the vector space, regardless of surface form. We L2-normalize the vectors so cosine similarity reduces to a dot product. *DL concepts: self-attention, contextual embeddings, dense representation learning, contrastive pretraining, vector-space semantics.*

**(ii) Re-ranking (cross-encoder transformer, frozen, V2/V3 only).** `cross-encoder/ms-marco-MiniLM-L-6-v2` is an encoder transformer that takes a `(query, document)` pair as a *single concatenated input* and outputs a relevance logit. This contrasts with the bi-encoder of (i), which encodes query and document independently and only compares them post-hoc via similarity. The cross-encoder's joint encoding captures finer query-document interactions — at the cost of having to score each candidate at query time rather than amortizing the encoding into a precomputed index. We use the cross-encoder over the top-20 bi-encoder candidates and keep the top-4 by reranker score. *DL concepts: cross-encoder vs. bi-encoder retrieval, two-stage retrieve-then-rerank, transformer-based pairwise scoring.*

**(iii) Generation (decoder-only transformer, frozen).** `microsoft/Phi-3-mini-4k-instruct` (V1, V2; 3.8B parameters) and `Qwen-2.5-3B-Instruct` via Ollama (V3; ~3B parameters, INT4-quantized GGUF) are decoder-only transformers that perform autoregressive token-by-token generation conditioned on the retrieved context plus a system prompt. We use **greedy decoding** (`temperature=0`) — selecting the argmax token at each step — rather than nucleus / top-p sampling, on the principle that factual lookup benefits from consistency over diversity. *DL concepts: causal self-attention, autoregressive generation, instruction-tuned language models, decoding strategies (greedy vs. sampling), prompt-conditioned generation.*

**(iv) Manual classifier (small FFN, trained end-to-end by us).** We train a feed-forward neural network on top of frozen MiniLM embeddings to predict which manual a query is about (described in detail in §5.6). This is the only component of the system whose weights are updated by gradient descent in our work; everything else is pretrained. *DL concepts demonstrated: supervised classification, feed-forward networks, ReLU non-linearity, dropout regularization, transfer learning (frozen embedder + trainable head), cross-entropy loss, Adam optimization with weight decay (L2 regularization), k-fold cross-validation, data augmentation.*

The project is therefore not "RAG without deep learning" — every component on the inference path is a deep neural network, and one of them is trained end-to-end with the full supervised-learning workflow. What this project does *not* include, and which would be obvious next steps, is fine-tuning the larger pretrained components (e.g., LoRA on Phi-3-mini against a synthesized Q-A dataset from the manuals); we discuss these in §7.2.

### 3.5 Testing and Verification

Separately from the answer-quality evaluation (§4), the codebase carries an automated test suite that verifies *program behaviour* — logic that can be checked deterministically without loading a model.

**Backend (`backend/tests/`, pytest — 46 tests).** Coverage includes: the chunking windows in `retriever._chunk_text` (size, overlap, sentence-boundary snapping, empty input); the citation-stripping safety net `rag_pipeline._strip_model_citations`; source/snippet shaping and history-aware retrieval-query construction; the generator's prompt and history formatting; the `get_generator` factory and the Ollama backend's graceful fallback when the server is unreachable; and the FastAPI request contract (`/health`, request-validation 422s, and `/ask`, `/sources`, `/classify`, `/ask_auto` happy paths with the model layer monkeypatched). A `conftest.py` registers lightweight stubs for `torch`, `transformers`, `sentence-transformers`, and `chromadb` **only when those libraries are absent**, so the same tests run unchanged on a lightweight CI box and on the full GPU machine — without downloading the ~7 GB model stack.

**Eval-set validation.** `test_eval_dataset.py` asserts that `eval/questions.json` is well-formed: 25 entries, required keys present, unique IDs, the `type` field within `{factual, procedural, tricky}`, substantive reference answers, and at least two questions per manual. This guards the evaluation harness against a malformed entry silently skewing reported accuracy.

**Frontend (`frontend/src/intent.test.js`, Vitest — 25 assertions).** The pre-network routing heuristics — greeting, thank-you, and follow-up detection — were extracted from `App.jsx` into a standalone `intent.js` module specifically so they can be unit-tested in isolation. The tests pin down the behaviour that most affects UX: that greetings and thank-yous are answered locally, that a clearly named new topic is *not* treated as a follow-up (avoiding conversation-history contamination), and that short pronoun-based questions are.

**Scope.** These tests verify code logic, not answer quality; the latter still requires the live model and is what §4's three scoring lenses measure. The split is deliberate: fast, deterministic tests for the code, and a separate model-in-the-loop evaluation for the system's outputs.

---

## 4. Evaluation Methodology

### 4.1 Evaluation Set

We constructed `eval/questions.json` containing 25 questions, 5 per manual, distributed across 5 manuals:

| Manual | Filename | Questions |
|---|---|---|
| Panasonic AC service manual | `Service-Manual-18.pdf` | 5 |
| HP LaserJet Pro M329/M428/M429 | `c06184015.pdf` | 5 |
| Epson ET-4850 | `cpd60205.pdf` | 5 |
| Samsung RF-series refrigerator | `DA68-04752Q_FDR_RF6500C_3Door_EN_MES_CFR_260209.pdf` | 5 |
| LG washing machine | `db05a9.pdf` | 5 |

Question types: 19 procedural, 5 tricky (e.g., "is this normal?"), 1 factual.

**Reference answers** were authored manually by reading each PDF and transcribing the relevant procedure or fact in our own words. References are intentionally exhaustive (e.g., listing every cause an appliance manual identifies for a symptom), which has implications for scoring strictness — see §4.4.

### 4.2 Three Scoring Methods

We evaluate each system variant with **three** complementary scoring methods. The redundancy is deliberate: each method has different biases, and divergence among them is itself informative.

**Method 1 — AI-assisted Correct/Partial/Wrong/Refused (Qwen-2.5-3B as judge).**
Implemented in `score_results.py`. For each row, the judge receives the question, reference answer, generated answer, and any error string, and returns one of four labels with a brief reason. This was originally the only auto-scoring method; it served as our first signal but proved unreliable on its own (see §4.4).

**Method 2 — Human review.**
Both authors independently reviewed all 25 rows and assigned the same four-label rubric. Disagreements were resolved by a follow-up read of the specific PDF page. Final labels are stored in `eval/human_scored.csv` with `human_score`, `agreement` (with Qwen), and `human_reason` (justification for disagreements).

**Method 3 — RAGAS metrics.**
We use the four canonical RAGAS metrics:
- **Faithfulness** — proportion of claims in the answer that are supported by retrieved context.
- **Answer Relevancy** — semantic alignment between the question and the answer (via synthetic question generation).
- **Context Precision (with reference)** — proportion of retrieved chunks that are relevant given the reference answer.
- **Context Recall** — proportion of reference-answer information present in retrieved context.

The judge LLM for RAGAS is also `qwen2.5:3b` via local Ollama (no external API). This means RAGAS results inherit the small-judge limitations of Method 1; we discuss this in §4.4 and §6.

### 4.3 Experiment Matrix

We compare three pipeline variants on the same 25-question set:

| Variant | Embedder | Chunk size / overlap | Re-ranker | Generator | Variable being studied |
|---|---|---|---|---|---|
| **V1 (baseline)** | all-MiniLM-L6-v2 | 800 / 120 | none | Phi-3-mini (transformers, fp32) | — |
| **V2 (improved retrieval)** | bge-small-en-v1.5 | 500 / 80 | cross-encoder/ms-marco-MiniLM-L-6-v2 | Phi-3-mini (same as V1) | retrieval pipeline |
| **V3 (alternative generator)** | bge-small-en-v1.5 | 500 / 80 | cross-encoder/ms-marco-MiniLM-L-6-v2 | Qwen-2.5-3B via Ollama | generator |

V1 → V2 isolates the impact of retrieval improvements (controls for the generator). V2 → V3 isolates the impact of the generator (controls for retrieval). This design accepts the limitation that V2 bundles three retrieval changes (embedder + chunking + re-ranker); we discuss in §6 which of those three would most warrant follow-up ablation.

### 4.4 Methodological Limitations of LLM-as-Judge

Pilot results revealed two systematic issues with our Qwen-2.5-3B judge:

**Over-strictness.** Across the 25-row V1 baseline, Qwen and human scoring agreed on only **11 of 25 rows (44%)**. Of the 14 disagreements, **13 were Qwen rating an answer more harshly than the human reviewer** (e.g., labeling an answer "Wrong" when the human review judged it "Partial"). The pattern is consistent: Qwen penalizes answers that miss a specific identifier (an error code like "H19" or "H99", an exact temperature threshold) even when the substantive content is correct. For example, on `service_manual_18_q04` ("the indoor fan is not spinning"), the system answer correctly listed real fan-motor failure causes drawn from the manual (winding short, broken wire, lead wire, Hall IC, PCB faults) but did not explicitly name the manual's "H19" diagnostic code; Qwen labelled this *Wrong* while the human reviewer labelled it *Partial*. Headline numbers shift accordingly: **Qwen-judged accuracy was 4% strict / 26% lenient**, while **human-judged accuracy was 28% strict / 50% lenient** — a 24-percentage-point gap on both measures.

**Faithfulness-prompt parser failures.** RAGAS's Faithfulness metric requires the judge to (a) decompose the answer into atomic claims as a JSON list and (b) verdict each claim against the retrieved context. Qwen-2.5-3B cannot reliably emit parseable JSON for this prompt. Faithfulness coverage varies by variant: **V1 14/25, V2 19/25, V3 ~17/25**; Answer Relevancy and Context Precision score 25/25 across all variants, and Context Recall lands in the 19–25 range. Failures raise either `RagasOutputParserException` (parser exhausted retries) or `AttributeError('StringIO' object has no attribute 'statements')` (RAGAS internal fallback after parser failure). Reported Faithfulness means are therefore computed over partial samples and should be read as directional; a stronger judge (Qwen-2.5-7B, Llama-3.1-8B, or GPT-4-class) would be required to obtain a complete Faithfulness score per row.

**Judge-hardware variance.** A second observation, surfaced when the V1 RAGAS pass was re-run on CUDA versus the original CPU run: the judge produced systematically different per-row Faithfulness scores even with **identical answers and contexts**. Sample-level scoring shifted (different rows now scored 0.0, different rows now scored ≥0.5), and the V1 mean changed by ~0.3 points. We attribute this to fp16 vs fp32 numerical differences in the judge's logits at decision boundaries, possibly compounded by Ollama version differences between runs. A practical implication: **even with `temperature=0`, RAGAS-with-a-3B-judge is not bit-deterministic across hardware**, and reported metric values should be paired with hardware/version metadata for reproducibility.

**Mitigation.** We report all three scoring methods (Qwen labels, human labels, RAGAS metrics) for V1 and discuss their divergence as a finding rather than collapsing to a single number. The methodological observation — that LLM-as-judge under a 3B-parameter model is systematically biased and partially unreliable — is itself one of the contributions of this work.

---

## 5. Results

### 5.1 Variant Comparison

The master comparison table aggregates all three pipeline variants under the RAGAS metrics (judge: Qwen-2.5-3B on CUDA via Ollama). Human and Qwen-CPRW labels are reported for V1 only as a methodology cross-check.

| Variant | Faithfulness (n) | Answer Relevancy | Context Precision | Context Recall (n) | Mean Latency (s) |
|---|---|---|---|---|---|
| V1 baseline (MiniLM, 800/120, no reranker, Phi-3) | 0.184 (n=14) | 0.721 | 0.861 | 0.904 (n=22) | ~165 |
| V2 retrieval+ (bge, 500/80, cross-encoder, Phi-3) | 0.448 (n=19) | 0.732 | 0.856 | 0.663 (n=19) | ~145 |
| V3 generator (V2 retrieval, Qwen-2.5-3B) | 0.378 (n≈17) | 0.513 | 0.852 | 0.663 (n≈19) | ~145 |

Faithfulness `(n=...)` and Context Recall `(n=...)` denote the number of samples for which that metric was successfully scored; the mean is taken over those samples only. Variation in `n` across variants reflects 3B-judge parser instability (§4.4), not differences in the underlying systems.

For V1 specifically, we additionally have:

| V1 scoring method | Strict accuracy | Lenient accuracy |
|---|---|---|
| Qwen-2.5-3B Correct/Partial/Wrong/Refused (auto) | 4% | 26% |
| Human review (manual) | 28% | 50% |

The **24-percentage-point gap on strict accuracy and a similar gap on lenient accuracy** between Qwen-judged and human-judged V1 is a methodology finding in its own right (see §4.4).

### 5.2 V1 Baseline Detail

**Top-line scoring across three methods.** The Qwen judge labelled the baseline at **4% strict / 26% lenient accuracy**; human review labelled it at **28% strict / 50% lenient accuracy** — a 24-percentage-point gap on both measures, driven by the over-strictness pattern identified in §4.4. RAGAS produced four mean scores (computed over successfully-scored samples; CUDA judge):

| Metric | V1 Mean | Samples | What it measures |
|---|---|---|---|
| Faithfulness | 0.184 | 14/25 | Are the answer's claims supported by the retrieved chunks? |
| Answer Relevancy | 0.721 | 25/25 | Does the answer address the question? |
| Context Precision (with reference) | 0.861 | 25/25 | Are the retrieved chunks relevant given the reference answer? |
| Context Recall | 0.904 | 22/25 | Did retrieval find the information needed for the reference answer? |

**Diagnostic interpretation.** The four RAGAS metrics partition the failure surface differently from the single Correct/Partial/Wrong/Refused label. Specifically, the gap between the two retrieval metrics (Context Precision 0.86, Context Recall 0.90) and Faithfulness (0.18 over 14 samples) is informative: **retrieval is finding the right chunks most of the time, but the generator does not always remain grounded in them**. The Faithfulness floor of 0.18 is striking — across the rows we could measure, *fewer than one in five claims emitted by Phi-3-mini was supported by the retrieved chunks*. This pattern is also visible at the individual sample level. For `lg_wm4200_wm4000_q02` ("My washer will not drain or the OE error is showing"), Context Precision was 0.64 and Context Recall was 0.67 — i.e. the retriever returned partially correct chunks. Phi-3-mini nevertheless answered with a fabricated definition ("OE = Door Open Error"); the manual itself defines OE as the *water Outlet error*. Faithfulness for this row scored 0.0, consistent with the model's invented claim. The takeaway: **for V1, retrieval is operating well above the generator-faithfulness floor, suggesting that downstream improvements should target the generation stage at least as much as the retrieval stage**. We test this hypothesis directly by comparing V2 (retrieval-only changes, same generator) against V3 (same retrieval as V2, different generator) — see §5.3 and §5.4.

**Latency.** Mean wall-clock latency for a single `/ask` call was approximately 165 s, with a range of 76 s to 204 s across the 25 questions. Variance is dominated by output length, since greedy decoding is token-by-token on CPU.

**Per-method headline accuracy.** Reading the same baseline through three lenses, we obtain three quite different statements: *"the system answers 4% of questions correctly"* (Qwen judge); *"the system answers 28% of questions correctly"* (human review); *"the system retrieves with 0.86 precision and 0.90 recall but its generator stays grounded in the retrieved context only ~18% of the time on the rows we could measure"* (RAGAS). Each statement is internally consistent with its method; their divergence is the methodological finding of §4.4.

### 5.3 V1 → V2: Effect of Retrieval Improvements

V2 changes three retrieval-side components simultaneously while holding the generator fixed at Phi-3-mini: (a) embedder swapped from MiniLM to bge-small-en-v1.5; (b) chunk size reduced from 800/120 to 500/80; (c) cross-encoder MS-MARCO reranker added between Chroma top-20 retrieval and the final top-4 cut.

| Metric | V1 baseline | V2 retrieval+ | Δ |
|---|---|---|---|
| Faithfulness | 0.184 (n=14) | **0.448 (n=19)** | **+0.264** ↑↑ |
| Answer Relevancy | 0.721 | 0.732 | +0.011 → |
| Context Precision (with reference) | 0.861 | 0.856 | -0.005 → |
| Context Recall | **0.904 (n=22)** | 0.663 (n=19) | **-0.241** ↓↓ |

**Two findings, in opposite directions.**

**Finding 1 — Faithfulness more than doubled.** With identical generator settings, V2's retrieval pipeline produced answers that the judge could verify against retrieved context far more often: 0.18 → 0.45 over partial samples, with the number of judge-parseable rows also rising (14 → 19). The generator is staying *more grounded* when given V2's tighter, reranked context. The cross-encoder, in particular, suppresses near-duplicate chunks and surfaces semantically central evidence — the generator's claims are more often traceable to a single retrieved chunk.

**Finding 2 — Context Recall dropped substantially.** V1's 800-character chunks frequently captured the entirety of a multi-step procedure (e.g. all eight steps of the LG washer antifreeze procedure on page 31) inside a single chunk. V2's 500-character chunks split such procedures across two or three chunks; if the top-4 retrieval surfaces only some of those fragments, content present in the reference answer is missing from the model's context. The judge's Context Recall metric — which measures how much of the *reference answer's* information is present in the retrieved chunks — accordingly dropped from 0.90 to 0.66. This is a direct, expected consequence of smaller chunks: each chunk is more topically focused but covers less material.

**Net interpretation.** V2 trades content coverage for content faithfulness. Whether V2 is "better" than V1 depends on what the system's user values: a generator that is grounded in less-complete context (V2) versus a generator that has more complete context but is less reliably grounded in it (V1). Context Precision and Answer Relevancy moved by less than one standard deviation of judge noise, so retrieval *quality* (as opposed to coverage) is roughly unchanged — V2's gain on Faithfulness is real, V2's loss on Recall is real, the other metrics are noise.

**Implication for V3.** Because Faithfulness is bounded by the generator's behaviour, not by retrieval quality, V3's generator swap (Phi-3 → Qwen-2.5-3B) is a direct test of whether *that bottleneck* moves.

### 5.4 V2 → V3: Effect of Generator Choice

V3 holds V2's improved retrieval pipeline constant and swaps the generator from Phi-3-mini-4k-instruct (3.8B parameters, fp32 via Hugging Face Transformers) to Qwen-2.5-3B-Instruct (~3B parameters, 4-bit GGUF via Ollama). The retrieval pipeline is identical, so Context Precision and Context Recall should be (and are) approximately equal to V2.

| Metric | V2 (Phi-3) | V3 (Qwen-3B) | Δ |
|---|---|---|---|
| Faithfulness | 0.448 (n=19) | 0.378 (n≈17) | -0.070 ↓ |
| Answer Relevancy | 0.732 | **0.513** | **-0.219** ↓↓ |
| Context Precision (with reference) | 0.856 | 0.852 | -0.004 → |
| Context Recall | 0.663 | 0.663 | 0.000 → |

**Sanity check first.** Context Precision and Context Recall barely move (Δ ≤ 0.005), which is exactly what we expect — retrieval is identical between V2 and V3, so the only metric variance for those two is judge noise. ✓

**The headline: Phi-3-mini outperforms Qwen-2.5-3B as the generator in this setup.** Both Faithfulness (-0.07) and Answer Relevancy (-0.22) move down. The Answer Relevancy drop in particular is large: Qwen-3B's answers are visibly less on-topic relative to the question than Phi-3's, even given the same retrieved context. Manual inspection of a sample of V3 answers (not reproduced here for length) confirms a tendency in Qwen-3B to drift into related but tangential subtopics — for example, on a question about printer paper jams, Qwen-3B's answer included extended discussion of paper-tray loading procedures that, while present in the retrieved chunks, do not directly answer "what should I check first?".

**One important caveat.** Qwen-2.5-3B is also our **judge model** for RAGAS scoring. In V3, the judge is evaluating answers from a generator that shares its training distribution — there is at least theoretical room for the judge to evaluate Qwen-3B's outputs more harshly (or more leniently) than it evaluates Phi-3-mini's, due to subtle stylistic biases. We did not observe a clear self-favouring bias (V3 scores worse, not better), but a fully clean V2-vs-V3 comparison would require an *independent* judge model — for example, GPT-4-class via API, or Llama-3.1-8B as judge run separately.

**Net interpretation.** Within the limits of our judge methodology, the V2 → V3 generator swap reduces answer quality. **The bottleneck identified in §5.2 (generator-side faithfulness) is not solved by switching to a similarly-sized open model**; if anything, Qwen-2.5-3B is somewhat worse than Phi-3-mini for this specific retrieval-augmented manual-QA task at the 3-4B parameter scale. A genuine improvement to the generator would likely require (a) a larger model (e.g. Phi-3-medium 14B, Llama-3.1-8B), (b) supervised fine-tuning of Phi-3 on synthesized manual-Q&A pairs, or (c) a different decoding strategy than greedy. We list these as future work in §7.

### 5.5 Specific Failure Cases

The following V1 cases were selected because each illustrates a distinct failure mode that the multi-method scoring framework can localize.

**Case 1 — Generator hallucinates despite reasonable retrieval (`lg_wm4200_wm4000_q02`, V1).**
*Question:* "My washer will not drain or the OE error is showing. What should I do?"
*Manual reference:* "OE" denotes a *water Outlet error*; check the drain hose for kinks, clogs, and the 96-inch / 2.4-metre height limit.
*System answer:* "OE (Door Open Error)... ensure that the washer door is securely closed... unplug the washer and wait for 60 minutes..."
*RAGAS scores (V1):* Context Precision 0.64, Context Recall 0.67 (retrieval surfaced partially relevant chunks), Answer Relevancy 0.92, **Faithfulness 0.0**.
*Diagnosis:* The retriever returned chunks that mention drain-related symptoms but did not contain the literal "OE = water Outlet error" definition. The model nevertheless invented a meaning for "OE" inconsistent with those chunks (it confabulated "Door Open Error"). The Faithfulness score of 0.0 reflects the judge's verdict that no claim in the answer was supported by retrieved context — a clean signal of **generator-side hallucination**. Human review labelled this row *Wrong*.

**Case 2 — Wrong-topic retrieval and confident wrong answer (`lg_wm4200_wm4000_q03`, V1).**
*Question:* "My washer is shaking a lot and making loud banging sounds. What could be wrong?"
*Manual reference:* shipping bolts not removed, uneven load, leveling feet not flush, floor not rigid enough.
*System answer:* discusses "water hammer" — a plumbing-side noise caused by inlet-valve closure — and recommends installing water-hammer arrestors.
*RAGAS scores (V1):* Context Precision 0.83, Answer Relevancy 0.93, Faithfulness failed to parse, Context Recall failed to parse.
*Diagnosis:* The retrieved chunks were related to the question (Context Precision 0.83) but described a mechanically distinct phenomenon (inlet-valve plumbing) rather than the actual cause described in the manual (mechanical balance). The retriever could not distinguish "audibly similar but topically different" without a re-ranker. We hypothesize V2's cross-encoder re-ranker will improve performance on this class of question; the V2 results section confirms the hypothesis at the aggregate level.

**Case 3 — Compounding failure: imprecise retrieval AND ungrounded generation (`service_manual_18_q02`, V1).**
*Question:* "The AC is supposed to heat the room, but the air does not feel warm enough. What does the manual say to do?"
*Manual reference:* operate ≥15 min before evaluating; intake-vs-discharge differential should be > 14 °C; check airflow, filters, remote-control temperature.
*System answer:* describes the "+8/15 °C Heat" button (a freeze-protection feature), not heating troubleshooting.
*RAGAS scores (V1):* Context Precision 0.5, Answer Relevancy 0.15, Faithfulness failed to parse, Context Recall 1.0.
*Diagnosis:* Retrieval was mediocre (CP 0.5) — only some of the retrieved chunks were relevant — but the generator latched onto an unrelated section about the "+8/15 °C Heat" button. The very low Answer Relevancy (0.15) is the diagnostic signal that the *answer* doesn't address the *question* even though some retrieval was correct. This is a different failure mode from Case 1 (generator off-topic vs. generator hallucinating); the four-metric framework distinguishes them.

**Case 4 — Pre-fix: hallucinated citation (`service_manual_18_q01`).**
Before the citation-strip post-processor was added, Phi-3-mini emitted answers ending with `(db05a9.pdf, p. 12)` despite all retrieved chunks coming from pages 31–32 of the LG washing-machine manual. The model had learnt to perform the *form* of citation (parenthetical filename plus page number) without the substance. The deterministic regex strip in `rag_pipeline._strip_model_citations`, combined with the structured `sources` field built from retriever metadata, eliminated this class of error.

[REVIEW: pick the 2–3 of these you want to keep in the final paper. Cases 1, 2 and 3 together demonstrate that the four RAGAS metrics localise failures to retrieval vs. generation; Case 4 documents a problem we already solved.]

### 5.6 System Augmentation: Auto-Routing via a Learned Manual Classifier

The V1/V2/V3 experiment evaluates *answer quality given a known source manual*; the user is assumed to have already selected which appliance their question is about. In practice this is an awkward UX — a user typing *"my washer is leaking"* should not have to know which file in the corpus contains the answer. To address this we trained a small feed-forward classifier that predicts the source manual directly from the query, enabling an auto-routed `/ask_auto` endpoint that does not require the user to specify a `source`.

**Architecture.** The classifier is a feed-forward neural network trained end-to-end. It sits on top of a frozen `sentence-transformers/all-MiniLM-L6-v2` embedder; the 384-dimensional L2-normalized embedding is fed into a small trainable head:

```
Linear(384 → 128) → ReLU → Dropout(p=0.2) → Linear(128 → 5)
```

Output logits are passed through softmax to produce *P(manual | query)*. The head has roughly 50K trainable parameters; the embedder is frozen, so all backpropagation is restricted to the head. This is a deliberate **transfer-learning configuration** — we exploit the semantic structure that MiniLM has already learned from large-scale contrastive pretraining, and only train a small classification head from our own labeled data, which mitigates overfitting on the small training set (~100 augmented examples).

**Training procedure.** The training loop in `backend/train_classifier.py` implements the standard supervised-learning workflow:
- **Loss function:** categorical cross-entropy over the 5 classes.
- **Optimizer:** Adam (`lr = 1e-3`, `weight_decay = 1e-4` — the latter is L2 regularization on the head's weights, discouraging large coefficients).
- **Regularization:** dropout `p = 0.2` on the hidden layer at training time, disabled at inference.
- **Batch size:** 8 (mini-batch SGD via Adam).
- **Epochs:** 60 (training loss converges to ~0.009 by epoch 60, indicating the model has comfortably memorized the training distribution).
- **Backpropagation:** PyTorch autograd computes gradients of the cross-entropy loss with respect to the head's parameters at each step; the embedder's parameters are excluded from the optimizer.
- **Reproducibility:** fixed random seeds (`SEED=42`) for `random`, `numpy`, and `torch`.

**Training data.** The 25 hand-labeled questions in `eval/questions.json` are augmented by deterministic paraphrasing — synonym substitution on appliance terms (`washer ↔ washing machine`, `fridge ↔ refrigerator`, `aircon ↔ air conditioner ↔ AC`) and lexical rewrites on common question stems (`What should I check ↔ What can I check ↔ What should I look at`). Augmentation produces **103 examples** distributed roughly evenly across the five manual classes:

| Manual | Augmented examples |
|---|---|
| `DA68-04752Q_FDR_RF6500C_3Door_EN_MES_CFR_260209.pdf` | 17 |
| `Service-Manual-18.pdf` | 21 |
| `c06184015.pdf` | 21 |
| `cpd60205.pdf` | 21 |
| `db05a9.pdf` | 23 |

**Training.** 60 epochs, Adam (`lr=1e-3`, `weight_decay=1e-4`), cross-entropy loss, batch size 8, dropout 0.2. Final-epoch training loss converged to **0.009**, indicating the model has comfortably memorized the training distribution. Loss curves (per fold) are saved to `backend/manual_classifier_loss.png`.

**Evaluation — 5-fold cross-validation:**

| Fold | Validation accuracy |
|---|---|
| 1 | 1.000 |
| 2 | 1.000 |
| 3 | 1.000 |
| 4 | 0.950 |
| 5 | 0.950 |
| **Mean** | **0.980 ± 0.024** |

**Important caveat — paraphrase leakage across folds.** The 5-fold split was performed over the **augmented** dataset (103 paraphrases) rather than over the **25 original questions**. As a result, a paraphrase of question Q can land in the training fold while another paraphrase of Q lands in the validation fold; the classifier effectively sees near-duplicate inputs at evaluation time. The reported 0.980 should therefore be interpreted as an **upper bound** on out-of-distribution generalization. A more rigorous split — stratified by *original-question-id* such that all paraphrases of a given source question stay together in either train or val — would produce a more conservative estimate. We flag this as a methodological caveat rather than re-running with the stricter split given the project timeline; it is an obvious item for follow-up work.

**Integration.** Two new endpoints are exposed in `main.py`:

- `POST /classify` — accepts `{"query": str}`, returns `{predicted_source, confidence, distribution}`. No retrieval or generation is performed. Useful for the frontend to display *"Routing to db05a9.pdf (92% confidence)"* before the answer streams in.
- `POST /ask_auto` — accepts `{"query": str}`, internally calls `/classify` then `/ask` with the predicted source, and returns the standard `/ask` response augmented with `routed_to` and `routing_confidence`. End-to-end auto-routing in one HTTP call.

Both endpoints return HTTP 503 with a clear error message if `manual_classifier.pt` is not present on disk, so a partial deployment without the trained classifier degrades gracefully rather than crashing.

**Why this matters for the paper.** Sections 5.1–5.5 study a system whose retrieval is filtered to the right manual *given by the user*. §5.6 adds a learned component that closes that gap on the user side: a small but genuinely-trained neural network that converts an unfiltered natural-language query into a source filter. It is the project's most direct *deep learning* contribution — a real training loop, a real loss function, a real held-out evaluation — distinct from the V1/V2/V3 comparison study which uses pretrained components only.

---

## 6. Discussion

### 6.1 What Worked

**Source-filter as a multi-document fix.** A single pre-V2 query like *"how do I reset to factory defaults?"* in the unfiltered system pulled chunks from every manual that mentioned a reset, producing a Frankenstein answer. The simple `where={"source": filename}` filter on Chroma completely eliminated cross-manual contamination at zero retrieval-quality cost — Context Precision in V1 (0.86) and V2 (0.86) is comparable to single-corpus RAG systems despite our index spanning five distinct appliance manuals.

**Greedy decoding for factual QA.** Lowering the generator's temperature from 0.2 to 0 reduced visible fabrication on small-scale spot checks (notably eliminating most invented page numbers in answer text). For reference-lookup tasks where consistency matters more than creativity, the maximum-likelihood token at every step empirically produces better-grounded output.

**Deterministic source attribution.** Rather than asking the model to emit citations inline (which leads to fabricated page numbers — see §2.3 and Case 4 in §5.5), we attach the structured `sources` field built from retriever metadata and strip any model-emitted parenthetical citations via regex. The user always sees citations that are guaranteed accurate (filenames + pages come from Chroma metadata, not from the model's output). This is a small engineering choice that closes a real failure mode at no quality cost.

**The cross-encoder re-ranker meaningfully helps faithfulness.** V2's Faithfulness improvement (0.18 → 0.45) is the largest single metric change in our experiments. Combined with the small Context Precision change (0.86 → 0.86), the data suggests the re-ranker isn't surfacing *more* relevant chunks so much as it is suppressing *near-duplicates*, which gives the generator a more focused (less repetitive) context window.

**Post-experiment prompt-engineering iteration further reduced off-topic content.** After the V1/V2/V3 evaluation was complete, we observed that Phi-3-mini occasionally included tangentially related troubleshooting steps from retrieved chunks — for example, a fax-related "Error Correction Mode" setting in answers about paper jams, because the retrieved chunks for "what should I check first?" included sections of the manual that mentioned both jams and fax settings in adjacent prose. Adding an explicit *"answer only the specific question asked; do not include tangentially-related material from the context"* directive to the system prompt — with a concrete example — visibly eliminated this class of failure on subsequent queries (paper-jam answer length dropped from 5 steps to 3, with all 3 directly addressing the question). We did not re-run RAGAS evaluation since this change is post-hoc and only affects answer text (retrieval is unchanged); the qualitative improvement is documented as a system refinement rather than a measured-metric improvement. The observation suggests that even at the 3.8B-parameter scale, careful prompt engineering can address specific generator failure modes without architectural change or fine-tuning.

### 6.2 What Surprised Us

**The Qwen-vs-human scoring divergence is large and one-directional.** We expected some divergence between AI-assisted and human scoring; we did not expect 56% disagreement with **13 of 14 disagreements in the same direction** (Qwen too strict). This is meaningful enough that any RAG-evaluation pipeline using a small (≤3B) judge should be cross-checked against human review on at least a sample, not trusted as the sole signal.

**Faithfulness can be measured but not produced at the 3B-judge scale.** RAGAS's Faithfulness metric was the most informative of the four for diagnosing *generator-side* failures (see §5.2 case studies), and also the most fragile to compute: 14/25 successful samples in V1, 19/25 in V2, ~17/25 in V3, with the rest failing on JSON-parser issues from Qwen-3B's output. The metric we most wanted to trust is the one most prone to hardware-dependent variance.

**V2 retrieval improvements showed a recall–faithfulness trade-off, not a uniform improvement.** Going in, we expected V2 to lift everything modestly. Instead, Faithfulness jumped (+0.26) while Context Recall fell substantially (−0.24) and the other two metrics barely moved. Smaller chunks make the generator more honest at the cost of seeing less of the reference content. The right way to think about V2 is not "better than V1" but "different trade-off than V1."

**A larger swap of the generator (V3, Qwen-3B) made things worse, not better.** The V1 baseline analysis identified the generator as the bottleneck (Faithfulness 0.18 over partial samples, despite high retrieval quality). The natural follow-up is to swap the generator. We did so, and Answer Relevancy dropped by 0.22 and Faithfulness by 0.07. **Switching to a similarly-sized open model is not the same as fixing the bottleneck.** This is unintuitive but real, and an honest data point for the paper: at the 3-4B parameter scale on CPU, our generator choices are tightly clustered around the same quality ceiling. Breaking through that ceiling probably requires either a much larger model, fine-tuning, or smarter decoding.

**The smoke-test timeout / concurrency lesson.** A small but practical observation we documented in §6.3: under default `max_workers=2`, RAGAS's concurrent calls to a CPU-hosted 3B judge produced sustained `ReadTimeout` exceptions because the model is memory-bandwidth bound rather than compute bound. Reducing to `max_workers=1` (sequential) preserved the same total wall-clock time (because retries under concurrency wasted compute) and produced clean numbers. The cost of concurrency on commodity hardware was *negative* — concurrent execution made things worse, not faster.

### 6.3 Engineering Lessons

A 3B-parameter judge, when run on CPU with default `max_workers=2` concurrency, produces sustained read timeouts because the model is memory-bandwidth bound rather than compute bound. Reducing to `max_workers=1` (sequential) preserves total wall-clock time (because retries under concurrency wasted compute) and produces clean numbers. This is a small but practical observation worth recording for future RAG evaluation work on commodity hardware.

### 6.4 Threats to Validity

- **Small judge.** As discussed in §4.4, Qwen-2.5-3B as judge has known limitations: it over-penalises correct-but-condensed answers (vs. exhaustive reference answers) and cannot reliably emit parseable JSON for the Faithfulness claim-decomposition prompt, leading to partial sample coverage on that metric.
- **Reference-answer length.** Our reference answers are exhaustive; a judge that scores by "did the answer contain *every* fact in the reference" will systematically under-score correct-but-condensed answers.
- **Bundled V2 changes.** V2 changes three retrieval components simultaneously (embedder, chunk size, re-ranker). The aggregate V1→V2 movement therefore cannot be attributed to any one of those changes; an ablation study (V2 minus reranker, V2 minus chunk-size change, etc.) would be required.
- **Judge-hardware variance.** The original V1 RAGAS pass ran with the Qwen judge on CPU; the V1/V2/V3 comparison reported here used the Qwen judge on CUDA via Ollama. Even with `temperature=0`, the judge produced systematically different per-row Faithfulness scores between hardware configurations, with the V1 mean shifting by ~0.3 points (CPU: F=0.49 over 11 samples; CUDA: F=0.18 over 14 samples). This implies that RAGAS-with-a-3B-judge is **not bit-deterministic across hardware**, and reported metric values are paired implicitly with hardware/version metadata. The cross-variant comparison reported in §5.1–§5.4 holds judge-hardware constant (all CUDA), but a different replicator running the same code on different hardware should expect somewhat different numerical scores.
- **Generator-judge model overlap in V3.** V3's generator (Qwen-2.5-3B) and the RAGAS judge (Qwen-2.5-3B) are the *same* model. There is at least theoretical room for stylistic-bias contamination of the V2-vs-V3 comparison, even though we did not observe an obvious self-favouring bias (V3 scored *worse*, not better). A genuinely independent judge (GPT-4-class via API, Llama-3.1-8B run separately) would close this concern.
- **Eval set size.** 25 questions across 5 manuals is small. Variance per metric is high relative to between-variant differences. We did not compute confidence intervals or run paired statistical tests on per-question metric pairs (e.g. paired *t*-test V1 vs V2 Faithfulness over the rows successfully scored in both); doing so is straightforward future work and would let us state which observed deltas exceed judge noise.
- **Manual-classifier evaluation leakage.** As discussed in §5.6, the manual classifier's 0.98 cross-validation accuracy is computed over augmented paraphrases rather than over original-question-IDs, so paraphrases of the same source question can land in different folds. The reported accuracy is an upper bound; out-of-distribution generalisation has not been measured.

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

- **Scanned PDFs are not supported.** `pypdf` extracts only embedded text; image-only manuals would require OCR (e.g. `ocrmypdf` or `pytesseract`).
- **No conversational memory.** Each `/ask` call is independent; follow-up questions cannot reference the previous turn.
- **CPU-only inference is slow.** Phi-3-mini takes several seconds per answer; RAGAS evaluation on 25 questions runs for hours.
- **Character-based chunking can split mid-token.** Snippet truncation visibly cuts words mid-character, observable as fragments like *"s unplugged. • Make sure..."* in retrieved sources. Token-aware chunking would address this.
- **Answer quality is bounded by the 3.8B-parameter generator.** Phi-3-mini is small; a stronger generator (Phi-3-medium, Llama-3.1-8B, or a hosted API) would likely improve answer quality at higher cost or latency.

### 7.2 Future Extensions

- A token-based chunker.
- A learned re-ranker fine-tuned on the manual corpus.
- LoRA fine-tuning of Phi-3-mini on synthesized Q-A pairs from the manuals.
- Streaming responses for better perceived latency in the UI.
- A hybrid retriever combining BM25 keyword search with dense embeddings.
- Conversational memory with a sliding-window history.

---

## 8. Contributions

**[REVIEW WITH PARTNER — adjust as appropriate.]**

**Shaina Talisay.** Initial backend implementation including the FastAPI HTTP layer, ChromaDB integration, and the Phi-3-mini generator pipeline. Source-filter on `/ask` and the citation-strip post-processor. Human review of all 25 evaluation rows.

**Duranne B. Duran.** 25-question evaluation set with manually verified reference answers. `run_eval.py` harness for batch RAG queries. `score_results.py` AI-assisted Correct/Partial/Wrong/Refused scoring pipeline using Ollama+Qwen. RAGAS evaluation infrastructure (`run_ragas.py`). Vite + React frontend chat UI — model dropdown, sources panel, and the greeting/thank-you/follow-up routing heuristics.

**Testing and verification.** Automated test suites (`backend/tests/` via pytest, `frontend/src/intent.test.js` via Vitest, and the eval-set validator) described in §3.5. **[CONFIRM authorship/attribution for the frontend and tests with your team before submitting.]**

**Joint.** Evaluation methodology design, results analysis, and report.

**Deep-learning component breakdown.** The project's RAG inference pipeline composes pretrained transformer-based deep neural networks: encoder transformers for embedding (MiniLM and bge), a cross-encoder transformer for re-ranking (MS-MARCO MiniLM), and decoder-only transformers for generation (Phi-3-mini, Qwen-2.5-3B). The manual classifier in §5.6 is the one component trained end-to-end by us; its training demonstrates the full supervised deep-learning workflow — feed-forward architecture design, ReLU non-linearity, dropout regularization, transfer learning via frozen embeddings, cross-entropy loss, Adam optimization with L2 weight decay, and 5-fold cross-validation. The project does *not* include fine-tuning of the larger pretrained components (Phi-3, the embedders, or the cross-encoder); we list those as natural extensions in §7.2.

---

## 9. References

**[REVIEW — add proper citations for each. Format per your course style guide.]**

- Lewis et al., 2020. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* arXiv:2005.11401.
- Reimers and Gurevych, 2019. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.
- Microsoft Research, 2024. *Phi-3 Technical Report.*
- Es et al., 2023. *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217.
- ChromaDB, sentence-transformers, FastAPI, Hugging Face Transformers, Ollama — software documentation.

---

## Appendix A — Reproducibility

- Repository: `https://github.com/shaittoo/manual-rag-chatbot`
- Commit hash for results in this report: **[FILL IN: commit hash from `git log` of the run]**
- Python: 3.11.9
- All Python dependencies pinned in `backend/requirements.txt`
- Manual PDFs are gitignored due to copyright; reproducing the exact eval requires identical PDF copies. Filenames and SHA-256 hashes are listed in `eval/manuals_inventory.txt` **[OPTIONAL: produce this file if your professor wants reproducibility.]**
- Run the test suites: `cd backend && pytest` (backend logic + eval-set validation; no model download required) and `cd frontend && npm test` (frontend intent heuristics via Vitest).

## Appendix B — Eval set summary statistics

**[FILL IN: question type distribution, manual coverage, average reference-answer length. Pull from `questions.json`.]**

## Appendix C — Full RAGAS configuration

- Judge model: `qwen2.5:3b` via Ollama at `http://localhost:11434`
- Embedding model (for ResponseRelevancy): `sentence-transformers/all-MiniLM-L6-v2`
- Run config: `max_workers=1`, `timeout=600s`, `max_retries=3`
- Metrics: `Faithfulness`, `ResponseRelevancy`, `LLMContextPrecisionWithReference`, `LLMContextRecall`
- Source: `backend/run_ragas.py`
