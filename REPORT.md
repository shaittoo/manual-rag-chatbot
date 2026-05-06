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

**[FILL IN AFTER V1/V2/V3]** *(write this last — 1 paragraph, ~150 words. Should answer: what problem, what we built, what we measured, what we found, what it means.)*

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

**Generator.** `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters) loaded via Hugging Face `transformers`. The system prompt constrains the model to answer only from retrieved context, refuse if the context does not support an answer, and **not** emit inline citations. Decoding is greedy (`temperature=0.0`) to minimize fabrication.

**Pipeline orchestrator (`rag_pipeline.ask`).** Composes the three components: retrieve → generate → strip any leftover citations → return a structured `Answer` containing the cleaned text and a list of source records.

**HTTP API (`main.py`).** FastAPI exposing four endpoints:
- `GET /health` — liveness check
- `GET /sources` — list of indexed filenames (for frontend dropdown)
- `POST /ingest` — wipe and rebuild the Chroma collection from `manuals/`
- `POST /ask` — RAG query, optionally scoped to a single source

### 2.3 Notable Design Decisions

**Source-filter on `/ask`.** Without filtering, a query like *"how do I reset to factory defaults?"* retrieves chunks from every manual that mentions a reset, and the generator hallucinates a Frankenstein procedure. A user-supplied `source` parameter restricts retrieval to one manual, which is the cheapest way to handle a multi-appliance corpus. (For an automatic version, a manual-classifier stage could be added.)

**Citation strip post-processing.** Despite being instructed *not* to cite inline, Phi-3-mini repeatedly emitted parenthetical references like `(db05a9.pdf, p. 12)` — and the page numbers were fabricated. In one observed case the model claimed evidence from page 12 when all retrieved chunks were from pages 31–32. Rather than continually re-tuning the prompt, we apply a regex post-filter (`rag_pipeline._strip_model_citations`) that removes any matching pattern. The structured `sources` field returned alongside the answer is built from retriever metadata, not the model's output, so cited sources are guaranteed accurate.

**Greedy decoding (temperature=0).** Lowered from an initial `0.2` after observing that low-but-nonzero temperatures produced occasional fabricated facts. For factual lookup the most-likely token at each step is empirically more grounded.

---

## 3. Implementation

### 3.1 Code Organization

```
manu/
├── backend/
│   ├── main.py              # FastAPI app + routes
│   ├── rag_pipeline.py      # Orchestrates retrieve + generate
│   ├── embedder.py          # sentence-transformers wrapper (MiniLM)
│   ├── retriever.py         # PDF parsing, chunking, ChromaDB I/O
│   ├── generator.py         # Phi-3-mini-4k-instruct via Transformers
│   ├── run_eval.py          # Hits /ask for each question, logs results
│   ├── score_results.py     # AI-assisted scoring via Ollama+Qwen
│   ├── run_ragas.py         # RAGAS evaluation
│   ├── requirements.txt
│   └── manuals/             # PDFs (gitignored)
└── eval/
    ├── questions.json       # 25 questions with reference answers
    ├── results.csv          # raw /ask outputs
    ├── results_scored.csv   # Qwen judge labels
    ├── human_scored.csv     # human review labels
    └── ragas_v1.csv         # RAGAS scores (per variant)
```

### 3.2 Stack

- **FastAPI** + **Uvicorn** — HTTP API
- **ChromaDB (PersistentClient)** — local vector store
- **sentence-transformers** — embedding model
- **Hugging Face Transformers** + **PyTorch** — Phi-3-mini inference
- **pypdf** — PDF text extraction
- **RAGAS** — evaluation framework
- **Ollama** + **Qwen-2.5-3B** — local LLM judge for scoring
- **LangChain (Ollama / HuggingFace adapters)** — bridging Ollama and RAGAS

### 3.3 Hardware

All experiments were run on **CPU-only** Windows 11 hardware. Across the 25-question V1 baseline, Phi-3-mini fp32 on CPU produced an answer in **~166 s on average** (range: 76 s to 204 s; the variance is driven primarily by output length, since greedy decoding is token-by-token). RAGAS scoring with Qwen-2.5-3B as judge averaged **~349 s per metric per question** under sequential (`max_workers=1`) execution; the full 25-question × 4-metric V1 run took **~9.7 hours** of wall-clock time.

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

**Faithfulness-prompt parser failures.** RAGAS's Faithfulness metric requires the judge to (a) decompose the answer into atomic claims as a JSON list and (b) verdict each claim against the retrieved context. Qwen-2.5-3B cannot reliably emit parseable JSON for this prompt: in the V1 full run, **only 11 of 25 samples (44%)** produced a usable Faithfulness score; the remaining 14 samples raised either `RagasOutputParserException` (parser exhausted retries) or `AttributeError('StringIO' object has no attribute 'statements')` (RAGAS internal fallback after parser failure). The other three RAGAS metrics scored reliably (Answer Relevancy 25/25, Context Precision 25/25, Context Recall 22/25). Our reported Faithfulness mean is therefore computed over a partial sample and should be read as directional; a stronger judge (Qwen-2.5-7B, Llama-3.1-8B, or GPT-4-class) would be required to obtain a reliable Faithfulness score for every row.

**Mitigation.** We report all three scoring methods (Qwen labels, human labels, RAGAS metrics) for V1 and discuss their divergence as a finding rather than collapsing to a single number. The methodological observation — that LLM-as-judge under a 3B-parameter model is systematically biased and partially unreliable — is itself one of the contributions of this work.

---

## 5. Results

### 5.1 Variant Comparison

The master comparison table aggregates all three scoring methods across the three pipeline variants. V2 and V3 columns are filled in as their experiments complete.

| Variant | Faithfulness (n) | Answer Relevancy | Context Precision | Context Recall (n) | Human Strict | Human Lenient | Mean Latency (s) |
|---|---|---|---|---|---|---|---|
| V1 baseline | 0.494 (n=11) | 0.756 | 0.821 | 0.870 (n=22) | 28% | 50% | 165.8 |
| V2 retrieval | *[FILL IN AFTER V2]* | — | — | — | — | — | — |
| V3 generator | *[FILL IN AFTER V3]* | — | — | — | — | — | — |

The `(n=...)` annotation on Faithfulness and Context Recall denotes the number of samples for which that metric was successfully computed; the mean is taken over those samples only. Faithfulness coverage (11/25) is constrained by the 3B-judge's structured-output failures discussed in §4.4. Answer Relevancy and Context Precision were computed for all 25 samples in V1.

### 5.2 V1 Baseline Detail

**Top-line scoring across three methods.** The Qwen judge labelled the baseline at **4% strict / 26% lenient accuracy**; human review labelled it at **28% strict / 50% lenient accuracy** — a 24-percentage-point gap on both measures, driven by the over-strictness pattern identified in §4.4. RAGAS produced four mean scores (computed over successfully-scored samples):

| Metric | V1 Mean | Samples | What it measures |
|---|---|---|---|
| Faithfulness | 0.494 | 11/25 | Are the answer's claims supported by the retrieved chunks? |
| Answer Relevancy | 0.756 | 25/25 | Does the answer address the question? |
| Context Precision (with reference) | 0.821 | 25/25 | Are the retrieved chunks relevant given the reference answer? |
| Context Recall | 0.870 | 22/25 | Did retrieval find the information needed for the reference answer? |

**Diagnostic interpretation.** The four RAGAS metrics partition the failure surface differently from the single Correct/Partial/Wrong/Refused label. Specifically, the gap between the two retrieval metrics (Context Precision 0.82, Context Recall 0.87) and Faithfulness (0.49 over 11 samples) is informative: **retrieval is finding the right chunks most of the time, but the generator does not always remain grounded in them**. This pattern is also visible at the individual sample level. For `lg_wm4200_wm4000_q02` ("My washer will not drain or the OE error is showing"), Context Precision was 0.92 and Context Recall was 1.0 — i.e. the retriever returned the correct chunks. Phi-3-mini nevertheless answered with a fabricated definition ("OE = Door Open Error"); the manual itself defines OE as the *water Outlet error*. Faithfulness for this row failed to parse and was therefore not scored, but the row would have scored near zero on Faithfulness if it had completed. The takeaway: **for V1, retrieval is operating well above the generator-faithfulness floor, suggesting that downstream improvements should target the generation stage at least as much as the retrieval stage**. We test this hypothesis directly by comparing V2 (retrieval-only changes, same generator) against V3 (same retrieval as V2, different generator) — see §5.3 and §5.4.

**Latency.** Mean wall-clock latency for a single `/ask` call was 165.8 s, with a range of 76 s to 204 s across the 25 questions. Variance is dominated by output length, since greedy decoding is token-by-token on CPU.

**Per-method headline accuracy.** [REVIEW: this paragraph reframes the same data three different ways for the reader.] Reading the same baseline through three lenses, we obtain three quite different statements: "the system answers 4% of questions correctly" (Qwen judge); "the system answers 28% of questions correctly" (human review); "the system retrieves with 0.82 precision and 0.87 recall but its generator stays grounded in the retrieved context only ~49% of the time on the rows we could measure" (RAGAS). Each statement is internally consistent with its method; their divergence is the methodological finding of §4.4.

### 5.3 V1 → V2: Effect of Retrieval Improvements

**[FILL IN AFTER V2]**

### 5.4 V2 → V3: Effect of Generator Choice

**[FILL IN AFTER V3]**

### 5.5 Specific Failure Cases

The following V1 cases were selected because each illustrates a distinct failure mode that the multi-method scoring framework can localize.

**Case 1 — Generator hallucinates despite correct retrieval (`lg_wm4200_wm4000_q02`).**
*Question:* "My washer will not drain or the OE error is showing. What should I do?"
*Manual reference:* "OE" denotes a *water Outlet error*; check the drain hose for kinks, clogs, and the 96-inch / 2.4-metre height limit.
*System answer:* "OE (Door Open Error)... ensure that the washer door is securely closed... unplug the washer and wait for 60 minutes..."
*RAGAS scores:* Context Precision 0.92, Context Recall 1.0 (retrieval was strong); Faithfulness failed to parse.
*Diagnosis:* The retriever returned the correct chunks; the model nevertheless invented a meaning for "OE" inconsistent with those chunks. This is **a generation-side failure on top of correct retrieval** — exactly the case where Faithfulness is the diagnostic signal we needed and was lost to parser failure. Human review labelled this row *Wrong*.

**Case 2 — Wrong-topic retrieval (`lg_wm4200_wm4000_q03`).**
*Question:* "My washer is shaking a lot and making loud banging sounds. What could be wrong?"
*Manual reference:* shipping bolts not removed, uneven load, leveling feet not flush, floor not rigid enough.
*System answer:* discusses "water hammer" — a plumbing-side noise caused by inlet-valve closure — and recommends installing water-hammer arrestors.
*RAGAS scores:* Context Precision 0.83, Context Recall 1.0, Answer Relevancy 0.95; Faithfulness failed to parse.
*Diagnosis:* Counterintuitively, the retrieval metrics are high here too: Context Precision 0.83 indicates the retrieved chunks are *related to the question*, but they describe a mechanically distinct phenomenon (inlet-valve plumbing) rather than the actual cause described in the manual (mechanical balance). The retriever cannot distinguish "audibly similar but topically different" without a re-ranker. We hypothesize V2's cross-encoder re-ranker will improve performance on this class of question.

**Case 3 — Compounding failure: bad retrieval AND bad generation (`service_manual_18_q02`).**
*Question:* "The AC is supposed to heat the room, but the air does not feel warm enough. What does the manual say to do?"
*Manual reference:* operate ≥15 min before evaluating; intake-vs-discharge differential should be > 14 °C; check airflow, filters, remote-control temperature.
*System answer:* describes the "+8/15 °C Heat" button (a freeze-protection feature), not heating troubleshooting.
*RAGAS scores:* Context Precision 0.0, Context Recall 0.33, Faithfulness 0.0.
*Diagnosis:* Both retrieval and generation failed. Retrieval brought back chunks about a separate feature; the generator dutifully grounded its answer in those (irrelevant) chunks. **Faithfulness scored 0.0 not because the answer was unfaithful to the retrieval but because the retrieval was unfaithful to the question** — an instructive limitation of metric-level interpretation.

**Case 4 — Pre-fix: hallucinated citation (`service_manual_18_q01`).**
Before the citation-strip post-processor was added, Phi-3-mini emitted answers ending with `(db05a9.pdf, p. 12)` despite all retrieved chunks coming from pages 31–32 of the LG washing-machine manual. The model had learnt to perform the *form* of citation (parenthetical filename plus page number) without the substance. The deterministic regex strip in `rag_pipeline._strip_model_citations`, combined with the structured `sources` field built from retriever metadata, eliminated this class of error.

[REVIEW: pick the 2–3 of these you want to keep in the final paper. Cases 1, 2 and 3 together demonstrate that the four RAGAS metrics localise failures to retrieval vs. generation; Case 4 documents a problem we already solved.]

---

## 6. Discussion

### 6.1 What Worked

**[FILL IN AFTER V1/V2/V3]** *(themes to consider: source-filter as a multi-document fix; greedy decoding for factual QA; deterministic source attribution.)*

### 6.2 What Surprised Us

**[FILL IN AFTER V1/V2/V3]** *(themes to consider: the citation-hallucination story; the 56% Qwen/human disagreement; the smoke-test timeout failures and concurrency lesson; the Faithfulness parser failures.)*

### 6.3 Engineering Lessons

A 3B-parameter judge, when run on CPU with default `max_workers=2` concurrency, produces sustained read timeouts because the model is memory-bandwidth bound rather than compute bound. Reducing to `max_workers=1` (sequential) preserves total wall-clock time (because retries under concurrency wasted compute) and produces clean numbers. This is a small but practical observation worth recording for future RAG evaluation work on commodity hardware.

### 6.4 Threats to Validity

- **Small judge.** As discussed in §4.4, Qwen-2.5-3B as judge has known limitations.
- **Reference-answer length.** Our reference answers are exhaustive; a judge that scores by "did the answer contain *every* fact in the reference" will systematically under-score correct-but-condensed answers.
- **Bundled V2 changes.** We change three retrieval components simultaneously; an ablation would be needed to attribute V2 gains.
- **Eval set size.** 25 questions across 5 manuals is small. Variance per metric is high relative to between-variant differences.

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

**Duranne B. Duran.** 25-question evaluation set with manually verified reference answers. `run_eval.py` harness for batch RAG queries. `score_results.py` AI-assisted Correct/Partial/Wrong/Refused scoring pipeline using Ollama+Qwen. RAGAS evaluation infrastructure (`run_ragas.py`).

**Joint.** Evaluation methodology design, results analysis, and report.

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

## Appendix B — Eval set summary statistics

**[FILL IN: question type distribution, manual coverage, average reference-answer length. Pull from `questions.json`.]**

## Appendix C — Full RAGAS configuration

- Judge model: `qwen2.5:3b` via Ollama at `http://localhost:11434`
- Embedding model (for ResponseRelevancy): `sentence-transformers/all-MiniLM-L6-v2`
- Run config: `max_workers=1`, `timeout=600s`, `max_retries=3`
- Metrics: `Faithfulness`, `ResponseRelevancy`, `LLMContextPrecisionWithReference`, `LLMContextRecall`
- Source: `backend/run_ragas.py`
