"""
run_ragas.py
------------
RAGAS evaluation for the manuals RAG system.

Reads:
    eval/results.csv          (produced by run_eval.py)
    eval/questions.json       (the 25-question set; for ground-truth references)

Writes:
    eval/ragas_<tag>.csv          per-question RAGAS scores joined onto results
    eval/ragas_<tag>_summary.json mean scores + run config

Why this layout:
    run_eval.py is the *RAG runner* — it hits the FastAPI server and logs
    answers + retrieved contexts. This script is the *judge* — it scores those
    logged outputs offline. Keeping them separate means we can re-score without
    re-paying the generation cost, and we can score V1, V2, V3 with the
    same judge config for an apples-to-apples comparison.

Judge model: qwen2.5:3b via Ollama (local, no API cost).

CAVEAT (be honest in your report):
    RAGAS was designed assuming a strong judge LLM (GPT-4 class). A 3B model
    is small. Expect the judge to be noisy, especially on `faithfulness` which
    requires careful claim decomposition. Treat absolute scores with skepticism;
    treat *relative* differences across V1/V2/V3 as more meaningful — both
    versions are judged by the same noisy judge, so the noise partially cancels.
    For your written report: spot-check 5-10 RAGAS verdicts against your own
    judgment to estimate judge accuracy. If qwen2.5:3b is wildly off, drop it
    in the report and rely on the human-scored CSV instead.

Run:
    cd backend
    # Make sure Ollama is running and qwen2.5:3b is pulled:
    #   ollama list   -> should show qwen2.5:3b
    python run_ragas.py --tag v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# RAGAS + LangChain wrappers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS_PATH = EVAL_DIR / "results.csv"
QUESTIONS_PATH = EVAL_DIR / "questions.json"

# Judge LLM via Ollama. qwen2.5:3b is a CPU-runnable 3B model.
JUDGE_MODEL = "qwen2.5:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Embeddings for ResponseRelevancy (same family as the retriever embedder).
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Be patient — a 3B model on CPU is slow.
# Bumped from 300 → 600 after smoke test had 3/8 jobs hit ReadTimeout
# on Faithfulness (which decomposes claims and needs longer responses).
LLM_TIMEOUT_SECONDS = 600

# RAGAS run config. Keep concurrency low so we don't swamp Ollama.
# Dropped from 2 → 1 (fully sequential): on CPU a 3B model is memory-bandwidth
# bound, so two parallel requests double per-call latency and push us over
# the timeout ceiling. Sequential = same total wall time but no timeouts.
MAX_WORKERS = 1
RUN_TIMEOUT_SECONDS = 60 * 60 * 4  # 4h ceiling for the whole eval


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

@dataclass
class EvalRow:
    """One sample fed to RAGAS, plus pass-through metadata for the output CSV."""
    qid: str
    manual_filename: str
    qtype: str
    question: str
    reference: str          # gold answer from questions.json
    response: str           # actual_answer from results.csv
    contexts: list[str]     # retrieved chunks (we use the snippet field)
    latency_ms: float
    status_code: int
    error: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def _load_questions(path: Path) -> dict[str, dict]:
    """Index questions.json by id for fast reference-answer lookup."""
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    return {q["id"]: q for q in items}


def _parse_sources(raw: str) -> list[str]:
    """
    The `sources` column is a JSON-encoded list of {source, page, score, snippet}.
    For RAGAS we just need the chunk text — use the snippet.

    Note: snippets are truncated to ~600 chars (see rag_pipeline._to_source).
    That's enough text for the judge to assess relevance, but it does mean
    `faithfulness` could be a slight underestimate if the supporting fact lives
    in the truncated tail. This is consistent across V1/V2/V3, so the
    comparison stands.
    """
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out: list[str] = []
    for item in items:
        snippet = (item.get("snippet") or "").strip()
        if snippet:
            # Prefix with [source, page] so the judge can see the provenance,
            # mirroring how the generator sees the context.
            header = f"[{item.get('source', '?')}, p. {item.get('page', '?')}]"
            out.append(f"{header}\n{snippet}")
    return out


def _read_results_csv(path: Path) -> list[dict[str, str]]:
    """Plain-stdlib CSV read so we don't add a pandas dependency for this."""
    import csv
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_eval_rows(
    results_path: Path,
    questions_path: Path,
) -> list[EvalRow]:
    questions = _load_questions(questions_path)
    rows = _read_results_csv(results_path)

    out: list[EvalRow] = []
    skipped: list[tuple[str, str]] = []
    for r in rows:
        qid = r.get("id", "")
        gold = questions.get(qid)
        if gold is None:
            skipped.append((qid, "no matching question id in questions.json"))
            continue

        actual = (r.get("actual_answer") or "").strip()
        status = int(r.get("status_code") or 0)
        err = (r.get("error") or "").strip()
        if not actual or status != 200 or err:
            skipped.append((qid, f"bad row (status={status}, err={err[:80]})"))
            continue

        contexts = _parse_sources(r.get("sources", ""))
        if not contexts:
            skipped.append((qid, "no retrieved contexts in results.csv"))
            continue

        out.append(EvalRow(
            qid=qid,
            manual_filename=r.get("manual_filename", ""),
            qtype=r.get("type", ""),
            question=(r.get("question") or "").strip(),
            reference=(gold.get("reference_answer") or "").strip(),
            response=actual,
            contexts=contexts,
            latency_ms=float(r.get("latency_ms") or 0.0),
            status_code=status,
        ))

    if skipped:
        print(f"[load] skipped {len(skipped)} row(s):")
        for qid, why in skipped:
            print(f"   - {qid}: {why}")
    print(f"[load] kept {len(out)} sample(s) for RAGAS scoring")
    return out


# ---------------------------------------------------------------------
# RAGAS SETUP
# ---------------------------------------------------------------------

def _build_judge():
    """
    ChatOllama -> LangchainLLMWrapper. RAGAS calls `.generate()` etc., which
    the wrapper translates into LangChain chat completions.

    temperature=0.0 because we want deterministic judging. RAGAS prompts already
    include exemplars; sampling adds noise we don't want in evaluation.
    """
    llm = ChatOllama(
        model=JUDGE_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        num_predict=384,           # judge responses are short JSON; 512 was generous
        # ChatOllama doesn't take a top-level timeout; pass it through to httpx.
        client_kwargs={"timeout": LLM_TIMEOUT_SECONDS},
    )
    return LangchainLLMWrapper(llm)


def _build_embeddings():
    """
    HuggingFace MiniLM, same family as the retriever's embedder.
    Used by ResponseRelevancy (it embeds question + generated questions).
    """
    hf = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(hf)


def _to_eval_dataset(rows: list[EvalRow]) -> EvaluationDataset:
    samples = []
    for r in rows:
        samples.append({
            "user_input": r.question,
            "retrieved_contexts": r.contexts,
            "response": r.response,
            "reference": r.reference,
        })
    return EvaluationDataset.from_list(samples)


# ---------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------

def _write_per_question_csv(
    rows: list[EvalRow],
    scored_df,                   # pandas DataFrame returned by EvaluationResult.to_pandas()
    out_path: Path,
) -> None:
    """
    Join RAGAS scores back onto the original rows so the CSV has both the
    question metadata and the metric scores in one place.

    We write with stdlib csv to avoid a hard pandas dep at import time.
    """
    import csv

    metric_cols = [c for c in scored_df.columns
                   if c not in {"user_input", "retrieved_contexts",
                                "response", "reference"}]

    fieldnames = [
        "id", "manual_filename", "type", "question",
        "latency_ms", "status_code",
    ] + metric_cols

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r, (_, scored) in zip(rows, scored_df.iterrows()):
            row = {
                "id": r.qid,
                "manual_filename": r.manual_filename,
                "type": r.qtype,
                "question": r.question,
                "latency_ms": round(r.latency_ms, 2),
                "status_code": r.status_code,
            }
            for m in metric_cols:
                v = scored[m]
                # NaN-safe: leave empty when the metric couldn't score the row.
                if v is None or (isinstance(v, float) and (v != v)):
                    row[m] = ""
                else:
                    row[m] = round(float(v), 4)
            w.writerow(row)


def _summary(scored_df, tag: str, sample_count: int, elapsed_s: float) -> dict:
    metric_cols = [c for c in scored_df.columns
                   if c not in {"user_input", "retrieved_contexts",
                                "response", "reference"}]
    means: dict[str, float | None] = {}
    for m in metric_cols:
        col = scored_df[m].dropna()
        means[m] = round(float(col.mean()), 4) if len(col) else None
    return {
        "tag": tag,
        "sample_count": sample_count,
        "elapsed_seconds": round(elapsed_s, 1),
        "judge_model": JUDGE_MODEL,
        "embed_model": EMBED_MODEL,
        "metrics": means,
        "metric_columns": metric_cols,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        default="v1",
        help="Output suffix: writes ragas_<tag>.csv and ragas_<tag>_summary.json",
    )
    parser.add_argument(
        "--results",
        default=str(RESULTS_PATH),
        help="Path to results.csv (default: eval/results.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only score the first N samples (handy for a smoke test)",
    )
    args = parser.parse_args()

    results_path = Path(args.results).resolve()
    if not results_path.exists():
        print(f"ERROR: results file not found: {results_path}", file=sys.stderr)
        return 2
    if not QUESTIONS_PATH.exists():
        print(f"ERROR: questions.json not found: {QUESTIONS_PATH}", file=sys.stderr)
        return 2

    rows = load_eval_rows(results_path, QUESTIONS_PATH)
    if args.limit is not None:
        rows = rows[: args.limit]
        print(f"[load] limited to first {len(rows)} sample(s) for smoke test")
    if not rows:
        print("ERROR: no usable rows after filtering. Aborting.", file=sys.stderr)
        return 2

    print(f"[setup] judge: {JUDGE_MODEL} via {OLLAMA_BASE_URL}")
    print(f"[setup] embed: {EMBED_MODEL}")
    judge = _build_judge()
    embeddings = _build_embeddings()

    # The four canonical RAGAS metrics for retrieval-augmented QA.
    metrics = [
        Faithfulness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=embeddings),
        LLMContextPrecisionWithReference(llm=judge),
        LLMContextRecall(llm=judge),
    ]
    print(f"[setup] metrics: {[m.name for m in metrics]}")

    dataset = _to_eval_dataset(rows)

    run_config = RunConfig(
        timeout=RUN_TIMEOUT_SECONDS,
        max_workers=MAX_WORKERS,
        max_retries=3,
        max_wait=60,
    )

    print(f"[ragas] scoring {len(rows)} sample(s) — go grab a coffee")
    t0 = time.perf_counter()
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,   # one bad sample shouldn't kill the run
        show_progress=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"[ragas] done in {elapsed:.1f}s")

    scored_df = result.to_pandas()

    out_csv = EVAL_DIR / f"ragas_{args.tag}.csv"
    out_json = EVAL_DIR / f"ragas_{args.tag}_summary.json"

    _write_per_question_csv(rows, scored_df, out_csv)
    summary = _summary(scored_df, tag=args.tag,
                       sample_count=len(rows), elapsed_s=elapsed)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"Wrote per-question scores -> {out_csv}")
    print(f"Wrote summary             -> {out_json}")
    print("=" * 60)
    print("Mean scores:")
    for m, v in summary["metrics"].items():
        v_str = f"{v:.4f}" if isinstance(v, float) else "n/a"
        print(f"  {m:40s} {v_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
