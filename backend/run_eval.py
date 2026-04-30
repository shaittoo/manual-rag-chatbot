"""
run_eval.py
-----------
Runs the manual RAG evaluation set in batches.

Expected input:
    eval/questions.json

Output:
    eval/results.csv

Each question should follow this schema:
{
  "id": "...",
  "manual_filename": "...",
  "question": "...",
  "reference_answer": "...",
  "type": "factual" | "procedural" | "tricky"
}

Run:
    cd backend
    python run_eval.py

Make sure FastAPI is already running:
    python -m uvicorn main:app --port 8000

Recommended:
    Run uvicorn WITHOUT --reload for long eval runs.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"
if not QUESTIONS_PATH.exists():
    QUESTIONS_PATH = BACKEND_DIR / "eval" / "questions.json"

RESULTS_DIR = PROJECT_ROOT / "eval"
RESULTS_PATH = RESULTS_DIR / "results.csv"

ASK_URL = "http://localhost:8000/ask"

# Retrieval setting
TOP_K = 4

# Timeout per question, in seconds.
# Phi-3 can be slow on laptop GPUs/CPU, so keep this generous.
REQUEST_TIMEOUT = 600

# Batch range.
# Python slicing means:
#   START_INDEX inclusive
#   END_INDEX exclusive
#
# Examples:
#   0, 5    -> questions 1 to 5
#   5, 10   -> questions 6 to 10
#   10, 15  -> questions 11 to 15
#   15, 20  -> questions 16 to 20
#   20, 25  -> questions 21 to 25
START_INDEX = 20
END_INDEX = 25

# If True, append to existing results.csv.
# Keep this True when running batches.
APPEND_RESULTS = True


# ---------------------------------------------------------------------
# HTTP CALL
# ---------------------------------------------------------------------

def post_ask(
    query: str,
    source: str | None = None,
    top_k: int = TOP_K,
) -> tuple[dict[str, Any], int, float]:
    """
    Calls POST /ask and returns:
        response_json, status_code, latency_ms

    status_code = 0 means the request failed before receiving an HTTP response,
    usually timeout, connection refused, or server crash.
    """
    payload = {
        "query": query,
        "top_k": top_k,
        "source": source,
    }

    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        ASK_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            latency_ms = (time.perf_counter() - start) * 1000
            status_code = response.status
            data = json.loads(response.read().decode("utf-8"))
            return data, status_code, latency_ms

    except urllib.error.HTTPError as e:
        latency_ms = (time.perf_counter() - start) * 1000
        error_body = e.read().decode("utf-8", errors="replace")
        return {"error": error_body}, e.code, latency_ms

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return {"error": str(e)}, 0, latency_ms


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"questions.json not found at {path}. "
            "Create eval/questions.json first."
        )

    with path.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a JSON array.")

    return questions


# ---------------------------------------------------------------------
# CSV HELPERS
# ---------------------------------------------------------------------

FIELDNAMES = [
    "id",
    "manual_filename",
    "type",
    "question",
    "expected",
    "actual_answer",
    "sources",
    "latency_ms",
    "status_code",
    "error",
]


def should_write_header(path: Path, append: bool) -> bool:
    """
    Write the CSV header if:
    - we are not appending, or
    - the file does not exist, or
    - the file exists but is empty.
    """
    if not append:
        return True
    if not path.exists():
        return True
    return path.stat().st_size == 0


def make_row(
    q: dict[str, Any],
    result: dict[str, Any],
    status_code: int,
    latency_ms: float,
    fallback_id: str,
) -> dict[str, Any]:
    qid = q.get("id", fallback_id)
    manual_filename = q.get("manual_filename")
    question = q.get("question", "")
    expected = q.get("reference_answer", "")
    qtype = q.get("type", "")

    actual_answer = result.get("answer", "")
    sources = result.get("sources", [])
    error = result.get("error", "")

    return {
        "id": qid,
        "manual_filename": manual_filename,
        "type": qtype,
        "question": question,
        "expected": expected,
        "actual_answer": actual_answer,
        "sources": json.dumps(sources, ensure_ascii=False),
        "latency_ms": round(latency_ms, 2),
        "status_code": status_code,
        "error": error,
    }


def print_debug(row: dict[str, Any], batch_pos: int, batch_total: int) -> None:
    print(f"[{batch_pos}/{batch_total}] DONE")
    print(f"status_code: {row['status_code']}")
    print(f"latency_ms: {row['latency_ms']}")
    print(f"error: {row['error'] if row['error'] else '(none)'}")
    print()

    print("actual_answer:")
    print(row["actual_answer"] if row["actual_answer"] else "(empty)")
    print()

    print("sources:")
    try:
        parsed_sources = json.loads(row["sources"])
        print(json.dumps(parsed_sources, indent=2, ensure_ascii=False))
    except Exception:
        print(row["sources"])
    print()

    print(f"Saved row to {RESULTS_PATH}")
    print("=" * 80)
    print()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    all_questions = load_questions(QUESTIONS_PATH)
    total_questions = len(all_questions)

    if START_INDEX < 0:
        raise ValueError("START_INDEX must be 0 or greater.")

    if END_INDEX > total_questions:
        raise ValueError(
            f"END_INDEX={END_INDEX} is greater than the number of questions "
            f"({total_questions})."
        )

    if START_INDEX >= END_INDEX:
        raise ValueError("START_INDEX must be less than END_INDEX.")

    batch_questions = all_questions[START_INDEX:END_INDEX]
    batch_total = len(batch_questions)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mode = "a" if APPEND_RESULTS else "w"
    write_header = should_write_header(RESULTS_PATH, APPEND_RESULTS)

    print(f"Loaded {total_questions} questions from {QUESTIONS_PATH}")
    print(f"Running batch indexes {START_INDEX}:{END_INDEX}")
    print(f"Batch size: {batch_total}")
    print(f"Calling {ASK_URL}")
    print(f"TOP_K: {TOP_K}")
    print(f"REQUEST_TIMEOUT: {REQUEST_TIMEOUT} seconds")
    print(f"Writing results live to {RESULTS_PATH}")
    print(f"CSV mode: {'append' if APPEND_RESULTS else 'overwrite'}")
    print()

    with RESULTS_PATH.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        if write_header:
            writer.writeheader()
            f.flush()

        for batch_pos, q in enumerate(batch_questions, start=1):
            absolute_pos = START_INDEX + batch_pos
            qid = q.get("id", f"q{absolute_pos}")
            manual_filename = q.get("manual_filename")
            question = q.get("question", "")
            qtype = q.get("type", "")

            print("=" * 80)
            print(f"[{batch_pos}/{batch_total}] START")
            print(f"absolute_question_number: {absolute_pos}/{total_questions}")
            print(f"id: {qid}")
            print(f"manual_filename: {manual_filename}")
            print(f"type: {qtype}")
            print(f"question: {question}")
            print()

            result, status_code, latency_ms = post_ask(
                query=question,
                source=manual_filename,
                top_k=TOP_K,
            )

            row = make_row(
                q=q,
                result=result,
                status_code=status_code,
                latency_ms=latency_ms,
                fallback_id=f"q{absolute_pos}",
            )

            writer.writerow(row)
            f.flush()

            print_debug(row, batch_pos=batch_pos, batch_total=batch_total)

    print()
    print(f"Done. Wrote/appended {batch_total} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    main()