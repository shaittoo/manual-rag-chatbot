"""
score_results.py
----------------------
Uses a free local Ollama model as an AI judge to score RAG chatbot answers.

Input:
    eval/results.csv

Output:
    eval/results_scored.csv

Run:
    cd backend
    python score_results_local.py

Before running:
    ollama pull qwen2.5:3b
"""

from __future__ import annotations

import csv
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

INPUT_PATH = PROJECT_ROOT / "eval" / "results_v1_cuda.csv"
OUTPUT_PATH = PROJECT_ROOT / "eval" / "results_scored_v1_cuda.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

VALID_SCORES = {"Correct", "Partial", "Wrong", "Refused"}


def build_prompt(row: dict[str, Any]) -> str:
    return f"""
You are evaluating a RAG chatbot for product manuals.

Compare the chatbot's actual answer against the expected reference answer.

Question:
{row.get("question", "")}

Expected reference answer:
{row.get("expected", "")}

Actual chatbot answer:
{row.get("actual_answer", "")}

Error column:
{row.get("error", "")}

Scoring labels:
- Correct: The actual answer contains the main required facts or steps from the expected answer. Wording does not need to be identical.
- Partial: The actual answer is relevant and includes some correct information, but misses important required details or is incomplete.
- Wrong: The actual answer is mostly unrelated, unsupported, misleading, contradictory, or answers a different question.
- Refused: The actual answer says it does not know, cannot answer, lacks context, timed out, or gives no useful answer.

Rules:
- Do not require exact wording.
- Be strict but fair.
- If actual_answer is empty, mark Refused.
- If the error column says timed out and actual_answer is empty, mark Refused.
- Return only valid JSON. No markdown. No explanation outside JSON.

Return exactly this format:
{{
  "score": "Correct",
  "reason": "Brief explanation."
}}
""".strip()


def call_ollama(prompt: str, timeout: int = 180) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
        return data.get("response", "").strip()


def extract_json(text: str) -> dict[str, str]:

    # Tries to parse JSON even if the model accidentally adds extra text.
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])

        raise


def judge_row(row: dict[str, Any]) -> dict[str, str]:
    actual_answer = row.get("actual_answer", "").strip()
    error = row.get("error", "").strip().lower()

    if not actual_answer:
        return {
            "score": "Refused",
            "reason": "The actual answer is empty or no useful answer was produced."
        }

    if "timed out" in error and not actual_answer:
        return {
            "score": "Refused",
            "reason": "The request timed out and no answer was produced."
        }

    prompt = build_prompt(row)

    try:
        raw = call_ollama(prompt)
        parsed = extract_json(raw)

        score = parsed.get("score", "").strip()
        reason = parsed.get("reason", "").strip()

        if score not in VALID_SCORES:
            return {
                "score": "Wrong",
                "reason": f"Local judge returned invalid score: {score}. Raw output: {raw}"
            }

        return {
            "score": score,
            "reason": reason
        }

    except Exception as e:
        return {
            "score": "Wrong",
            "reason": f"Local AI judge failed: {e}"
        }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Could not find {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("results.csv has no rows.")

    fieldnames = list(rows[0].keys())

    for col in ["ai_score", "ai_reason"]:
        if col not in fieldnames:
            fieldnames.append(col)

    scored_rows = []

    print(f"Loaded {len(rows)} rows from {INPUT_PATH}")
    print(f"Scoring with local Ollama model: {MODEL}")
    print()

    for i, row in enumerate(rows, start=1):
        print("=" * 80)
        print(f"[{i}/{len(rows)}] {row.get('id', '')}")
        print(f"Question: {row.get('question', '')}")

        result = judge_row(row)

        row["ai_score"] = result["score"]
        row["ai_reason"] = result["reason"]

        scored_rows.append(row)

        print(f"AI score: {result['score']}")
        print(f"Reason: {result['reason']}")

        with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(scored_rows)

        print(f"Saved progress to {OUTPUT_PATH}")
        print()

    counts = {score: 0 for score in VALID_SCORES}

    for row in scored_rows:
        score = row.get("ai_score", "")
        if score in counts:
            counts[score] += 1

    total = len(scored_rows)
    strict_accuracy = counts["Correct"] / total if total else 0
    lenient_accuracy = (counts["Correct"] + 0.5 * counts["Partial"]) / total if total else 0

    print("=" * 80)
    print("Summary")
    print(f"Correct: {counts['Correct']}")
    print(f"Partial: {counts['Partial']}")
    print(f"Wrong: {counts['Wrong']}")
    print(f"Refused: {counts['Refused']}")
    print(f"Strict accuracy: {strict_accuracy:.2%}")
    print(f"Lenient accuracy: {lenient_accuracy:.2%}")
    print()
    print(f"Done. Wrote scored results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()