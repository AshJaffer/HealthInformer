"""PubMedQA benchmark: accuracy on the expert-labeled yes/no/maybe split.

Loads the PubMedQA dataset ("qiaojin/PubMedQA", "pqa_labeled" split —
1,000 expert-annotated questions), runs each through our RAG chain,
extracts yes/no/maybe from the generated answer, and compares to ground
truth. Reports overall accuracy contextualized against the 78% human
ceiling from the PubMedQA paper.

Usage:
    from evaluation.pubmedqa_bench import run_pubmedqa_benchmark
    results_df = run_pubmedqa_benchmark(model="groq", max_questions=500)
"""

import re
import time
from typing import Any

import pandas as pd
from datasets import load_dataset

from config.settings import EVAL_RESULTS_DIR
from pipeline.rag_chain import RAGChain

# Human expert ceiling from the PubMedQA paper (Jin et al., 2019)
HUMAN_CEILING: float = 0.780


def _extract_yes_no_maybe(answer: str) -> str:
    """Extract a yes/no/maybe verdict from the generated answer.

    Heuristic approach: looks for explicit yes/no/maybe near the start
    of the answer or in common answer patterns. Falls back to "maybe"
    if ambiguous.
    """
    text = answer.lower().strip()

    # Check for explicit verdict patterns
    # e.g., "Yes, ...", "No, ...", "The answer is yes", "Based on ... yes"
    verdict_patterns = [
        (r'\b(?:the answer is|conclusion is|in short|overall)[:\s]*yes\b', "yes"),
        (r'\b(?:the answer is|conclusion is|in short|overall)[:\s]*no\b', "no"),
        (r'\b(?:the answer is|conclusion is|in short|overall)[:\s]*maybe\b', "maybe"),
        (r'^yes[,.\s]', "yes"),
        (r'^no[,.\s]', "no"),
        (r'^maybe[,.\s]', "maybe"),
    ]
    for pattern, label in verdict_patterns:
        if re.search(pattern, text):
            return label

    # Count occurrences as fallback
    yes_count = len(re.findall(r'\byes\b', text))
    no_count = len(re.findall(r'\bno\b', text))
    maybe_count = len(re.findall(r'\bmaybe\b|\bunclear\b|\binsufficient\b|\bnot enough\b', text))

    if yes_count > no_count and yes_count > maybe_count:
        return "yes"
    elif no_count > yes_count and no_count > maybe_count:
        return "no"
    else:
        return "maybe"


def run_pubmedqa_benchmark(
    model: str = "groq",
    max_questions: int = 500,
) -> pd.DataFrame:
    """Run PubMedQA benchmark and report accuracy.

    Args:
        model: RAG chain LLM backend ("groq" or "bedrock").
        max_questions: Max questions to evaluate (default 500 per proposal).

    Returns:
        DataFrame with per-question results.
    """
    # ── Load PubMedQA dataset ───────────────────────────────────────────
    print("Loading PubMedQA dataset (pqa_labeled split)...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    total_available = len(dataset)
    n = min(max_questions, total_available)
    print(f"Loaded {total_available} questions, evaluating {n}")

    chain = RAGChain(model=model)

    # ── Evaluate each question ──────────────────────────────────────────
    rows: list[dict[str, Any]] = []
    correct = 0

    print(f"\nRunning PubMedQA benchmark (model={model})...")
    for i in range(n):
        item = dataset[i]
        question = item["question"]
        ground_truth = item["final_decision"]  # "yes", "no", or "maybe"

        print(f"  [{i+1}/{n}] {question[:60]}...")
        start = time.time()
        try:
            result = chain.ask(question)
            elapsed = time.time() - start
            predicted = _extract_yes_no_maybe(result["answer"])
            is_correct = predicted == ground_truth

            if is_correct:
                correct += 1

            rows.append({
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": is_correct,
                "answer": result["answer"],
                "latency_sec": elapsed,
                "model": model,
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            rows.append({
                "question": question,
                "ground_truth": ground_truth,
                "predicted": "error",
                "correct": False,
                "answer": f"ERROR: {e}",
                "latency_sec": 0.0,
                "model": model,
            })

    # ── Save results ────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVAL_RESULTS_DIR / f"pubmedqa_{model}.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # ── Print summary ───────────────────────────────────────────────────
    accuracy = correct / len(df) if len(df) > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"PubMedQA Benchmark Results (model={model})")
    print(f"{'='*60}")
    print(f"  Questions evaluated:   {len(df)}")
    print(f"  Correct:               {correct}")
    print(f"  Accuracy:              {accuracy:.1%}")
    print(f"  Human ceiling:         {HUMAN_CEILING:.1%}")
    print(f"  Gap to human:          {HUMAN_CEILING - accuracy:+.1%}")
    print(f"  Avg latency:           {df['latency_sec'].mean():.1f}s")
    print(f"")
    # Breakdown by ground truth label
    for label in ["yes", "no", "maybe"]:
        subset = df[df["ground_truth"] == label]
        if len(subset) > 0:
            label_acc = subset["correct"].mean()
            print(f"  Accuracy ({label:5s}):       {label_acc:.1%}  ({len(subset)} questions)")
    print(f"{'='*60}")

    return df


if __name__ == "__main__":
    # Quick test with a small sample
    run_pubmedqa_benchmark(model="groq", max_questions=10)
