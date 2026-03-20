"""Entry point: run all evaluations and print a final summary table.

Usage:
    python run_eval.py                          # Run all evals with groq
    python run_eval.py --model bedrock          # Use Bedrock for generation
    python run_eval.py --ragas-only             # Only RAGAS metrics
    python run_eval.py --pubmedqa-only          # Only PubMedQA benchmark
    python run_eval.py --spot-check-only        # Only spot check
    python run_eval.py --evaluator bedrock      # Use Bedrock as RAGAS judge
    python run_eval.py --pubmedqa-max 100       # Limit PubMedQA questions
"""

import argparse
import sys

import pandas as pd

from config.settings import EVAL_RESULTS_DIR


# Proposal thresholds (from CLAUDE.md)
THRESHOLDS: dict[str, tuple[str, float]] = {
    "faithfulness":     ("RAGAS Faithfulness",           0.8),
    "answer_relevancy": ("RAGAS Answer Relevancy",       0.7),
    "context_precision": ("RAGAS Context Precision",     0.7),
    "LLMContextPrecisionWithReference": ("RAGAS Context Precision", 0.7),
    "context_recall":   ("RAGAS Context Recall",         0.6),
    "LLMContextRecall": ("RAGAS Context Recall",         0.6),
}


def _print_threshold_table(ragas_df: pd.DataFrame | None, pubmedqa_df: pd.DataFrame | None) -> None:
    """Print a summary table comparing results to proposal thresholds."""
    print(f"\n{'='*70}")
    print(f"{'METRIC':<40} {'RESULT':>8} {'TARGET':>8} {'STATUS':>8}")
    print(f"{'-'*70}")

    if ragas_df is not None and not ragas_df.empty:
        for col, (label, target) in THRESHOLDS.items():
            if col in ragas_df.columns:
                val = ragas_df[col].mean()
                status = "PASS" if val >= target else "MISS"
                print(f"  {label:<38} {val:>7.3f} {target:>7.1f}  {'  ' + status}")

        # Latency
        avg_lat = ragas_df["latency_sec"].mean()
        lat_status = "PASS" if avg_lat < 10.0 else "MISS"
        print(f"  {'Response Latency (sec)':<38} {avg_lat:>7.1f} {'<10.0':>8}  {'  ' + lat_status}")

    if pubmedqa_df is not None and not pubmedqa_df.empty:
        accuracy = pubmedqa_df["correct"].mean()
        print(f"  {'PubMedQA Accuracy':<38} {accuracy:>7.1%} {'78.0%':>8}  {'  context'}")

    print(f"{'='*70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HealthInformer evaluations")
    parser.add_argument("--model", type=str, default="groq", choices=["groq", "bedrock"],
                        help="LLM backend for answer generation (default: groq)")
    parser.add_argument("--evaluator", type=str, default="groq", choices=["groq", "bedrock"],
                        help="LLM backend for RAGAS evaluation (default: groq)")
    parser.add_argument("--ragas-only", action="store_true",
                        help="Run only RAGAS evaluation")
    parser.add_argument("--pubmedqa-only", action="store_true",
                        help="Run only PubMedQA benchmark")
    parser.add_argument("--spot-check-only", action="store_true",
                        help="Run only spot check")
    parser.add_argument("--pubmedqa-max", type=int, default=500,
                        help="Max PubMedQA questions (default: 500)")
    parser.add_argument("--spot-check-size", type=int, default=50,
                        help="Spot check sample size (default: 50)")
    args = parser.parse_args()

    # Determine which evals to run
    run_all = not (args.ragas_only or args.pubmedqa_only or args.spot_check_only)
    run_ragas = run_all or args.ragas_only
    run_pubmedqa = run_all or args.pubmedqa_only
    run_spot = run_all or args.spot_check_only

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ragas_df = None
    pubmedqa_df = None

    # ── RAGAS ───────────────────────────────────────────────────────────
    if run_ragas:
        print("\n" + "="*60)
        print("RUNNING RAGAS EVALUATION")
        print("="*60)
        from evaluation.ragas_eval import run_ragas_evaluation
        try:
            ragas_df = run_ragas_evaluation(
                model=args.model, evaluator=args.evaluator,
            )
        except Exception as e:
            print(f"RAGAS evaluation failed: {e}")

    # ── PubMedQA ────────────────────────────────────────────────────────
    if run_pubmedqa:
        print("\n" + "="*60)
        print("RUNNING PUBMEDQA BENCHMARK")
        print("="*60)
        from evaluation.pubmedqa_bench import run_pubmedqa_benchmark
        try:
            pubmedqa_df = run_pubmedqa_benchmark(
                model=args.model, max_questions=args.pubmedqa_max,
            )
        except Exception as e:
            print(f"PubMedQA benchmark failed: {e}")

    # ── Spot check ──────────────────────────────────────────────────────
    if run_spot:
        print("\n" + "="*60)
        print("RUNNING SPOT CHECK")
        print("="*60)
        from evaluation.spot_check import run_spot_check
        try:
            run_spot_check(
                model=args.model, sample_size=args.spot_check_size,
            )
        except Exception as e:
            print(f"Spot check failed: {e}")

    # ── Final summary ───────────────────────────────────────────────────
    _print_threshold_table(ragas_df, pubmedqa_df)


if __name__ == "__main__":
    main()
