"""RAGAS evaluation: faithfulness, answer relevancy, context precision/recall.

Runs each test question through the RAG chain, collects the
question/answer/contexts/ground_truth triples, and feeds them to RAGAS.
Results are saved as CSV to evaluation/results/.

METHODOLOGICAL NOTE — Evaluation circularity:
    When the evaluator LLM is from the same family as the generator LLM
    (e.g., both are Llama 3 via Groq), the evaluation may be biased —
    the model is effectively grading its own work. This is a known
    limitation of LLM-as-judge approaches. For a rigorous comparison,
    use a *different* model family as evaluator (e.g., evaluate Groq
    generations with a Bedrock evaluator, and vice versa). We flag this
    in the report and present results with appropriate caveats.

Usage:
    from evaluation.ragas_eval import run_ragas_evaluation
    results_df = run_ragas_evaluation(model="groq", evaluator="groq")
"""

import time
from pathlib import Path
from typing import Any

import pandas as pd
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    _AnswerRelevancy,
    _Faithfulness,
    _LLMContextPrecisionWithReference,
    _LLMContextRecall,
)

from config.settings import EVAL_RESULTS_DIR, GROQ_API_KEY
from evaluation.test_questions import TEST_QUESTIONS
from pipeline.rag_chain import RAGChain


def _get_evaluator_llm(evaluator: str = "groq") -> LangchainLLMWrapper:
    """Create a RAGAS-compatible LLM wrapper for the evaluator.

    Args:
        evaluator: "groq" or "bedrock". Defaults to groq (free).
    """
    if evaluator == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
            temperature=0.0,
        )
    elif evaluator == "bedrock":
        from langchain_aws import ChatBedrock
        llm = ChatBedrock(
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name="us-east-1",
        )
    else:
        raise ValueError(f"Unknown evaluator: {evaluator!r}")

    return LangchainLLMWrapper(llm)


def run_ragas_evaluation(
    model: str = "groq",
    evaluator: str = "groq",
    questions: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """Run RAGAS evaluation on the test question set.

    Args:
        model: RAG chain LLM backend ("groq" or "bedrock").
        evaluator: LLM to use as RAGAS judge ("groq" or "bedrock").
        questions: Override test questions. Defaults to TEST_QUESTIONS.

    Returns:
        DataFrame with per-question RAGAS scores.
    """
    questions = questions or TEST_QUESTIONS
    chain = RAGChain(model=model)

    # ── Run each question through the RAG chain ─────────────────────────
    samples: list[SingleTurnSample] = []
    timings: list[float] = []

    print(f"\nRunning {len(questions)} questions through RAG chain (model={model})...")
    for i, (question, category, ground_truth) in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {question[:60]}...")
        start = time.time()
        try:
            result = chain.ask(question)
            elapsed = time.time() - start
            timings.append(elapsed)

            sample = SingleTurnSample(
                user_input=question,
                response=result["answer"],
                retrieved_contexts=result["contexts"],
                reference=ground_truth,
            )
            samples.append(sample)
        except Exception as e:
            print(f"    ERROR: {e}")
            timings.append(0.0)

    if not samples:
        print("No successful samples. Cannot run RAGAS.")
        return pd.DataFrame()

    # ── Run RAGAS evaluation ────────────────────────────────────────────
    print(f"\nRunning RAGAS metrics (evaluator={evaluator})...")
    eval_llm = _get_evaluator_llm(evaluator)

    metrics = [
        _Faithfulness(llm=eval_llm),
        _AnswerRelevancy(llm=eval_llm),
        _LLMContextPrecisionWithReference(llm=eval_llm),
        _LLMContextRecall(llm=eval_llm),
    ]

    dataset = EvaluationDataset(samples=samples)
    eval_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=False,
        show_progress=True,
    )

    # ── Build results DataFrame ─────────────────────────────────────────
    df = eval_result.to_pandas()

    # Add metadata columns
    successful_qs = [q for q, _, _ in questions[:len(samples)]]
    categories = [c for _, c, _ in questions[:len(samples)]]
    df.insert(0, "question", successful_qs[:len(df)])
    df.insert(1, "category", categories[:len(df)])
    df["latency_sec"] = timings[:len(df)]
    df["model"] = model
    df["evaluator"] = evaluator

    # ── Save results ────────────────────────────────────────────────────
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVAL_RESULTS_DIR / f"ragas_{model}.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RAGAS Evaluation Summary (model={model}, evaluator={evaluator})")
    print(f"{'='*60}")
    metric_cols = [c for c in df.columns if c in [
        "faithfulness", "answer_relevancy",
        "context_precision", "LLMContextPrecisionWithReference",
        "context_recall", "LLMContextRecall",
    ]]
    for col in metric_cols:
        mean_val = df[col].mean()
        print(f"  {col:40s} {mean_val:.4f}")
    avg_latency = df["latency_sec"].mean()
    print(f"  {'avg_latency_sec':40s} {avg_latency:.1f}")
    print(f"  {'questions_evaluated':40s} {len(df)}")
    print(f"{'='*60}")

    return df


if __name__ == "__main__":
    run_ragas_evaluation(model="groq", evaluator="groq")
