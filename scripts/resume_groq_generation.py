"""One-off resume script: generate Groq answers for test questions 39-52
and merge them into the existing evaluation/results/generated_groq.json.

Why this exists:
- The full run at top_k=8 hit Groq's 100K tokens/day limit after 39/53 Qs.
- We don't want to re-run the first 39 (wasted tokens, and they're already
  saved). We also want to keep the resumed 14 consistent with the first 39,
  so we explicitly disable query rewriting here — the first 39 were
  generated without it.
- run_eval.py has no --resume flag, and the instruction was not to modify it.

Usage:
    python scripts/resume_groq_generation.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is importable when run as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.test_questions import TEST_QUESTIONS  # noqa: E402
from pipeline.rag_chain import RAGChain  # noqa: E402
from pipeline.retriever import Retriever  # noqa: E402
from vectorstore.embedder import Embedder  # noqa: E402
from vectorstore.store import VectorStore  # noqa: E402

RESULTS_PATH = ROOT / "evaluation" / "results" / "generated_groq.json"
START_INDEX = 39  # resume from question #40 (0-indexed 39)
TOP_K = 8


def main() -> None:
    # Load existing results
    existing = json.loads(RESULTS_PATH.read_text())
    done = existing["results"]
    print(f"Loaded {len(done)} existing Groq results from {RESULTS_PATH}")

    if len(done) != START_INDEX:
        print(
            f"WARNING: existing results have {len(done)} entries but "
            f"START_INDEX={START_INDEX}. Proceeding with slice anyway."
        )

    remaining = TEST_QUESTIONS[START_INDEX:]
    print(f"Will generate {len(remaining)} remaining questions (indices "
          f"{START_INDEX}-{START_INDEX + len(remaining) - 1})")

    # Build a RAGChain but inject a Retriever with rewrite=False for
    # consistency with the first 39 results.
    embedder = Embedder()
    store = VectorStore()
    retriever = Retriever(embedder=embedder, store=store, rewrite=False)

    chain = RAGChain(model="groq", top_k=TOP_K)
    # Override the auto-built retriever with our rewrite-disabled one.
    chain.retriever = retriever

    new_results: list[dict] = []
    for i, (question, category, ground_truth) in enumerate(remaining, START_INDEX + 1):
        print(f"  [{i}/{len(TEST_QUESTIONS)}] {question[:60]}...")
        start = time.time()
        attempt = 0
        while True:
            attempt += 1
            try:
                result = chain.ask(question)
                elapsed = time.time() - start
                new_results.append({
                    "question": question,
                    "category": category,
                    "ground_truth": ground_truth,
                    "answer": result["answer"],
                    "contexts": result["contexts"],
                    "latency_sec": round(elapsed, 3),
                })
                print(f"    OK ({elapsed:.1f}s)")
                break
            except Exception as e:
                msg = str(e)
                if ("rate_limit" in msg.lower() or "429" in msg) and attempt <= 1:
                    # Parse retry-after if present, otherwise sleep 60s and try once
                    wait = 60
                    print(f"    RATE LIMITED — sleeping {wait}s and retrying once...")
                    time.sleep(wait)
                    start = time.time()
                    continue
                print(f"    ERROR (giving up on this question): {msg[:200]}")
                break

        # Incremental flush: save after every question so a crash doesn't
        # lose work. Merges into the existing file each time.
        merged = done + new_results
        payload = {
            "model": "groq",
            "top_k": TOP_K,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_questions": len(TEST_QUESTIONS),
            "successful": len(merged),
            "results": merged,
            "resume_note": (
                f"Questions 1-{START_INDEX} generated in original run; "
                f"questions {START_INDEX + 1}+ generated via resume script "
                f"with retriever.rewrite=False for consistency."
            ),
        }
        RESULTS_PATH.write_text(json.dumps(payload, indent=2))

    print(f"\nDone. Final count: {len(done) + len(new_results)}/{len(TEST_QUESTIONS)}")


if __name__ == "__main__":
    main()
