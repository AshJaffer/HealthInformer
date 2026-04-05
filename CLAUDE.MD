# CLAUDE.md — HealthInformer Project Context

## What is this project?

HealthInformer is a RAG (Retrieval-Augmented Generation) medical question-answering system. A user types a health question → we retrieve relevant passages from medical literature → an LLM synthesizes a plain-language, cited answer.

It's a **medical literacy tool for the general public** — NOT a diagnostic tool. Think: a smarter, more trustworthy version of Googling health questions where every answer is grounded in real PubMed sources with citations.

This is a capstone project for the University of Michigan MADS program (SIADS 699). Due **April 21, 2026**.

## Current status

All 5 phases are built and working. The system is functional end-to-end.

| Phase | Status | Details |
|---|---|---|
| Phase 1: Data Ingestion | ✅ Complete | 21,926 articles, 40,096 chunks across 36 categories (124 queries) |
| Phase 2: Vector Store | ✅ Complete | All 40K chunks embedded with PubMedBERT, stored in ChromaDB |
| Phase 3: RAG Pipeline | ✅ Complete | Groq (Llama 3.3 70B) + Bedrock (Claude 3.5 Haiku) both working |
| Phase 4: Streamlit UI | ✅ Complete | Chat interface with citations, source cards, disclaimers, demographic inputs |
| Phase 5: Evaluation | ✅ Bedrock complete | RAGAS + PubMedQA + spot-check all run for Bedrock. Groq pending. |

### Completed evaluation
- ✅ **RAGAS Bedrock (post-HyDE, top_k=8):** Faithfulness 0.856, Answer Relevancy 0.923, Context Precision 0.478, Context Recall 0.624
- ✅ **PubMedQA Bedrock:** 65.6% accuracy (78% human ceiling). Uses gold context from dataset, not our retriever.
- ✅ **Spot-check:** 2/50 hallucinations (4%), mean quality 4.14/5. Annotated by Ashhad Jaffer.
- ✅ **HyDE retrieval fix:** Implemented in `pipeline/retriever.py`. Lay-query relevance improved from 28% → 66% on 4-question A/B test. Bedrock Haiku generates hypothetical abstract snippets before embedding.
- ✅ **Pre-HyDE baselines backed up:** `generated_bedrock_pre_hyde.json`, `ragas_bedrock_pre_hyde.csv` for before/after comparison in report.
- ✅ **PubMedQA benchmark fixed:** `pubmedqa_bench.py` rewritten to use gold context + deterministic verdict extraction. Broken run preserved as `pubmedqa_bedrock_broken.csv` (safety finding: system refuses out-of-corpus questions).

### Remaining work
- Groq: wipe `generated_groq.json`, re-generate all 53 with HyDE, then RAGAS score with Bedrock evaluator
- Populate `notebooks/visualization.ipynb` (3+ figures) and `notebooks/eda.ipynb`
- Write `README.md` and `DATA_ACCESS.md`
- Code cleanup + inline attributions
- Report, video, Slack gallery post
- Pre-flight checklist due April 14

## Core question

Can a RAG pipeline produce substantive, accurately cited health information that approaches the quality of expert-curated medical answers?

## Target user

The general public — people who Google health questions but want evidence-grounded answers. Responses must be:
- Written in plain language (not academic jargon)
- Cited with accessible PubMed links (not DOIs)
- Accompanied by disclaimers that this is educational, not medical advice

## Tech stack

| Layer | Tool | Notes |
|---|---|---|
| **Data sources** | PubMed E-utilities API, NIH/WHO guidelines | Free, no key needed (just email) |
| **Embeddings** | PubMedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`) | Domain-specific medical embeddings, 768-dim, 512 token max |
| **Vector store** | ChromaDB | Local, persistent, free. Collection name: `healthinformer` |
| **LLM (free)** | Groq API → Llama 3.3 70B Versatile | Free tier, 100K tokens/day rate limit |
| **LLM (paid)** | AWS Bedrock → Claude 3.5 Haiku | $200 AWS Academy credit via Learner Lab. Sonnet blocked by SCP. |
| **Orchestration** | LangChain (light) | Retrieval chain only, no agents |
| **Evaluation** | RAGAS framework + PubMedQA benchmark | RAGAS uses Groq as evaluator LLM + PubMedBERT for embeddings |
| **Frontend** | Streamlit | Chat interface with source cards |
| **Language** | Python 3.11+ | Running on macOS, Apple M5 Pro, 64GB RAM, 1TB storage |

## Repository structure

```
HealthInformer/
├── README.md
├── requirements.txt
├── .env.example                 # API keys template (never commit .env)
├── .env                         # gitignored — actual keys
├── .gitignore
├── DATA_ACCESS.md               # Required: how to access all data sources
├── CLAUDE.md                    # This file
│
├── ingest_data.py               # Entry: fetch PubMed → chunk → save. Supports --test, --max-results
├── build_vectorstore.py         # Entry: embed chunks → ChromaDB. Supports --test-query, --batch-size
├── run_pipeline.py              # Entry: CLI Q&A. Supports --model groq/bedrock, --top-k N
├── run_eval.py                  # Entry: full eval suite. Supports --ragas-only, --pubmedqa-only, --spot-check-only, --model, --evaluator
├── app.py                       # Entry: streamlit run app.py
│
├── config/
│   ├── __init__.py
│   ├── settings.py              # All tunable parameters (models, thresholds, paths, API keys)
│   └── constants.py             # 124 PubMed search queries across 36 topic categories
│
├── data/
│   ├── __init__.py
│   ├── fetch_pubmed.py          # PubMed E-utilities API (search + fetch abstracts)
│   ├── fetch_guidelines.py      # NIH/WHO guideline downloader
│   ├── preprocess.py            # Text chunking (sentence-boundary, overlapping, ~512 tokens)
│   └── raw/                     # gitignored — articles.json (21,926) + chunks.json (40,096)
│
├── vectorstore/
│   ├── __init__.py
│   ├── embedder.py              # PubMedBERT wrapper via sentence-transformers
│   ├── store.py                 # ChromaDB: init, add (upsert), query with cosine similarity
│   └── chroma_db/               # gitignored — persisted ChromaDB data
│
├── pipeline/
│   ├── __init__.py
│   ├── retriever.py             # Query → embed → ChromaDB top-k → chunks with metadata
│   ├── generator.py             # Chunks + query + demographic context → LLM prompt → cited answer
│   └── rag_chain.py             # Combines retriever + generator. ask() returns answer + sources + contexts
│
├── llm/
│   ├── __init__.py
│   ├── base.py                  # Abstract BaseLLM with generate(prompt, max_tokens, temperature, system_prompt)
│   ├── groq_client.py           # Groq chat completions API
│   └── bedrock_client.py        # AWS Bedrock Converse API (not invoke_model — needed for inference profiles)
│
├── prompts/
│   ├── system_prompt.txt        # Main: plain language, inline citations [1][2], sources list, disclaimer
│   ├── citation_prompt.txt      # Citation formatting rules
│   └── demographic_context.txt  # Optional age/sex personalization template
│
├── evaluation/
│   ├── __init__.py
│   ├── test_questions.py        # Curated test questions with ground truth sketches — ✅ 53 questions covering all 36 categories
│   ├── ragas_eval.py            # RAGAS runner using ChatGroq + PubMedBERT (no OpenAI dependency)
│   ├── pubmedqa_bench.py        # PubMedQA benchmark (1,000 Qs, yes/no/maybe extraction)
│   ├── spot_check.py            # Sample 50 responses → CSV for manual review
│   └── results/                 # Output CSVs and plots
│
├── notebooks/
│   ├── eda.ipynb                # Corpus stats and exploration
│   └── visualization.ipynb      # Generate report figures (need 3+ visuals)
│
└── tests/
    ├── __init__.py
    ├── test_retriever.py
    └── test_generator.py
```

## Corpus stats

- **124 PubMed queries** across 36 topic categories
- **21,926 unique articles** (deduplicated across queries)
- **40,096 chunks** (~512 tokens each with overlap)
- Top categories by chunk count: Nutrition (5,327), Mental Health (5,319), Exercise (3,848), Pain Management (2,829), Dermatology (1,878), Heart Health (1,644), Sexual Health (1,741), Digestive Health (1,406), Substance Use (1,434)

## Key architectural decisions

**Bedrock uses Converse API, not invoke_model.** Newer inference profiles (like Haiku) don't support the old `invoke_model` with `anthropic_version`. The `converse()` API is model-agnostic.

**Bedrock model is Claude 3.5 Haiku**, not Sonnet. Sonnet was blocked by the Learner Lab's Service Control Policy. Model ID: `us.anthropic.claude-3-5-haiku-20241022-v1:0`

**RAGAS evaluator uses Bedrock Haiku (as judge LLM) + PubMedBERT (embeddings).** `ragas_eval.py` wraps our PubMedBERT embedder in a LangChain `Embeddings` interface. Groq was originally used as evaluator but replaced with Bedrock Haiku to eliminate circularity — a separate model now judges both backends.

**RAGAS evaluation circularity** — Resolved. Refactored pipeline uses Bedrock Haiku as an independent evaluator LLM for both model backends, eliminating same-model evaluation bias.

**Groq free tier rate limit** is 100K tokens/day. Run eval in batches if needed. Bedrock has no rate limit but credentials expire every ~4 hours (Learner Lab).

**LLM abstraction via `llm/base.py`.** Both Groq and Bedrock inherit the same interface. Swapping models is a config change.

**Prompts as text files** in `prompts/`, not hardcoded.

**Metadata travels with chunks.** Every chunk carries PMID, title, authors, journal, year, URL, category. Used for citation generation.

**Light personalization.** Optional age range + sex injected into LLM prompt at generation time, NOT used for retrieval.

## Evaluation thresholds (from proposal)

| Metric | Target | What it measures |
|---|---|---|
| RAGAS Faithfulness | > 0.8 | Are generated claims supported by retrieved context? |
| RAGAS Context Precision | > 0.7 | Are retrieved passages actually relevant? |
| RAGAS Context Recall | > 0.6 | Did we retrieve all the relevant info? |
| RAGAS Answer Relevancy | > 0.7 | Is the response pertinent to the question? |
| PubMedQA Accuracy | Contextualize vs 78% human ceiling | How well do we answer yes/no/maybe medical Qs? |
| Hallucination Rate | < 10% | Manual spot-check of 50 responses |
| Response Latency | < 10 seconds | End-to-end query time |

**Bedrock results (post-HyDE, top_k=8):**

| Metric | Pre-HyDE | Post-HyDE | Target | Status |
|---|---|---|---|---|
| Faithfulness | 0.743 | **0.856** | > 0.8 | ✅ PASS |
| Answer Relevancy | 0.679 | **0.923** | > 0.7 | ✅ PASS |
| Context Precision | 0.213 | **0.478** | > 0.7 | ❌ MISS (+124%) |
| Context Recall | 0.296 | **0.624** | > 0.6 | ✅ PASS |
| PubMedQA Accuracy | — | **65.6%** | vs 78% ceiling | 12.4 pp gap |
| Hallucination Rate | — | **4%** (2/50) | < 10% | ✅ PASS |
| Response Latency | 7.8s | **12.8s** | < 10s | ❌ MISS (HyDE cost) |

Pre-HyDE baselines are from the original Bedrock run before implementing HyDE query expansion. The latency increase is the cost of an extra LLM call per query for HyDE rewriting. Context precision remains below target — a cross-encoder reranker or reducing top_k could close the gap but was not pursued given time constraints.

## Model comparison study

Run identical evaluation set through both backends:
- **Groq (Llama 3.3 70B Versatile)** — free tier baseline
- **AWS Bedrock (Claude 3.5 Haiku)** — paid via Learner Lab

Compare all RAGAS scores, PubMedQA accuracy, response quality, and latency. Report findings — no predetermined winner.

## .env variables needed

```
PUBMED_EMAIL=your_email@umich.edu
GROQ_API_KEY=gsk_your_key
AWS_ACCESS_KEY_ID=from_learner_lab
AWS_SECRET_ACCESS_KEY=from_learner_lab
AWS_SESSION_TOKEN=from_learner_lab
AWS_REGION=us-east-1
```

**Note:** AWS Learner Lab credentials expire every ~4 hours. Re-copy from Learner Lab → AWS Details each session.

## Coding conventions

- Python 3.11+
- Type hints on all function signatures
- Docstrings on all public functions explaining what, why, and how
- Config values imported from `config/settings.py`, never hardcoded
- Print progress during long operations (data fetching, embedding, eval)
- Handle errors gracefully — don't crash on a single bad article or API timeout
- Keep functions focused and small
- Use `if __name__ == "__main__"` blocks so modules are runnable standalone

## Deliverables (due April 21, 2026)

- GitHub repo with README, requirements.txt, DATA_ACCESS.md, reproducible code
- Written report (under 3,000 words) with 3+ data visualizations
- 3-5 minute video summary or poster
- Slack gallery post in #mads-capstone-gallery
- Statement of work in report
- In-line code attributions for external code