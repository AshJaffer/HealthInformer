# CLAUDE.md — HealthInformer Project Context

## What is this project?

HealthInformer is a RAG (Retrieval-Augmented Generation) medical question-answering system. A user types a health/general wellness question → we retrieve relevant passages from medical literature → an LLM synthesizes a plain-language, cited answer.

It's a **medical literacy tool for the general public** — NOT a diagnostic tool. Think: a smarter, more trustworthy version of Googling health questions where every answer is grounded in real PubMed sources with citations.

This is a capstone project for the University of Michigan MADS program (SIADS 699). Due **April 21, 2026**.

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
| **Vector store** | ChromaDB | Local, persistent, free |
| **LLM (free)** | Groq API → Llama 3.3 70B | Free tier, used as baseline |
| **LLM (paid)** | AWS Bedrock → Claude Sonnet | $200 AWS Academy credit, accessed via Learner Lab |
| **Orchestration** | LangChain | Keep it simple — just retrieval chain, no agents |
| **Evaluation** | RAGAS framework + PubMedQA benchmark | Primary automated evaluation |
| **Frontend** | Streamlit | Chat interface with source cards |
| **Language** | Python 3.11+ | Running on macOS, Apple M5 Pro, 64GB RAM |

## Repository structure

```
HealthInformer/
├── README.md
├── requirements.txt
├── .env.example                 # API keys template (never commit .env)
├── .gitignore
├── DATA_ACCESS.md               # Required: how to access all data sources
├── CLAUDE.md                    # This file
│
├── ingest_data.py               # Entry point: fetch PubMed → chunk → save
├── build_vectorstore.py         # Entry point: embed chunks → ChromaDB
├── run_pipeline.py              # Entry point: CLI to test RAG pipeline
├── run_eval.py                  # Entry point: RAGAS + PubMedQA evaluation
├── app.py                       # Entry point: streamlit run app.py
│
├── config/
│   ├── __init__.py
│   ├── settings.py              # All tunable parameters (models, thresholds, paths)
│   └── constants.py             # PubMed search queries, topic categories
│
├── data/
│   ├── __init__.py
│   ├── fetch_pubmed.py          # PubMed E-utilities API (search + fetch abstracts)
│   ├── fetch_guidelines.py      # NIH/WHO guideline downloader
│   ├── preprocess.py            # Text chunking (sentence-boundary, overlapping)
│   └── raw/                     # .gitignore'd — downloaded data
│
├── vectorstore/
│   ├── __init__.py
│   ├── embedder.py              # PubMedBERT wrapper (embed text → 768-dim vectors)
│   ├── store.py                 # ChromaDB interface (init, add, query)
│   └── chroma_db/               # .gitignore'd — persisted vector data
│
├── pipeline/
│   ├── __init__.py
│   ├── retriever.py             # Query → top-k chunks from ChromaDB
│   ├── generator.py             # Chunks + query → LLM prompt → cited answer
│   └── rag_chain.py             # Combines retriever + generator
│
├── llm/
│   ├── __init__.py
│   ├── base.py                  # Abstract LLM interface (common API)
│   ├── groq_client.py           # Groq API client (Llama 3)
│   └── bedrock_client.py        # AWS Bedrock client (Claude Sonnet)
│
├── prompts/
│   ├── system_prompt.txt        # Main system prompt for answer generation
│   ├── citation_prompt.txt      # Citation formatting instructions
│   └── demographic_context.txt  # Optional: light personalization template
│
├── evaluation/
│   ├── __init__.py
│   ├── ragas_eval.py            # RAGAS metric runner
│   ├── pubmedqa_bench.py        # PubMedQA benchmark accuracy
│   ├── spot_check.py            # Sample 50 responses for qualitative review
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

## Build phases (in order)

### Phase 1: Config + Data Ingestion
- `config/settings.py` — central config, all parameters
- `config/constants.py` — PubMed search queries defining corpus scope
- `data/fetch_pubmed.py` — E-utilities API: search for PMIDs, fetch abstracts as XML, parse to JSON
- `data/preprocess.py` — chunk abstracts into ~512-token pieces with overlap, preserve metadata
- `ingest_data.py` — entry point that runs fetch → chunk → save to `data/raw/`

### Phase 2: Embedding + Vector Store
- `vectorstore/embedder.py` — load PubMedBERT via sentence-transformers, embed text
- `vectorstore/store.py` — ChromaDB: create collection, add embedded chunks with metadata, query
- `build_vectorstore.py` — entry point: load chunks → embed → store in ChromaDB

### Phase 3: LLM Clients + RAG Pipeline
- `llm/base.py` — abstract base class with `generate(prompt, max_tokens, temperature)` method
- `llm/groq_client.py` — Groq API implementation (Llama 3.3 70B)
- `llm/bedrock_client.py` — AWS Bedrock implementation (Claude Sonnet)
- `pipeline/retriever.py` — takes query, returns top-k chunks from ChromaDB
- `pipeline/generator.py` — takes query + chunks, builds prompt, calls LLM, returns cited answer
- `pipeline/rag_chain.py` — combines retriever + generator into single `ask(question)` call
- `prompts/` — system prompt, citation instructions, demographic context template
- `run_pipeline.py` — CLI entry point to test end-to-end

### Phase 4: Streamlit UI
- `app.py` — chat interface with:
  - Text input for questions
  - Streaming response display
  - Source cards showing cited PubMed articles (title, journal, link)
  - Persistent disclaimer: "This is an educational tool, not medical advice"
  - Model toggle (Groq vs Bedrock) for demo purposes
  - Optional: age range / sex input for light personalization

### Phase 5: Evaluation
- `evaluation/ragas_eval.py` — run RAGAS metrics on a set of test questions
- `evaluation/pubmedqa_bench.py` — benchmark against PubMedQA test set (500 questions)
- `evaluation/spot_check.py` — sample 50 responses, save for manual review
- `run_eval.py` — entry point: runs all evaluation, saves results to `evaluation/results/`
- `notebooks/visualization.ipynb` — generate report figures from eval results

### Phase 6: Polish
- `README.md` — setup instructions, how to run, project overview
- `DATA_ACCESS.md` — how to access PubMed, PubMedQA, NIH/WHO data
- Clean code: remove unused blocks, convert absolute paths to relative, strip API keys
- `requirements.txt` — only libraries actually used

## Key architectural decisions

**Plain Python over heavy frameworks.** Use LangChain lightly for the retrieval chain, but keep the core logic explicit. Every component should be understandable and explainable.

**LLM abstraction via `llm/base.py`.** Both Groq and Bedrock inherit the same interface. Swapping models is a config change, not a code change. This makes the model comparison clean.

**Prompts as text files.** Stored in `prompts/`, not hardcoded. Easy to iterate, easy to reference in the report.

**Metadata travels with chunks.** Every chunk carries its source PMID, title, authors, journal, year, URL. This is how we generate citations in the answer.

**Light personalization.** User can optionally provide age range, sex, relevant conditions. This is injected into the LLM prompt at generation time, NOT used for retrieval. Simple and effective.

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

## Model comparison study

Run identical evaluation set through both backends:
- **Groq (Llama 3.3 70B)** — free tier baseline
- **AWS Bedrock (Claude Sonnet)** — paid, higher quality expected

Compare all RAGAS scores, PubMedQA accuracy, response quality, and latency. Report findings — no predetermined winner.

## Data sources

- **PubMed abstracts**: via E-utilities API (free, need email). Search ~20 health topic queries, fetch several thousand abstracts.
- **NIH/WHO guidelines**: public domain health guidelines and fact sheets.
- **PubMedQA dataset**: 1,000 expert-annotated yes/no/maybe questions. Used for evaluation only. Available on GitHub.

No raw data in the repo — provide scripts to reproduce ingestion + a DATA_ACCESS.md.

## Coding conventions

- Python 3.11+
- Type hints on all function signatures
- Docstrings on all public functions explaining what, why, and how
- Config values imported from `config/settings.py`, never hardcoded
- Print progress during long operations (data fetching, embedding, eval)
- Handle errors gracefully — don't crash on a single bad PubMed article
- Keep functions focused and small
- Use `if __name__ == "__main__"` blocks so modules are runnable standalone

## .env variables needed

```
PUBMED_EMAIL=ajaffer@umich.edu
GROQ_API_KEY=your_groq_key
AWS_ACCESS_KEY_ID=from_learner_lab
AWS_SECRET_ACCESS_KEY=from_learner_lab
AWS_SESSION_TOKEN=from_learner_lab
AWS_REGION=us-east-1
```

## Deliverables (due April 21, 2026)

- GitHub repo with README, requirements.txt, reproducible code
- Written report (under 3,000 words) with 3+ data visualizations
- 3-5 minute video summary
- Slack gallery post
- DATA_ACCESS.md
- Statement of work in report