# health-ai-rag/ — Repository Structure

```
health-ai-rag/
│
├── README.md                    # Project overview, setup instructions, how to run
├── requirements.txt             # All dependencies
├── .env.example                 # Template: GROQ_API_KEY, AWS keys, etc.
├── .gitignore                   # .env, __pycache__, chroma_db/, *.pkl
├── DATA_ACCESS.md               # Required by rubric — how to access PubMed, PubMedQA, etc.
│
│── ── ── ── ENTRY POINTS (run these in order) ── ── ── ──
│
├── ingest_data.py               # Step 1: Fetch PubMed abstracts via E-utilities API
├── build_vectorstore.py         # Step 2: Embed chunks → ChromaDB
├── run_pipeline.py              # Step 3: CLI — ask a question, get RAG answer
├── run_eval.py                  # Step 4: Run RAGAS eval + PubMedQA benchmark
├── app.py                       # Step 5: streamlit run app.py
│
│── ── ── ── MODULES ── ── ── ──
│
├── config/
│   ├── settings.py              # Central config (model names, chunk size, top_k, thresholds)
│   └── constants.py             # PubMed search terms, topic lists, file paths
│
├── data/
│   ├── fetch_pubmed.py          # PubMed E-utilities API wrapper
│   ├── fetch_guidelines.py      # NIH/WHO guideline scraper/downloader
│   ├── preprocess.py            # Chunking logic (split abstracts, clean text)
│   └── raw/                     # .gitignore'd — raw downloaded data lands here
│
├── vectorstore/
│   ├── embedder.py              # PubMedBERT embedding wrapper
│   ├── store.py                 # ChromaDB init, add, query interface
│   └── chroma_db/               # .gitignore'd — persisted ChromaDB data
│
├── pipeline/
│   ├── retriever.py             # Query → top-k chunks from ChromaDB
│   ├── generator.py             # Chunks + query → LLM prompt → cited answer
│   └── rag_chain.py             # Ties retriever + generator together
│
├── llm/
│   ├── base.py                  # Abstract LLM interface
│   ├── groq_client.py           # Groq API (Llama 3)
│   └── bedrock_client.py        # AWS Bedrock (Claude Sonnet)
│
├── prompts/
│   ├── system_prompt.txt        # Main system prompt for answer generation
│   ├── citation_prompt.txt      # Citation formatting instructions
│   └── demographic_context.txt  # Optional demographic injection template
│
├── evaluation/
│   ├── ragas_eval.py            # RAGAS runner (faithfulness, relevancy, precision, recall)
│   ├── pubmedqa_bench.py        # PubMedQA benchmark accuracy
│   ├── spot_check.py            # Sample 50 responses for qualitative review
│   └── results/                 # Eval output CSVs, plots
│
├── notebooks/
│   ├── eda.ipynb                # Data exploration, corpus stats
│   └── visualization.ipynb      # Generate report figures (3+ required)
│
└── tests/
    ├── test_retriever.py        # Does retrieval return relevant chunks?
    └── test_generator.py        # Does generation stay grounded?
```

---

## Design Rationale

### Five entry-point scripts at root
Each does one thing and runs independently. Makes the README clean:
1. `python ingest_data.py` — fetch data
2. `python build_vectorstore.py` — index it
3. `python run_pipeline.py` — test it
4. `python run_eval.py` — evaluate it
5. `streamlit run app.py` — demo it

Satisfies the rubric requirement that code generates reproducible results.

### `llm/base.py` with a common interface
Both Groq and Bedrock clients inherit from the same abstract class. `generator.py` doesn't care which backend it's using — swap models with one config change. This is what makes the model comparison clean.

### Prompts as text files
Easier to iterate, easier to reference in the report, and can be tweaked without touching Python.

### `evaluation/results/`
Where report figures come from. The visualization notebook reads CSVs from here and generates the 3+ required visualizations.

---

## Build Order (suggested)

| Phase | What | ~Timeline |
|-------|------|-----------|
| 1 | `config/`, `data/`, `ingest_data.py` — get data flowing | Days 1-3 |
| 2 | `vectorstore/`, `build_vectorstore.py` — embed + index | Days 3-5 |
| 3 | `llm/`, `pipeline/`, `prompts/`, `run_pipeline.py` — end-to-end RAG | Days 5-8 |
| 4 | `app.py` — Streamlit UI (basic working version) | Days 8-10 |
| 5 | `evaluation/`, `run_eval.py` — RAGAS + PubMedQA | Days 10-14 |
| 6 | Polish: README, DATA_ACCESS.md, notebooks, figures | Days 14-17 |