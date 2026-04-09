# HealthInformer

A Retrieval-Augmented Generation (RAG) system that answers health questions using evidence from PubMed medical literature. Users ask questions in plain language and receive cited, accessible responses grounded in peer-reviewed research. HealthInformer is a medical literacy tool for the general public -- not a diagnostic tool. Every answer includes inline citations linking to the original PubMed articles and a disclaimer that the information is educational, not medical advice.

Capstone project for the University of Michigan MADS program (SIADS 699).

## Tech Stack

| Layer | Tool |
|---|---|
| Data Source | PubMed E-utilities API |
| Embeddings | PubMedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`) |
| Vector Store | ChromaDB |
| LLM (commercial) | AWS Bedrock -- Claude 3.5 Haiku |
| LLM (open-source) | AWS Bedrock -- Llama 3.3 70B Instruct |
| Orchestration | LangChain (retrieval chain only) |
| Evaluation | RAGAS framework + PubMedQA benchmark |
| Frontend | Streamlit |

## Setup

```bash
git clone https://github.com/<your-org>/HealthInformer.git
cd HealthInformer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required environment variables:

| Variable | Source |
|---|---|
| `PUBMED_EMAIL` | Any valid email (PubMed API policy) |
| `GROQ_API_KEY` | [Groq Console](https://console.groq.com) (free tier available) |
| `AWS_ACCESS_KEY_ID` | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials |
| `AWS_SESSION_TOKEN` | AWS credentials (if using temporary credentials) |
| `AWS_REGION` | AWS region (default: `us-east-1`) |

## How to Run

Run the entry-point scripts in order to reproduce the full pipeline:

```bash
# 1. Fetch PubMed articles and chunk them
python ingest_data.py

# 2. Embed chunks and build ChromaDB vector store
python build_vectorstore.py

# 3. Ask a question via the CLI
python run_pipeline.py --model bedrock --top-k 8

# 4. Launch the Streamlit chat interface
streamlit run app.py

# 5. Run the evaluation suite
python run_eval.py --model bedrock --evaluator bedrock
```

## Repository Structure

```
HealthInformer/
├── ingest_data.py               # Fetch PubMed abstracts, chunk, and save
├── build_vectorstore.py         # Embed chunks into ChromaDB
├── run_pipeline.py              # CLI question-answering
├── run_eval.py                  # RAGAS + PubMedQA + spot-check evaluation
├── app.py                       # Streamlit chat interface
│
├── config/                      # Settings and PubMed search queries
├── data/                        # Data fetching and preprocessing
├── vectorstore/                 # PubMedBERT embedder and ChromaDB interface
├── pipeline/                    # Retriever, generator, and RAG chain
├── llm/                         # LLM client abstraction (Groq, Bedrock)
├── prompts/                     # System, citation, and demographic prompts
├── evaluation/                  # Eval modules and result artifacts
│   └── results/                 # RAGAS CSVs, PubMedQA scores, spot-checks
├── notebooks/                   # Visualization notebooks (3 figures)
└── figures/                     # Generated figures for the report
```

## Evaluation Results

Both models evaluated on the same vector store, same HyDE rewriter, and same evaluator LLM (Claude Haiku as independent judge).

### RAGAS (53 curated questions)

| Metric | Claude 3.5 Haiku | Llama 3.3 70B | Target |
|---|---|---|---|
| Faithfulness | **0.898** | 0.880 | > 0.8 |
| Answer Relevancy | **0.914** | 0.898 | > 0.7 |
| Context Precision | 0.474 | 0.470 | > 0.7 |
| Context Recall | 0.560 | **0.570** | > 0.6 |

### PubMedQA Benchmark (500 questions, gold context)

| Model | Accuracy | Human Ceiling |
|---|---|---|
| Claude 3.5 Haiku | 65.6% | 78% |
| Llama 3.3 70B | **77.2%** | 78% |

### Manual Spot-Check (50 questions each, annotated by lead developer)

| Metric | Claude 3.5 Haiku | Llama 3.3 70B |
|---|---|---|
| Hallucination Rate | 0% (0/50) | 0% (0/50) |
| Mean Quality (1--5) | **4.28** | 3.96 |
| Avg. Latency | 13.0s | **5.5s** |

## Team

- **Ashhad Jaffer** 
- **Naseem Heydari** 
