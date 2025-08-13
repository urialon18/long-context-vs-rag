# RAG Comparison: Regular RAG vs Long-Context RAG

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](#) [![Packaging: uv](https://img.shields.io/badge/packaging-uv-000000.svg)](#)

## Purpose
Compare two RAG strategies on long documents from the LooGLE dataset (`bigai-nlco/LooGLE`, `longdep_qa`):
- Regular RAG: chunk, embed, store, retrieve, then answer
- Long-Context: pass the full document context directly to the LLM

The goal is to assess cost/speed/robustness trade-offs under a fair setup.

## Design
- Dataset: LooGLE (`longdep_qa`) with filtering for non-sensitive content and max context length.
- Model: single LLM client (Gemini 2.5 Flash) used by both pipelines.
- Evaluation: direct comparison of predictions to ground-truth; metrics include accuracy, success rate, time, cost, tokens.
- Rate limiting: configurable sleeps between queries for each pipeline.

## Project Structure
```
long-rag-vs-regular-rag/
├── src/
│   ├── data/
│   │   └── loogle_loader.py         # Load, filter, and sample LooGLE questions
│   ├── regular_rag/
│   │   ├── chunker.py               # Chunking
│   │   ├── embedder.py              # Embeddings (sentence-transformers)
│   │   ├── vector_store.py          # ChromaDB storage
│   │   ├── retriever.py             # Retrieval (no query rewriting)
│   │   └── pipeline.py              # Regular RAG pipeline
│   ├── long_context_rag/
│   │   └── pipeline.py              # Long-Context RAG pipeline
│   ├── evaluation/
│   │   ├── loogle_evaluator.py      # Orchestrates runs and metrics
│   │   └── metrics.py               # Metrics calculation
│   └── utils/
│       ├── config.py                # All configuration
│       └── llm_client.py            # Gemini client
├── chroma_db/                       # ChromaDB persistence
├── results/                         # Saved JSON outputs
├── scripts/
│   └── clear_cache.py               # Clear caches (optional)
└── main.py                          # Entry point
```

## Quick Start
1) Install
```
uv sync
```
2) Set your API key in the shell (replace ... with your key)
```
export GOOGLE_API_KEY=...
```
3) Run
```
uv run main.py
```

## Configure
Edit `src/utils/config.py` (key fields):
- dataset_name / dataset_subset / dataset_split
- n_questions, random_seed
- max_context_length (characters)
- regular_rag_sleep, long_context_rag_sleep
- chunk_size, chunk_overlap, retrieval_k
- embedding_model, llm_model, llm_temperature

## Outputs
Saved under `results/`:
- Complete run: `loogle_rag_comparison_YYYYMMDD_HHMMSS.json`
- Per-pipeline: `loogle_regular_rag_*.json`, `loogle_long_context_rag_*.json`
