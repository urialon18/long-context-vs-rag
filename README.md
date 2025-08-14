# RAG Comparison: Regular RAG vs Long-Context RAG

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](#) [![Packaging: uv](https://img.shields.io/badge/packaging-uv-000000.svg)](#)

## Purpose
Compare two RAG strategies on long documents from the LooGLE dataset (`bigai-nlco/LooGLE`, `longdep_qa`):
- **Regular RAG**: chunk, embed, store, retrieve top-k chunks, then answer
- **Long-Context**: pass ALL available document contexts directly to the LLM

The goal is to assess accuracy, cost, and speed trade-offs with fair access to the same information pool.

## Key Features
- **Fair Comparison**: Both methods have access to ALL documents in the corpus
- **Aligned Information Access**: Long-Context gets all contexts combined; Regular RAG retrieves from entire corpus
- **Comprehensive Evaluation**: Tracks accuracy, cost, response time, and retrieved context
- **Duplicate-Free Indexing**: Regular RAG indexes only unique documents to prevent chunk duplication
- **Flexible Evaluation**: Run individual methods or both for direct comparison
- **Resume Support**: Continues from where previous runs left off

## Project Structure
```
long-rag-vs-regular-rag/
├── src/
│   ├── regular_rag/                 # Regular RAG implementation
│   │   ├── chunker.py               # Document chunking
│   │   ├── embedder.py              # Embeddings (sentence-transformers)
│   │   ├── vector_store.py          # ChromaDB storage
│   │   ├── retriever.py             # Chunk retrieval
│   │   └── pipeline.py              # Complete Regular RAG pipeline
│   ├── long_context_rag/
│   │   └── pipeline.py              # Long-Context RAG pipeline
│   ├── evaluation/                  # Evaluation utilities
│   └── utils/
│       ├── config.py                # Configuration settings
│       └── llm_client.py            # Gemini API client
├── data/                            # Dataset and results
│   ├── loogle_questions_with_short_answers.csv  # Evaluation dataset
│   └── results/                     # Output CSV files
├── scripts/                         # Utility scripts
│   ├── download_loogle_dataset.py   # Download LooGLE dataset
│   ├── create_context_corpus.py     # Filter dataset for short answers
│   └── clear_cache.py               # Clear ChromaDB cache
├── chroma_db/                       # ChromaDB persistence (auto-created)
└── main.py                          # Main evaluation script
```

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Set API Key
You need a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey):

**Option A: Environment Variable**
```bash
export GEMINI_API_KEY=your_actual_api_key_here
```

**Option B: .env File**
```bash
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
```

### 3. Prepare Dataset
Download and prepare the evaluation dataset:
```bash
# Download LooGLE dataset
uv run scripts/download_loogle_dataset.py

# Create filtered dataset with short answers (30 questions)
uv run scripts/create_context_corpus.py --k 30
```

### 4. Run Evaluation
```bash
# Compare both methods (recommended)
uv run main.py --method both --max-questions 10

# Run only Regular RAG
uv run main.py --method regular-rag --max-questions 10

# Run only Long Context
uv run main.py --method long-context --max-questions 10
```

## Command Line Options

```bash
uv run main.py [OPTIONS]
```

**Key Parameters:**
- `--method {long-context,regular-rag,both}`: Evaluation method (default: both)
- `--model MODEL`: LLM model to use (default: gemini/gemini-2.5-flash)  
- `--reasoning-effort {low,medium,high,disable}`: Reasoning effort level (default: high)
- `--max-questions N`: Maximum questions to process (default: 20)

**Examples:**
```bash
# Full comparison with medium reasoning
uv run main.py --method both --reasoning-effort medium --max-questions 20

# Quick test with 5 questions
uv run main.py --method both --max-questions 5

# Regular RAG only with high reasoning
uv run main.py --method regular-rag --reasoning-effort high
```

## Configuration

Edit `src/utils/config.py` for advanced settings:
- `chunk_size`, `chunk_overlap`: Chunking parameters for Regular RAG
- `retrieval_k`: Number of chunks to retrieve (default: 5)
- `embedding_model`: Sentence transformer model
- `llm_model`, `llm_temperature`: LLM settings

## Output Files

Results are saved in `data/results/`:
- `long_context_results_{model}_{reasoning_effort}.csv`: Long Context results
- `regular_rag_results_{model}_{reasoning_effort}.csv`: Regular RAG results

**CSV Columns:**
- `question`, `answer`: Original question and ground truth
- `llm_answer`: Generated answer
- `is_correct`: Boolean correctness (LLM-evaluated)
- `retrieved_context`: Context provided to LLM
- `actual_cost_usd`, `response_time_seconds`: Performance metrics

## Understanding Results

The evaluation provides:
- **Success Rate**: Percentage of correct answers
- **Cost Comparison**: API usage costs (Long Context typically much higher)
- **Speed Analysis**: Response times
- **Context Analysis**: What information was actually used (`retrieved_context` column)

**Example Output:**
```
🏆 COMPARISON:
   🥇 Long Context wins by 15.0% (65.0% vs 50.0%)
   💰 Long Context: $0.234 | Regular RAG: $0.012
```

## Troubleshooting

**"No module named 'regular_rag'" Error:**
- Make sure you're running from the project root directory
- Verify `src/` directory exists with the required modules

**"GEMINI_API_KEY not found" Error:**
- Set your API key as described in the Quick Start section
- Verify the key is valid by testing with Google AI Studio

**ChromaDB Issues:**
- Clear the vector store: `rm -rf chroma_db/`
- The system will automatically re-index on next run

**Empty Results:**
- Ensure the dataset exists: `data/loogle_questions_with_short_answers.csv`
- Run dataset preparation scripts if missing
