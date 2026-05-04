# League of Legends Patch Notes RAG

A Retrieval-Augmented Generation system that answers questions about League of Legends patch notes using an open-source LLM (TinyLlama-1.1B-Chat), pgvector for vector storage, and RAGAS for evaluation.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐
│  Scraper    │────▶│  Chunking    │────▶│  Embeddings         │
│  (requests  │     │  (section-   │     │  (all-MiniLM-L6-v2) │
│   + bs4)    │     │   aware)     │     └──────────┬──────────┘
└─────────────┘     └──────────────┘                │
                                                    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐
│  TinyLlama  │◀────│  RAG Engine  │◀────│  PostgreSQL         │
│  1.1B-Chat  │     │  (retrieve + │     │  + pgvector         │
│             │     │   generate)  │     │  (HNSW index)       │
└──────┬──────┘     └──────────────┘     └─────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  RAGAS Evaluation: faithfulness, relevancy, precision, recall   │
│  Custom Metrics: latency, MRR, context utilization, confidence  │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- ~4GB RAM minimum (for TinyLlama-1.1B in float16), GPU optional but recommended

## Quick Start

### 1. Setup

```bash
# Clone and enter the project
cd league_of_legends_rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
```

### 2. Start PostgreSQL + pgvector

```bash
docker compose up -d
```

This starts a PostgreSQL 16 container with the pgvector extension, automatically creating the schema on first run.

### 3. Scrape & Ingest Patch Notes

```bash
# Scrape patch notes (cached to data/raw/)
python cli.py scrape --max-patches 10

# Ingest into pgvector (chunk, embed, store)
python cli.py ingest --max-patches 10
```

### 4. Ask Questions

```bash
# Single question
python cli.py ask "What changes were made to Jinx?"

# Filter to a specific patch
python cli.py ask "What items were changed?" --patch 14.10

# Interactive chat mode
python cli.py chat
```

### 5. Run Evaluation

```bash
python cli.py evaluate
```

Results are saved to `data/eval_results/`.

## CLI Commands

| Command    | Description                                  |
|------------|----------------------------------------------|
| `scrape`   | Scrape patch notes from the LoL website      |
| `ingest`   | Full pipeline: scrape → chunk → embed → store|
| `ask`      | Ask a single question                        |
| `chat`     | Interactive question-answering session        |
| `evaluate` | Run RAGAS + custom metrics evaluation         |
| `status`   | Show ingested documents and chunk counts      |

## Project Structure

```
league_of_legends_rag/
├── cli.py                 # CLI entrypoint
├── docker-compose.yml     # PostgreSQL + pgvector
├── requirements.txt       # Python dependencies
├── .env.example           # Configuration template
├── db/
│   └── init.sql           # Database schema (auto-run on first start)
├── src/
│   ├── config.py          # Centralized configuration
│   ├── database.py        # pgvector operations
│   ├── scraper.py         # LoL patch notes web scraper
│   ├── chunking.py        # Section-aware text chunking
│   ├── embeddings.py      # Sentence-transformer embeddings
│   ├── llm.py             # TinyLlama-1.1B-Chat wrapper
│   ├── rag.py             # RAG query engine
│   └── evaluation.py      # RAGAS + custom evaluation
└── data/
    ├── raw/               # Cached scraped patch notes
    └── eval_results/      # Evaluation reports (JSON)
```

## Evaluation Metrics

### RAGAS (LLM-judged)
- **Faithfulness**: Does the answer only use facts from the context?
- **Answer Relevancy**: Is the answer relevant to the question?
- **Context Precision**: Are the retrieved chunks actually relevant?
- **Context Recall**: Did we retrieve all the info needed?

### Custom Metrics
- **Latency** (retrieval, generation, total)
- **Context Utilization**: Fraction of retrieved chunks used in the answer
- **Response Confidence**: Heuristic for hedging/uncertainty
- **Retrieval MRR**: Mean Reciprocal Rank of first relevant chunk

## Configuration

All settings are in `.env`. Key options:

| Variable           | Default                                   | Description                    |
|--------------------|-------------------------------------------|--------------------------------|
| `EMBEDDING_MODEL`  | `sentence-transformers/all-MiniLM-L6-v2`  | Embedding model (384 dims)     |
| `LLM_MODEL`        | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`     | Generation model               |
| `CHUNK_SIZE`       | `512`                                     | Max characters per chunk       |
| `CHUNK_OVERLAP`    | `64`                                      | Overlap between chunks         |
| `TOP_K`            | `5`                                       | Chunks retrieved per query     |
| `MAX_NEW_TOKENS`   | `512`                                     | Max generation length          |

## Experimentation Areas

This project is designed for experimenting with:

1. **Vector Storage**: Try different index types (HNSW vs IVFFlat), distance metrics (cosine vs L2), and index parameters (m, ef_construction).

2. **Chunking Strategies**: Modify `src/chunking.py` to test fixed-size vs semantic chunking, different overlap sizes, and section-aware splitting.

3. **Retrieval**: Adjust `TOP_K`, add re-ranking, try hybrid search (BM25 + vector), or metadata filtering.

4. **Generation**: Swap models, adjust temperature/top_p, modify the prompt template, or test different context window sizes.

5. **Evaluation**: Add custom questions, adjust RAGAS thresholds, or implement additional metrics.
