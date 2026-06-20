# IR Research Assistant

Project skeleton for an information retrieval research assistant focused on paper crawling, ingestion, chunking, embedding, indexing, retrieval, generation, and evaluation.

## Structure

- `src/domain`: core data models.
- `src/crawler`: paper discovery and download pipeline.
- `src/ingestion`: PDF loading, text cleaning, and metadata extraction.
- `src/chunking`: document section parsing and chunking.
- `src/embeddings`: embedding model wrappers and batch encoding.
- `src/indexing`: vector index and metadata storage.
- `src/retrieval`: dense, BM25, hybrid retrieval, and reranking.
- `src/generation`: prompt construction and LLM response generation.
- `src/evaluation`: retrieval/ranking metrics and benchmark utilities.
- `src/api`: FastAPI application.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_server.py
```

The API exposes a health check at `GET /health`.
