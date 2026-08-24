# IR Research Assistant (Level 3 Upgrade) 🔍🤖

A local, production-quality hybrid (Dense + Lexical) information retrieval system designed to discover, crawl, ingest, and index research papers with intelligent query understanding and intent-aware ranking.

This repository implements a highly decoupled retrieval pipeline using an **Embedding Service**, a **Paper Discovery Provider** interface, a parallel execution engine, **Reciprocal Rank Fusion (RRF)**, **Field-Aware Ranking (Level 2)**, and **Intelligent Query Understanding & Intent-Aware Retrieval (Level 3)**.

---

## 🗺️ Project Roadmap

- **Phase 1: Intelligent Hybrid Retrieval Engine (Current Status - Level 3 Complete)**
  - Stable paper metadata discovery across `cs.IR`, `cs.CL`, and `cs.LG` categories.
  - Concurrent dense (FAISS) + sparse (BM25) vector index building.
  - Reciprocal Rank Fusion (RRF) and Field-Aware Ranking (Level 2).
  - Intelligent Query Parsing, Metadata Filtering, and Intent-Aware Re-ranking (Level 3).
  - Dynamic Relevance Threshold filtering & 30-result ceiling control.
  - Browser-heartbeat-monitored automatic server shutdown and temporary index directory cleanup.
- **Phase 2: Research Workspace (Future Phase)**
  - Introduces selected paper workspaces, interactive document annotation tools, and local LLM document synthesis.
- **Phase 3: Research Copilot (Future Phase)**
  - Implements multi-document reasoning agents, adaptive flashcards creator, and automatic personalized learning roadmaps.

---

## 🌟 Advanced Level 3 Retrieval Pipeline

The search and retrieval flow is fully decoupled to ensure optimal information retrieval correctness:

$$\text{Query} \rightarrow \text{Query Parser} \rightarrow \text{Candidate Retrieval} \rightarrow \text{Metadata Filter} \rightarrow \text{Field-Aware Ranker} \rightarrow \text{Intent-Aware Ranker} \rightarrow \text{Score Combiner} \rightarrow \text{LLM}$$

### 1. Stage 1: Metadata Paper Discovery & Ingestion
- **Discovery Search**: Input a query topic. The crawler queries arXiv (broadened to `cs.IR`, `cs.CL`, and `cs.LG` categories) to find relevant works.
- **Temporary Cache Ingestion**: Discovered papers are checked out. The downloader pulls fresh PDFs, parses text, constructs chunks, computes embeddings, and builds hybrid indices inside temporary workspace directories.
- **Cleanup on Shutdown**: Once you close the browser tab, the server automatically shuts down via connection heartbeat within 8 seconds and recursively deletes all temporary index files and downloaded PDFs.

### 2. Stage 2: Intent-Aware Retrieval & Ranking
- **Query Parser**: Analyzes user questions to extract year constraints (e.g. `before 2026`, `between 2020 and 2025`), intent categories (`survey`, `comparison`, `dataset`), boost terms, and exclusion words (e.g. `-cnn`).
- **Metadata Filtering**: Discards non-matching document chunks before mathematical scoring to optimize CPU cycles.
- **Intent-Aware Re-ranking**: Adjusts ranking scores dynamically using keyword focus weights and title/abstract boosts.
- **Dynamic Relevance Threshold**: Users can adjust a relevance score threshold in the settings panel to dynamically filter out low-relevance results (up to a maximum cap of 30 results).

---

## 🛠️ Project Structure

*   `src/query_understanding`: Stateless regular-expression **Query Parser** extracting intent and metadata constraints.
*   `src/retrieval/filtering`: Stateless **Metadata Filter** applying hard constraints on candidate lists.
*   `src/retrieval/ranking`: **Field-Aware & Intent-Aware Ranker** containing re-ranking logic, score combiners, and presentation-decoupled explainability generators.
*   `src/paper_discovery`: Discovery layer defining `PaperProvider` abstractions and `ArxivProvider` implementation.
*   `src/embeddings`: Decoupled vector embedding interface `EmbeddingService` and `SentenceTransformer` implementation.
*   `src/indexing`: Builders (`DenseIndexBuilder`, `SparseIndexBuilder`) and `IndexManager` for persistent vector database compilation.
*   `src/retrieval`: Parallel retrieval handlers (`DenseRetriever`, `BM25Retriever`, `HybridRetriever`) and `ReciprocalRankFusion` strategy.
*   `src/domain`: Dataclass domain representations (`Query`, `RetrievalResult`, `RankingFeatures`).
*   `src/crawler`: PDF downloading and rate-limiting backoff.
*   `src/chunking`: Recursive splitters and text segment division.
*   `src/api`: Web clients, FastAPI routes, session managers, and heartbeat shutdown monitor.
*   `src/evaluation`: Benchmark metric calculation runner (`Recall`, `MRR`, `nDCG`).

---

## ⚙️ Prerequisites & Environment Setup

This project requires **Python 3.11** or **Python 3.13** inside a Conda environment.

### 1. Enable GPU Acceleration (Recommended)
If you have an NVIDIA GPU, install the CUDA 12.4 enabled PyTorch build to utilize GPU speed:

```bash
# Activate your environment
conda activate ir-rag

# Force-reinstall PyTorch with CUDA 12.4 support
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### 2. Install Dependencies
Install all package requirements:

```bash
pip install -r requirements.txt
pip install peft accelerate
```

---

## 🚀 Getting Started

### 1. Launch the Server
To spin up the FastAPI application and local static client, run the batch script:

```bash
.\run.bat
```

The application will launch and be available locally at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### 2. Run the Benchmark Evaluation Suite
Measure metrics across Dense, Lexical, and RRF Hybrid models using the evaluation script:

```bash
python -c "from src.evaluation.benchmark_runner import BenchmarkRunner; print(BenchmarkRunner().run())"
```

### 3. Running Unit Tests
```bash
python -m pytest
```
