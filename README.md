# IR Research Assistant (Phase 1) 🔍🤖

A local, production-quality hybrid (Dense + Lexical) information retrieval system designed to discover, crawl, ingest, and index research papers. 

This project forms the technical foundation for future phases. In Phase 1, it implements a highly decoupled retrieval pipeline using an **Embedding Service**, a **Paper Discovery Provider** interface, a parallel execution engine, **Reciprocal Rank Fusion (RRF)**, and a developer telemetric debug console.

---

## 🗺️ Project Roadmap

- **Phase 1: Hybrid Retrieval Engine (Current Phase)**
  - Infrastructure-only implementation focusing on stable paper metadata discovery, concurrent dense (FAISS) + sparse (BM25) vector index building, Reciprocal Rank Fusion ranking, retrieval caching, and telemetry logging.
- **Phase 2: Research Workspace (Future Phase)**
  - Introduces selected paper workspaces, interactive document annotation tools, and local LLM document synthesis capabilities.
- **Phase 3: Research Copilot (Future Phase)**
  - Implements multi-document reasoning agents, adaptive flashcards creator, and automatic personalized learning roadmaps.

---

## 🌟 Two-Stage Selected Paper Workflow

The system separates discovery concerns from ingestion index building:

### 1. Stage 1: Metadata Paper Discovery
- **Discovery Search**: Input a query topic. The system calls the `DiscoveryService` to pull matching paper records (titles, authors, categories, abstracts, publish dates) from selected repositories without downloading any files.
- **Curation Selection**: Discovered papers are presented in a selection table. Use checkboxes (with **Select All** and **Clear Selection** helpers) to pick which exact works to include.

### 2. Stage 2: Knowledge Base Builder
- **Checks Cache**: When you click **Import Selected**, the system begins sequential ingestion. It first checks the local directories to see if files already exist on disk to bypass redundant network requests.
- **Stage Checklist**: Watch the progress checklist update in real-time through *Downloading PDFs*, *Parsing text*, *Chunking*, and *Embedding indices creation*.
- **Summary Card**: Upon completion, a summary card shows build statistics: imported papers, generated chunks count, vector encoding times, and index space usage.
- **Dashboard Workspace**: Ask questions in the chatbot. Use the sidebar status badges to monitor ingestion, customize RRF penalty parameters, inspect retrieval latency metrics, and read intermediate ranked lists.

---

## 🛠️ Project Structure

*   `src/paper_discovery`: Isolated discovery layer defining `PaperProvider` abstractions and `ArxivProvider` implementation.
*   `src/embeddings`: Decooupled vector embedding interface `EmbeddingService` and `SentenceTransformer` implementation.
*   `src/indexing`: Builders (`DenseIndexBuilder`, `SparseIndexBuilder`) and `IndexManager` for persistent vector database compilation.
*   `src/retrieval`: Parallel retrieval handlers (`DenseRetriever`, `BM25Retriever`, `HybridRetriever`) and `ReciprocalRankFusion` strategy.
*   `src/retrieval/ranking`: **Field-Aware Ranking Stage (Level 2)** containing `FieldAwareRanker`, `ScoreCombiner`, and `RankingConfig` configuration parser.
*   `src/domain`: Base query models and result representations (`Query`, `RetrievalResult`, `RankingFeatures`).
*   `src/crawler`: PDF downloading and HTTP rate-limiting backoff.
*   `src/ingestion`: PDF parsing, structures cleanups, and layout extraction.
*   `src/chunking`: Recursive splitters and text segment division including `ChunkBuilder` metadata enricher.
*   `src/api`: Web clients, FastAPI routes, and telemetry.
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
