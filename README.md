# IR Research Assistant 🔍🤖

A local, GPU-accelerated Retrieval-Augmented Generation (RAG) assistant designed to search, crawl, ingest, index, and fine-tune Large Language Models (LLMs) on arXiv research papers.

This application runs entirely locally, utilizing **PyTorch (CUDA)** to accelerate vector embeddings creation (via FAISS) and run **LoRA fine-tuning** on consumer GPUs (e.g., NVIDIA RTX series) to adapt models to specific research domains on-the-fly.

---

## 🌟 Key Features

*   **Real-Time arXiv Crawling & Extraction**: Searches and crawls academic papers by topic, handles rate limits (`429`) with automatic exponential backoff, and extracts PDFs locally.
*   **Intelligent Text Chunking & Ingestion**: Cleans extracted PDF text and chunks it using semantic boundaries while preserving paper metadata (title, authors, abstract, URL).
*   **FAISS Vector Indexing**: Encodes chunks using `SentenceTransformers` and indexes them into a local FAISS database for dense semantic retrieval.
*   **Local GPU-Accelerated Fine-Tuning**: Trains a local LLM (such as `Qwen2.5-0.5B-Instruct` or `TinyLlama`) on the indexed document chunks using **PEFT / LoRA** adapters.
*   **Conversational RAG Chatbot**: Features a sleek, modern dark-themed web dashboard with live scrolling logs, interactive progress bars, and a question-answering chat interface.
*   **Session Caching & Reusability**: Bypasses indexing and training stages to instantly load completed sessions in under a second on subsequent starts.
*   **Selective Resource Cleanup**: Automatically removes partial download artifacts if aborted (e.g., via `Ctrl+C` mid-run), while retaining successful sessions permanently on disk.

---

## 🛠️ Project Structure

*   `src/domain`: Core data structures (e.g., `Paper`, `Chunk`).
*   `src/crawler`: Discovering and downloading papers using the arXiv API.
*   `src/ingestion`: PDF parsing, text normalization, and cleaner pipelines.
*   `src/chunking`: Text splitters and section partitioners.
*   `src/embeddings`: Sentence transformer wrapper models.
*   `src/indexing`: Vector database and ID index management (FAISS).
*   `src/retrieval`: Dense semantic search and retrieval models.
*   `src/generation`: Custom local LLM prompt builders and ChatML inference clients.
*   `src/generation/finetune.py`: LoRA adapters SFT training loop.
*   `src/api`: FastAPI endpoints and static web UI client dashboard.
*   `tests`: Unit tests for retrieval, chunking, crawling, and API controllers.

---

## ⚙️ Prerequisites & Environment Setup

This project requires **Python 3.11** or **Python 3.13** inside a Conda environment.

### 1. Enable GPU Acceleration (Recommended)
If you have an NVIDIA GPU (e.g., RTX 4050 Laptop GPU or higher), install the CUDA 12.4 enabled PyTorch build to utilize GPU speed:

```bash
# Activate your environment
conda activate ir-rag

# Force-reinstall PyTorch with CUDA 12.4 support
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### 2. Install Dependencies
Install all package requirements, including Hugging Face components and PEFT:

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

The application will be running locally at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### 2. Prepare a Topic
1.  Open the web dashboard in your browser.
2.  Input a research topic (e.g., `dense retrieval` or `neural reranking`).
3.  Choose the base LLM model (e.g., **Qwen 2.5 0.5B**).
4.  Set your paper target and click **Prepare**.
5.  Watch the console logs and progress indicators update in real-time as the RAG index builds and the model is fine-tuned.
6.  Once training hits `100%`, query the chatbot directly in the text input box!

---

## 🧪 Running Tests

Ensure all features are healthy and verified by running the test suite:

```bash
python -m pytest
```
