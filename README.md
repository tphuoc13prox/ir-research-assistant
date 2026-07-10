# IR Research Assistant 🔍🤖

A local, GPU-accelerated Retrieval-Augmented Generation (RAG) assistant designed to search, crawl, ingest, index, and fine-tune Large Language Models (LLMs) on arXiv research papers.

This application runs entirely locally, utilizing **PyTorch (CUDA)** to accelerate vector embeddings creation (via FAISS) and run **LoRA fine-tuning** on consumer GPUs (e.g., NVIDIA RTX series) to adapt models to specific research domains on-the-fly.

---

## 🌟 Two-Stage Research Workflow

The project is structured into two distinct operational stages:

### 1. Stage 1: Automated 10-Paper Search & Indexing
- **Fixed Ingestion Target**: The user inputs a research topic. The system automatically queries the arXiv API for **exactly 10 most relevant papers** (crawling limit is locked for stability).
- **Processing pipeline**: It downloads the PDFs, chunks the extracted text, builds a local FAISS dense vector index, and automatically fine-tunes a local LLM (such as `Qwen 2.5 0.5B`) using LoRA adapters on the indexed chunks.

### 2. Stage 2: Interactive Sidebar & Click-to-Summarize Chatbot
- **Slide-in Workspace**: Once the index and SFT tuning are completed, a responsive side-by-side dashboard reveals a sidebar showing the 10 downloaded paper titles and authors on the left, and the chatbot interface on the right.
- **Instant Summaries**: Clicking any paper card triggers the chatbot to immediately generate and print a structured summary of that paper (covering **Core Objective**, **Methodology**, and **Results**) using the fine-tuned local LLM.
- **RAG QA Chat**: You can still converse with the assistant about the entire collection of papers using the text box at the bottom.

---

## 🛠️ Project Structure

*   `src/domain`: Core data structures (e.g., `Paper`, `Chunk`).
*   `src/crawler`: Discovering and downloading papers using the arXiv API.
*   `src/ingestion`: PDF parsing, text normalization, and cleaning.
*   `src/chunking`: Text splitters and section partitioners.
*   `src/embeddings`: Sentence transformer wrapper models.
*   `src/indexing`: Vector database and ID index management (FAISS).
*   `src/retrieval`: Dense semantic search and retrieval models.
*   `src/generation`: Custom local LLM prompt builders and ChatML inference clients.
*   `src/generation/finetune.py`: SFT training loop via PEFT/LoRA.
*   `src/api`: FastAPI endpoints, no-cache resource middleware, and web client.
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

The application will launch and be available locally at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### 2. Prepare a Topic
1.  Open the web dashboard in your browser.
2.  Input a research topic (e.g., `dense retrieval` or `neural reranking`).
3.  Choose the base LLM model (e.g., **Qwen 2.5 0.5B**) and click **Prepare**.
4.  Watch the progress indicators update in real-time as the RAG index builds and the model is fine-tuned.
5.  Once training hits `100%`, click on any paper card in the left sidebar to generate a structured summary, or ask questions in the text input box!

---

## 🧪 Running Tests

Ensure all features are healthy and verified by running the test suite:

```bash
python -m pytest
```
