from __future__ import annotations

import atexit
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
import time
import threading

from src.chunking.chunker import Chunker
from src.crawler.downloader import Downloader
from src.domain.paper import Paper
from src.embeddings.interfaces.embedding_service import EmbeddingService
from src.embeddings.embedder import SentenceTransformerEmbedder
from src.generation.generator import Generator
from src.generation.llm_client import ExtractiveLLMClient, LocalFinetunedLLMClient
from src.generation.prompt_builder import PromptBuilder
from src.indexing.index_manager import IndexManager
from src.ingestion.pdf_loader import PdfLoader
from src.ingestion.text_cleaner import TextCleaner
from src.retrieval.interfaces.base_retriever import BaseRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.cache import RetrievalCache


@dataclass(slots=True)
class TopicSession:
    topic: str
    session_dir: Path
    papers: list[Paper]
    pdf_paths: list[Path]
    chunks_count: int
    retriever: BaseRetriever
    generator: Generator


class RagSessionManager:
    def __init__(
        self,
        *,
        base_dir: Path = Path("data/sessions"),
        model_name: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 1000,
        batch_size: int = 32,
        download_delay_seconds: float = 3.0,
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    ) -> None:
        self.base_dir = base_dir
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.download_delay_seconds = download_delay_seconds
        self.base_model_name = base_model_name

        self.active_session: TopicSession | None = None
        self._embedder: SentenceTransformerEmbedder | None = None
        
        self.downloaded_files: list[Path] = []
        atexit.register(self.cleanup)

        self.status = {
            "stage": "idle",
            "message": "Ready",
            "current": 0,
            "total": 0,
            "summary": None,
        }

        # Dynamic retrieval settings
        self.settings = {
            "hybrid_enabled": True,
            "dense_top_k": 5,
            "sparse_top_k": 5,
            "fusion_top_k": 5,
            "rrf_k": 60,
            "dense_weight": 1.0,
            "sparse_weight": 1.0,
        }

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def start_background(self, topic: str, papers_metadata: list[dict], *, base_model_name: str | None = None) -> None:
        """Starts the pipeline in the background thread."""
        clean_topic = " ".join(topic.split())
        if not clean_topic:
            raise ValueError("topic must not be empty")

        if base_model_name:
            self.base_model_name = base_model_name

        self.status.update({
            "stage": "started",
            "message": f'Started Knowledge Base build for "{clean_topic}"',
            "current": 0,
            "total": len(papers_metadata),
            "summary": None,
        })

        thread = threading.Thread(
            target=self._run_session,
            args=(clean_topic, papers_metadata),
            daemon=True,
        )
        thread.start()

    def get_status(self) -> dict:
        return dict(self.status)

    def get_active(self) -> TopicSession | None:
        return self.active_session

    def cleanup(self) -> None:
        """Cleanup newly downloaded PDF files on app shutdown."""
        if not self.downloaded_files:
            return

        print(f"\n[CLEANUP] Deleting {len(self.downloaded_files)} temporary PDFs...")
        for path in self.downloaded_files:
            try:
                if path.exists():
                    path.unlink()
                    print(f"[CLEANUP] Deleted file: {path.name}")
            except Exception as exc:
                print(f"[CLEANUP] Error deleting file {path}: {exc}")

        dirs_to_check = set(path.parent for path in self.downloaded_files)
        for parent in dirs_to_check:
            try:
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
                    print(f"[CLEANUP] Deleted empty directory: {parent}")
                    
                    grandparent = parent.parent
                    if grandparent.exists() and not any(grandparent.iterdir()):
                        grandparent.rmdir()
                        print(f"[CLEANUP] Deleted empty session directory: {grandparent}")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # INTERNAL PIPELINE (BACKGROUND RUNNER)
    # ------------------------------------------------------------------

    def _run_session(self, clean_topic: str, papers_metadata: list[dict]) -> None:
        try:
            start_time = time.time()
            session_dir = self.base_dir / self._slugify(clean_topic)
            papers_dir = session_dir / "papers"
            index_dir = session_dir / "index"

            papers_file = session_dir / "papers.jsonl"
            faiss_index_file = index_dir / "faiss.index"
            chunks_file = index_dir / "chunks.jsonl"

            # 1. Parse and build Paper objects
            papers = []
            for p in papers_metadata:
                papers.append(Paper(
                    paper_id=p["paper_id"],
                    title=p["title"],
                    authors=p["authors"],
                    abstract=p["abstract"],
                    source_url=p.get("pdf_url"),
                    published_at=p.get("published_year"),
                    metadata=p,
                ))

            # 2. Check if cache exists
            if (
                papers_file.exists()
                and faiss_index_file.exists()
                and chunks_file.exists()
            ):
                self.status.update({
                    "stage": "loading",
                    "message": "Loading hybrid vector indexes...",
                    "current": 4,
                    "total": 5,
                })
                
                cached_papers = self._load_papers(papers_file)
                pdf_paths = list(papers_dir.glob("*.pdf"))
                
                manager = IndexManager(
                    index_path=faiss_index_file,
                    sparse_index_path=index_dir / "bm25.pkl",
                    ids_path=index_dir / "ids.txt",
                    chunks_path=chunks_file,
                )
                index, stored_chunks, bm25_instance = manager.load_hybrid()

                dense_retriever = DenseRetriever(self._get_embedder(), index, stored_chunks)
                sparse_retriever = BM25Retriever(stored_chunks, bm25_instance)
                from src.retrieval.fusion.rrf import ReciprocalRankFusion
                retriever = HybridRetriever(
                    dense_retriever,
                    sparse_retriever,
                    fusion_strategy=ReciprocalRankFusion(
                        k=self.settings["rrf_k"],
                        weights=(self.settings.get("dense_weight", 1.0), self.settings.get("sparse_weight", 1.0))
                    )
                )

                session = TopicSession(
                    topic=clean_topic,
                    session_dir=session_dir,
                    papers=cached_papers,
                    pdf_paths=pdf_paths,
                    chunks_count=len(stored_chunks),
                    retriever=retriever,
                    generator=Generator(
                        LocalFinetunedLLMClient(self.base_model_name, self.base_model_name),
                        PromptBuilder(),
                    ),
                )
                self.active_session = session
                # Calculate index size for cached files
                index_size = 0
                if (index_dir / "faiss.index").exists():
                    index_size += (index_dir / "faiss.index").stat().st_size
                if (index_dir / "bm25.pkl").exists():
                    index_size += (index_dir / "bm25.pkl").stat().st_size

                summary_info = {
                    "papers_imported": len(cached_papers),
                    "chunks_generated": len(stored_chunks),
                    "embedding_time_seconds": 0.0,
                    "index_size_bytes": index_size,
                    "total_time_seconds": round(time.time() - start_time, 2),
                }

                self.status.update({
                    "stage": "ready",
                    "message": "Ready",
                    "current": len(stored_chunks),
                    "total": len(stored_chunks),
                    "summary": summary_info,
                })
                return

            # Cache does not exist, run builder pipeline
            session_dir.mkdir(parents=True, exist_ok=True)
            papers_dir.mkdir(parents=True, exist_ok=True)
            index_dir.mkdir(parents=True, exist_ok=True)

            self._save_papers(papers_file, papers)

            # Stage 1: Download
            pdf_paths = self._download_pdfs(papers, papers_dir)
            if not pdf_paths:
                raise ValueError("No PDF papers could be downloaded.")

            # Stage 2: Chunking
            chunks = self._build_chunks(pdf_paths, papers)
            if not chunks:
                raise ValueError("No text chunks extracted")

            manager = IndexManager(
                index_path=index_dir / "faiss.index",
                sparse_index_path=index_dir / "bm25.pkl",
                ids_path=index_dir / "ids.txt",
                chunks_path=index_dir / "chunks.jsonl",
            )

            # Stage 3 & 4: Embedding & Indexing
            self.status.update({
                "stage": "embedding",
                "message": "Creating embeddings and indexing...",
                "current": 0,
                "total": len(chunks),
            })

            def on_embedding_progress(current: int, total: int) -> None:
                self.status.update({
                    "stage": "embedding",
                    "message": f"Creating embeddings ({current}/{total} chunks)...",
                    "current": current,
                    "total": total,
                })

            embedding_start = time.time()
            index = manager.build(
                chunks,
                self._get_embedder(),
                batch_size=self.batch_size,
                progress_callback=on_embedding_progress,
            )
            embedding_time = time.time() - embedding_start

            # Load the hybrid indexes
            index, stored_chunks, bm25_instance = manager.load_hybrid()

            dense_retriever = DenseRetriever(self._get_embedder(), index, stored_chunks)
            sparse_retriever = BM25Retriever(stored_chunks, bm25_instance)
            from src.retrieval.fusion.rrf import ReciprocalRankFusion
            retriever = HybridRetriever(
                dense_retriever,
                sparse_retriever,
                fusion_strategy=ReciprocalRankFusion(
                    k=self.settings["rrf_k"],
                    weights=(self.settings.get("dense_weight", 1.0), self.settings.get("sparse_weight", 1.0))
                )
            )

            session = TopicSession(
                topic=clean_topic,
                session_dir=session_dir,
                papers=papers,
                pdf_paths=pdf_paths,
                chunks_count=len(chunks),
                retriever=retriever,
                generator=Generator(
                    LocalFinetunedLLMClient(self.base_model_name, self.base_model_name),
                    PromptBuilder(),
                ),
            )

            self.active_session = session
            self.downloaded_files.clear()  # Keep files for successful sessions

            # Calculate index size
            index_size = 0
            if (index_dir / "faiss.index").exists():
                index_size += (index_dir / "faiss.index").stat().st_size
            if (index_dir / "bm25.pkl").exists():
                index_size += (index_dir / "bm25.pkl").stat().st_size

            # Set summary card details
            summary_info = {
                "papers_imported": len(pdf_paths),
                "chunks_generated": len(chunks),
                "embedding_time_seconds": round(embedding_time, 2),
                "index_size_bytes": index_size,
                "total_time_seconds": round(time.time() - start_time, 2),
            }

            self.status.update({
                "stage": "ready",
                "message": "Ready",
                "current": len(chunks),
                "total": len(chunks),
                "summary": summary_info,
            })

        except Exception as exc:
            self.status.update({
                "stage": "error",
                "message": f"Error: {exc}",
            })

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _download_pdfs(self, papers: list[Paper], output_dir: Path) -> list[Path]:
        downloader = Downloader()
        paths: list[Path] = []

        for i, paper in enumerate(papers, start=1):
            self.status.update({
                "stage": "downloading",
                "message": f"Downloading ({i}/{len(papers)}): {paper.title}",
                "current": i,
                "total": len(papers),
            })

            pdf_path = output_dir / f"{paper.paper_id}.pdf"
            
            # Check local cache first to avoid duplicate downloads
            if pdf_path.exists():
                paths.append(pdf_path)
                continue

            pdf_url = paper.metadata.get("pdf_url")
            if not pdf_url:
                continue

            try:
                def on_download_progress(msg: str) -> None:
                    self.status.update({
                        "stage": "downloading",
                        "message": f"Downloading ({i}/{len(papers)}): {paper.title} - {msg}",
                        "current": i,
                        "total": len(papers),
                    })

                pdf_file = downloader.download(pdf_url, output_dir, progress_callback=on_download_progress)
                paths.append(pdf_file)
                self.downloaded_files.append(pdf_file)
                time.sleep(self.download_delay_seconds)
            except Exception:
                continue

        return paths

    def _build_chunks(self, pdf_paths: list[Path], papers: list[Paper]) -> list[Chunk]:
        from src.chunking.chunk_builder import ChunkBuilder
        loader = PdfLoader()
        cleaner = TextCleaner()
        chunk_builder = ChunkBuilder(chunk_size=self.chunk_size)
        paper_by_id = {paper.paper_id: paper for paper in papers}
        chunks = []

        total_pdfs = len(pdf_paths)
        for i, pdf_path in enumerate(pdf_paths, start=1):
            paper_id = pdf_path.stem
            paper = paper_by_id.get(paper_id)
            title = paper.title if paper else paper_id

            self.status.update({
                "stage": "parsing",
                "message": f"Parsing and chunking ({i}/{total_pdfs}): {title}",
                "current": i,
                "total": total_pdfs,
            })

            try:
                text = cleaner.clean(loader.load_text(pdf_path))
            except Exception:
                continue

            if not text:
                continue

            if paper:
                paper_chunks = chunk_builder.build_chunks(paper, text)
                chunks.extend(paper_chunks)
            else:
                from src.chunking.chunker import Chunker
                paper_chunks = Chunker().split(paper_id, text, chunk_size=self.chunk_size)
                chunks.extend(paper_chunks)

        return chunks

    def _get_embedder(self) -> SentenceTransformerEmbedder:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(model_name=self.model_name)
        return self._embedder

    @staticmethod
    def _save_papers(path: Path, papers: list[Paper]) -> None:
        path.write_text(
            "\n".join(json.dumps(asdict(paper), ensure_ascii=True) for paper in papers),
            encoding="utf-8",
        )

    @staticmethod
    def _load_papers(path: Path) -> list[Paper]:
        papers = []
        if not path.exists():
            return papers
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    papers.append(Paper(
                        paper_id=data["paper_id"],
                        title=data["title"],
                        authors=data["authors"],
                        abstract=data["abstract"],
                        source_url=data.get("source_url"),
                        published_at=data.get("published_at"),
                        metadata=data.get("metadata", {}),
                    ))
                except Exception:
                    continue
        return papers

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
        return slug[:80] or "topic"
