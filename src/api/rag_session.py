from __future__ import annotations

import atexit
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
import time
import threading

from src.chunking.chunker import Chunker
from src.crawler.arxiv_client import ArxivClient
from src.crawler.downloader import Downloader
from src.domain.paper import Paper
from src.embeddings.embedder import SentenceTransformerEmbedder
from src.generation.generator import Generator
from src.generation.llm_client import ExtractiveLLMClient, LocalFinetunedLLMClient
from src.generation.prompt_builder import PromptBuilder
from src.generation.finetune import run_finetuning
from src.indexing.index_manager import IndexManager
from src.ingestion.pdf_loader import PdfLoader
from src.ingestion.text_cleaner import TextCleaner
from src.retrieval.dense_retriever import DenseRetriever


@dataclass(slots=True)
class TopicSession:
    topic: str
    session_dir: Path
    papers: list[Paper]
    pdf_paths: list[Path]
    chunks_count: int
    retriever: DenseRetriever
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
        }

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def start_background(self, topic: str, *, max_papers: int = 100, base_model_name: str | None = None) -> None:
        """Khởi động pipeline trong background thread, không block request."""
        clean_topic = " ".join(topic.split())
        if not clean_topic:
            raise ValueError("topic must not be empty")

        if base_model_name:
            self.base_model_name = base_model_name

        self.status.update({
            "stage": "started",
            "message": f'Started session for "{clean_topic}"',
            "current": 0,
            "total": max_papers,
        })

        thread = threading.Thread(
            target=self._run_session,
            args=(clean_topic, max_papers),
            daemon=True,
        )
        thread.start()

    def get_status(self) -> dict:
        return dict(self.status)

    def get_active(self) -> TopicSession | None:
        return self.active_session

    def cleanup(self) -> None:
        """Xóa sạch các file PDF mới được tải về trong phiên làm việc này khi tat chuong trinh."""
        if not self.downloaded_files:
            return

        print(f"\n[CLEANUP] Dang xoa {len(self.downloaded_files)} file paper moi duoc tai ve...")
        for path in self.downloaded_files:
            try:
                if path.exists():
                    path.unlink()
                    print(f"[CLEANUP] Da xoa file: {path.name}")
            except Exception as exc:
                print(f"[CLEANUP] Loi khi xoa file {path}: {exc}")

        # Xoa cac thu muc empty
        dirs_to_check = set(path.parent for path in self.downloaded_files)
        for parent in dirs_to_check:
            try:
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
                    print(f"[CLEANUP] Da xoa thu muc trong: {parent}")
                    
                    grandparent = parent.parent
                    if grandparent.exists() and not any(grandparent.iterdir()):
                        grandparent.rmdir()
                        print(f"[CLEANUP] Da xoa thu muc phien trong: {grandparent}")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # INTERNAL PIPELINE (CHẠY TRONG BACKGROUND)
    # ------------------------------------------------------------------

    def _run_session(self, clean_topic: str, max_papers: int) -> None:
        try:
            session_dir = self.base_dir / self._slugify(clean_topic)
            papers_dir = session_dir / "papers"
            index_dir = session_dir / "index"
            finetuned_model_dir = session_dir / "finetuned_model"

            # Check if cache exists
            papers_file = session_dir / "papers.jsonl"
            faiss_index_file = index_dir / "faiss.index"
            chunks_file = index_dir / "chunks.jsonl"
            model_adapter_file = finetuned_model_dir / "adapter_config.json"

            if (
                papers_file.exists()
                and faiss_index_file.exists()
                and chunks_file.exists()
                and model_adapter_file.exists()
            ):
                # Load existing session
                self.status.update({
                    "stage": "searching",
                    "message": "Found existing session. Loading papers...",
                    "current": 1,
                    "total": 5,
                })
                time.sleep(0.3)
                
                # Load papers
                papers = self._load_papers(papers_file)
                
                # Load pdf paths
                pdf_paths = list(papers_dir.glob("*.pdf"))

                self.status.update({
                    "stage": "embedding",
                    "message": "Loading FAISS vector index...",
                    "current": 3,
                    "total": 5,
                })
                time.sleep(0.3)
                
                manager = IndexManager(
                    index_path=faiss_index_file,
                    ids_path=index_dir / "ids.txt",
                    chunks_path=chunks_file,
                )
                index, stored_chunks = manager.load()

                self.status.update({
                    "stage": "finetuning",
                    "message": "Loading fine-tuned local LLM model...",
                    "current": 4,
                    "total": 5,
                })
                time.sleep(0.3)

                session = TopicSession(
                    topic=clean_topic,
                    session_dir=session_dir,
                    papers=papers,
                    pdf_paths=pdf_paths,
                    chunks_count=len(stored_chunks),
                    retriever=DenseRetriever(
                        self._get_embedder(),
                        index,
                        stored_chunks,
                    ),
                    generator=Generator(
                        LocalFinetunedLLMClient(finetuned_model_dir, self.base_model_name),
                        PromptBuilder(),
                    ),
                )

                self.active_session = session

                self.status.update({
                    "stage": "ready",
                    "message": "Ready",
                    "current": len(stored_chunks),
                    "total": len(stored_chunks),
                })
                return

            # Searching
            self.status.update({
                "stage": "searching",
                "message": "Searching arXiv papers...",
                "current": 0,
                "total": max_papers,
            })

            session_dir.mkdir(parents=True, exist_ok=True)
            papers_dir.mkdir(parents=True, exist_ok=True)

            def on_search_progress(msg: str) -> None:
                self.status.update({
                    "stage": "searching",
                    "message": msg,
                })

            papers = ArxivClient().search_ir_papers(
                clean_topic,
                max_results=max_papers,
                progress_callback=on_search_progress,
            )

            self.status.update({
                "stage": "search_complete",
                "message": f"Found {len(papers)} papers on arXiv",
                "current": len(papers),
                "total": len(papers),
            })

            self._save_papers(session_dir / "papers.jsonl", papers)

            # Download PDFs
            self.status.update({
                "stage": "downloading",
                "message": f"Downloading PDFs (0/{len(papers)})",
                "current": 0,
                "total": len(papers),
            })

            pdf_paths = self._download_pdfs(papers, papers_dir)

            # Chunking
            self.status.update({
                "stage": "chunking",
                "message": "Extracting and chunking PDFs...",
                "current": 0,
                "total": len(pdf_paths),
            })

            chunks = self._build_chunks(pdf_paths, papers)
            if not chunks:
                raise ValueError("No text chunks extracted")

            manager = IndexManager(
                index_path=index_dir / "faiss.index",
                ids_path=index_dir / "ids.txt",
                chunks_path=index_dir / "chunks.jsonl",
            )

            # Embedding
            self.status.update({
                "stage": "embedding",
                "message": "Creating embeddings...",
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

            index = manager.build(
                chunks,
                self._get_embedder(),
                batch_size=self.batch_size,
                progress_callback=on_embedding_progress,
            )

            _, stored_chunks = manager.load()

            # Fine-tuning local LLM
            self.status.update({
                "stage": "finetuning",
                "message": "Initializing local LLM fine-tuning...",
                "current": 0,
                "total": 10,
            })

            def on_finetune_progress(msg: str) -> None:
                current_step = 0
                if "step " in msg:
                    try:
                        step_part = msg.split("step ")[1].split(" ")[0]
                        current_step = int(step_part.split("/")[0])
                    except Exception:
                        pass
                
                self.status.update({
                    "stage": "finetuning",
                    "message": msg,
                    "current": current_step,
                    "total": 10,
                })

            finetuned_model_dir = session_dir / "finetuned_model"
            
            run_finetuning(
                base_model_name=self.base_model_name,
                chunks_path=index_dir / "chunks.jsonl",
                output_dir=finetuned_model_dir,
                progress_callback=on_finetune_progress,
            )

            session = TopicSession(
                topic=clean_topic,
                session_dir=session_dir,
                papers=papers,
                pdf_paths=pdf_paths,
                chunks_count=len(chunks),
                retriever=DenseRetriever(
                    self._get_embedder(),
                    index,
                    stored_chunks,
                ),
                generator=Generator(
                    LocalFinetunedLLMClient(finetuned_model_dir, self.base_model_name),
                    PromptBuilder(),
                ),
            )

            self.active_session = session
            # Clear downloaded files list to prevent auto-cleanup of this successful session on shutdown
            self.downloaded_files.clear()

            self.status.update({
                "stage": "ready",
                "message": "Ready",
                "current": len(chunks),
                "total": len(chunks),
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
            pdf_url = paper.metadata.get("pdf_url")
            if not pdf_url:
                continue

            self.status.update({
                "stage": "downloading",
                "message": f"Downloading ({i}/{len(papers)}): {paper.title}",
                "current": i,
                "total": len(papers),
            })

            try:
                def on_download_progress(msg: str) -> None:
                    self.status.update({
                        "stage": "downloading",
                        "message": f"Downloading ({i}/{len(papers)}): {paper.title} - {msg}",
                        "current": i,
                        "total": len(papers),
                    })

                pdf_path = downloader.download(pdf_url, output_dir, progress_callback=on_download_progress)
                paths.append(pdf_path)
                self.downloaded_files.append(pdf_path)
                time.sleep(self.download_delay_seconds)
            except Exception:
                continue

        return paths

    def _build_chunks(self, pdf_paths: list[Path], papers: list[Paper]):
        loader = PdfLoader()
        cleaner = TextCleaner()
        chunker = Chunker()
        paper_by_id = {paper.paper_id: paper for paper in papers}
        chunks = []

        total_pdfs = len(pdf_paths)
        for i, pdf_path in enumerate(pdf_paths, start=1):
            paper_id = pdf_path.stem
            paper = paper_by_id.get(paper_id)
            title = paper.title if paper else paper_id

            self.status.update({
                "stage": "chunking",
                "message": f"Chunking ({i}/{total_pdfs}): {title}",
                "current": i,
                "total": total_pdfs,
            })

            try:
                text = cleaner.clean(loader.load_text(pdf_path))
            except Exception:
                continue

            if not text:
                continue

            for chunk in chunker.split(paper_id, text, chunk_size=self.chunk_size):
                if paper:
                    chunk.metadata.update({
                        "title": paper.title,
                        "source_url": paper.source_url or "",
                        "published_at": paper.published_at or "",
                    })
                chunks.append(chunk)

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
