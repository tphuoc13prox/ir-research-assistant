from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
import time

from src.chunking.chunker import Chunker
from src.crawler.arxiv_client import ArxivClient
from src.crawler.downloader import Downloader
from src.domain.paper import Paper
from src.embeddings.embedder import SentenceTransformerEmbedder
from src.generation.generator import Generator
from src.generation.llm_client import ExtractiveLLMClient
from src.generation.prompt_builder import PromptBuilder
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
        download_delay_seconds: float = 0.5,
    ) -> None:
        self.base_dir = base_dir
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.download_delay_seconds = download_delay_seconds
        self.active_session: TopicSession | None = None
        self._embedder: SentenceTransformerEmbedder | None = None

    def start(self, topic: str, *, max_papers: int = 100) -> TopicSession:
        clean_topic = " ".join(topic.split())
        if not clean_topic:
            raise ValueError("topic must not be empty")

        session_dir = self.base_dir / self._slugify(clean_topic)
        papers_dir = session_dir / "papers"
        index_dir = session_dir / "index"
        session_dir.mkdir(parents=True, exist_ok=True)
        papers_dir.mkdir(parents=True, exist_ok=True)

        papers = ArxivClient().search_ir_papers(clean_topic, max_results=max_papers)
        self._save_papers(session_dir / "papers.jsonl", papers)

        pdf_paths = self._download_pdfs(papers, papers_dir)
        chunks = self._build_chunks(pdf_paths, papers)
        if not chunks:
            raise ValueError("No text chunks could be extracted from downloaded PDFs")

        manager = IndexManager(
            index_path=index_dir / "faiss.index",
            ids_path=index_dir / "ids.txt",
            chunks_path=index_dir / "chunks.jsonl",
        )
        index = manager.build(chunks, self._get_embedder(), batch_size=self.batch_size)
        _, stored_chunks = manager.load()

        session = TopicSession(
            topic=clean_topic,
            session_dir=session_dir,
            papers=papers,
            pdf_paths=pdf_paths,
            chunks_count=len(chunks),
            retriever=DenseRetriever(self._get_embedder(), index, stored_chunks),
            generator=Generator(ExtractiveLLMClient(), PromptBuilder()),
        )
        self.active_session = session
        return session

    def get_active(self) -> TopicSession | None:
        return self.active_session

    def _download_pdfs(self, papers: list[Paper], output_dir: Path) -> list[Path]:
        downloader = Downloader()
        paths: list[Path] = []
        for paper in papers:
            pdf_url = paper.metadata.get("pdf_url")
            if not pdf_url:
                continue
            try:
                paths.append(downloader.download(pdf_url, output_dir))
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

        for pdf_path in pdf_paths:
            paper_id = pdf_path.stem
            paper = paper_by_id.get(paper_id)
            try:
                text = cleaner.clean(loader.load_text(pdf_path))
            except Exception:
                continue
            if not text:
                continue
            for chunk in chunker.split(paper_id, text, chunk_size=self.chunk_size):
                if paper is not None:
                    chunk.metadata.update(
                        {
                            "title": paper.title,
                            "source_url": paper.source_url or "",
                            "published_at": paper.published_at or "",
                        }
                    )
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
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
        return slug[:80] or "topic"
