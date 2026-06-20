from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.chunker import Chunker
from src.embeddings.embedder import SentenceTransformerEmbedder
from src.indexing.index_manager import IndexManager
from src.ingestion.pdf_loader import PdfLoader
from src.ingestion.text_cleaner import TextCleaner


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a dense FAISS index from paper PDFs.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/papers"))
    parser.add_argument("--index-path", type=Path, default=Path("data/index/faiss.index"))
    parser.add_argument("--ids-path", type=Path, default=Path("data/index/ids.txt"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/index/chunks.jsonl"))
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    loader = PdfLoader()
    cleaner = TextCleaner()
    chunker = Chunker()
    chunks = []

    for pdf_path in sorted(args.input_dir.glob("*.pdf")):
        paper_id = pdf_path.stem
        text = cleaner.clean(loader.load_text(pdf_path))
        chunks.extend(chunker.split(paper_id, text, chunk_size=args.chunk_size))

    if not chunks:
        raise SystemExit(f"No PDF chunks found in {args.input_dir}")

    embedder = SentenceTransformerEmbedder(model_name=args.model)
    manager = IndexManager(
        index_path=args.index_path,
        ids_path=args.ids_path,
        chunks_path=args.chunks_path,
    )
    manager.build(chunks, embedder, batch_size=args.batch_size)
    print(f"Indexed {len(chunks)} chunks from {args.input_dir}")


if __name__ == "__main__":
    main()
