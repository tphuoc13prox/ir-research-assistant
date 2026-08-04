from src.domain.chunk import Chunk
from src.domain.paper import Paper
from src.chunking.chunker import Chunker

class ChunkBuilder:
    def __init__(self, chunk_size: int = 1000) -> None:
        self.chunker = Chunker()
        self.chunk_size = chunk_size

    def build_chunks(self, paper: Paper, text: str) -> list[Chunk]:
        chunks = self.chunker.split(paper.paper_id, text, self.chunk_size)
        
        # Get keywords/concepts/categories from paper metadata
        keywords = paper.metadata.get("keywords", [])
        if not keywords:
            keywords = paper.metadata.get("concepts", [])
        if not keywords:
            keywords = paper.metadata.get("categories", [])
            
        # Ensure it's a list or structure
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        elif not isinstance(keywords, list):
            keywords = []

        for chunk in chunks:
            chunk.metadata.update({
                "title": paper.title,
                "abstract": paper.abstract or "",
                "source_url": paper.source_url or "",
                "published_at": paper.published_at or "",
                "year": paper.published_at.split("-")[0] if paper.published_at else "N/A",
                "keywords": keywords,
            })
        return chunks
