from src.domain.chunk import Chunk


class Chunker:
    def split(self, paper_id: str, text: str, chunk_size: int = 1000) -> list[Chunk]:
        chunks: list[Chunk] = []
        for index, start in enumerate(range(0, len(text), chunk_size)):
            end = min(start + chunk_size, len(text))
            chunks.append(
                Chunk(
                    chunk_id=f"{paper_id}:{index}",
                    paper_id=paper_id,
                    text=text[start:end],
                    start_char=start,
                    end_char=end,
                )
            )
        return chunks
