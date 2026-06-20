class Deduplicator:
    def deduplicate(self, papers: list[dict]) -> list[dict]:
        seen: set[str] = set()
        unique: list[dict] = []

        for paper in papers:
            key = paper.get("paper_id") or paper.get("id") or paper.get("title")
            if key in seen:
                continue
            seen.add(key)
            unique.append(paper)

        return unique
