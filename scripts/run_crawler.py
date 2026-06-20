from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crawler.paper_finder import PaperFinder


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch IR papers from arXiv.")
    parser.add_argument("query", nargs="?", default="information retrieval")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("data/arxiv/papers.jsonl"))
    args = parser.parse_args()

    papers = PaperFinder().find(args.query, max_results=args.max_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(asdict(paper), ensure_ascii=True) for paper in papers),
        encoding="utf-8",
    )
    print(f"Wrote {len(papers)} papers to {args.output}")


if __name__ == "__main__":
    main()
