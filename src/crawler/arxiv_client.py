from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import httpx

from src.domain.paper import Paper


class ArxivClient:
    API_URL = "https://export.arxiv.org/api/query"
    NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    def __init__(self, *, timeout: float = 30.0, user_agent: str | None = None) -> None:
        self.timeout = timeout
        self.headers = {
            "User-Agent": user_agent
            or "ir-research-assistant/0.1 (mailto:research@example.com)"
        }

    def search(
        self,
        query: str,
        max_results: int = 10,
        *,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> list[dict[str, Any]]:
        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
            response = client.get(self.API_URL, params=params)
            response.raise_for_status()
        return self._parse_feed(response.text)

    def search_ir_papers(
        self,
        query: str,
        max_results: int = 10,
        *,
        category: str = "cs.IR",
    ) -> list[Paper]:
        arxiv_query = f"cat:{category} AND all:{query}" if query else f"cat:{category}"
        return [self._to_paper(item) for item in self.search(arxiv_query, max_results)]

    def _parse_feed(self, xml_text: str) -> list[dict[str, Any]]:
        root = ET.fromstring(xml_text)
        papers: list[dict[str, Any]] = []
        for entry in root.findall("atom:entry", self.NS):
            links = self._links(entry)
            papers.append(
                {
                    "paper_id": self._paper_id(self._text(entry, "atom:id")),
                    "title": self._clean(self._text(entry, "atom:title")),
                    "authors": [
                        self._text(author, "atom:name")
                        for author in entry.findall("atom:author", self.NS)
                    ],
                    "abstract": self._clean(self._text(entry, "atom:summary")),
                    "source_url": self._text(entry, "atom:id"),
                    "pdf_url": links.get("pdf"),
                    "published_at": self._text(entry, "atom:published"),
                    "updated_at": self._text(entry, "atom:updated"),
                    "categories": [
                        category.attrib.get("term", "")
                        for category in entry.findall("atom:category", self.NS)
                        if category.attrib.get("term")
                    ],
                    "doi": self._text(entry, "arxiv:doi") or None,
                    "journal_ref": self._text(entry, "arxiv:journal_ref") or None,
                }
            )
        return papers

    def _to_paper(self, item: dict[str, Any]) -> Paper:
        metadata = {
            "pdf_url": item.get("pdf_url") or "",
            "categories": ",".join(item.get("categories", [])),
        }
        if item.get("doi"):
            metadata["doi"] = item["doi"]
        if item.get("journal_ref"):
            metadata["journal_ref"] = item["journal_ref"]

        return Paper(
            paper_id=item["paper_id"],
            title=item["title"],
            authors=item["authors"],
            abstract=item["abstract"],
            source_url=item.get("source_url"),
            published_at=self._date_only(item.get("published_at")),
            metadata=metadata,
        )

    def _links(self, entry: ET.Element) -> dict[str, str]:
        links: dict[str, str] = {}
        for link in entry.findall("atom:link", self.NS):
            href = link.attrib.get("href")
            if not href:
                continue
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                links["pdf"] = href
            elif link.attrib.get("rel") == "alternate":
                links["alternate"] = href
        return links

    def _text(self, element: ET.Element, path: str) -> str:
        found = element.find(path, self.NS)
        return found.text.strip() if found is not None and found.text else ""

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _paper_id(source_url: str) -> str:
        path = urlparse(source_url).path.rstrip("/")
        return path.rsplit("/", maxsplit=1)[-1]

    @staticmethod
    def _date_only(value: str | None) -> str | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return value
