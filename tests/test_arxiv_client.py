from src.crawler.arxiv_client import ArxivClient


SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <updated>2024-01-02T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title> Neural Ranking for IR </title>
    <summary>
      We study dense retrieval models.
    </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Grace Hopper</name></author>
    <category term="cs.IR" />
    <link href="http://arxiv.org/abs/2401.00001v1" rel="alternate" type="text/html" />
    <link title="pdf" href="http://arxiv.org/pdf/2401.00001v1" rel="related" type="application/pdf" />
    <arxiv:doi>10.0000/example</arxiv:doi>
  </entry>
</feed>
"""


def test_parse_arxiv_feed() -> None:
    client = ArxivClient()

    papers = client._parse_feed(SAMPLE_FEED)

    assert papers == [
        {
            "paper_id": "2401.00001v1",
            "title": "Neural Ranking for IR",
            "authors": ["Ada Lovelace", "Grace Hopper"],
            "abstract": "We study dense retrieval models.",
            "source_url": "http://arxiv.org/abs/2401.00001v1",
            "pdf_url": "http://arxiv.org/pdf/2401.00001v1",
            "published_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
            "categories": ["cs.IR"],
            "doi": "10.0000/example",
            "journal_ref": None,
        }
    ]


def test_arxiv_item_converts_to_domain_paper() -> None:
    client = ArxivClient()
    item = client._parse_feed(SAMPLE_FEED)[0]

    paper = client._to_paper(item)

    assert paper.paper_id == "2401.00001v1"
    assert paper.title == "Neural Ranking for IR"
    assert paper.published_at == "2024-01-01"
    assert paper.metadata["pdf_url"] == "http://arxiv.org/pdf/2401.00001v1"
    assert paper.metadata["categories"] == "cs.IR"
