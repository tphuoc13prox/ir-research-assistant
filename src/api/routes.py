from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.domain.query import Query
from src.api.rag_session import RagSessionManager

router = APIRouter()
session_manager = RagSessionManager()


class TopicRequest(BaseModel):
    topic: str = Field(min_length=1)
    max_papers: int = Field(default=100, ge=1, le=150)
    base_model_name: str | None = Field(default=None)


class TopicResponse(BaseModel):
    topic: str
    papers_found: int
    pdfs_downloaded: int
    chunks_indexed: int
    session_dir: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceResponse(BaseModel):
    chunk_id: str
    paper_id: str
    score: float
    text: str
    metadata: dict[str, str]


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/session/status")
def session_status() -> dict[str, object]:
    session = session_manager.get_active()
    if session is None:
        return {"ready": False}
    return {
        "ready": True,
        "topic": session.topic,
        "papers_found": len(session.papers),
        "pdfs_downloaded": len(session.pdf_paths),
        "chunks_indexed": session.chunks_count,
        "session_dir": str(session.session_dir),
        "papers": [
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "authors": p.authors,
                "abstract": p.abstract,
                "source_url": p.source_url or "",
            }
            for p in session.papers
        ]
    }


@router.get("/session/progress")
def session_progress() -> dict[str, object]:
    return session_manager.get_status()


@router.post("/session/start")
def start_session(request: TopicRequest) -> dict[str, str]:
    try:
        session_manager.start_background(
            request.topic,
            max_papers=10,
            base_model_name=request.base_model_name,
        )
        return {"status": "started"}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not prepare topic: {exc}") from exc

    return {
        "status": "started",
        "topic": request.topic,
    }


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session = session_manager.get_active()
    if session is None:
        raise HTTPException(
            status_code=409,
            detail="Please choose a research topic before chatting.",
        )

    results = session.retriever.retrieve(Query(text=request.question, top_k=request.top_k))
    answer = session.generator.generate(request.question, results)
    return ChatResponse(
        answer=answer,
        sources=[
            SourceResponse(
                chunk_id=result.chunk_id,
                paper_id=result.paper_id,
                score=result.score,
                text=result.text,
                metadata=result.metadata,
            )
            for result in results
        ],
    )


class SummarizeRequest(BaseModel):
    paper_id: str


@router.post("/paper/summarize")
def summarize_paper(request: SummarizeRequest) -> dict[str, str]:
    session = session_manager.get_active()
    if session is None:
        raise HTTPException(
            status_code=409,
            detail="Please choose a research topic before summarizing papers.",
        )

    paper = next((p for p in session.papers if p.paper_id == request.paper_id), None)
    if not paper:
        raise HTTPException(
            status_code=444,
            detail=f"Paper with ID '{request.paper_id}' not found in active session.",
        )

    # Find chunks belonging to this paper
    paper_chunks = [
        chunk for chunk in session.retriever.chunks.values()
        if chunk.paper_id == request.paper_id
    ]

    if not paper_chunks:
        text_to_summarize = f"Title: {paper.title}\nAbstract: {paper.abstract}"
    else:
        try:
            paper_chunks = sorted(paper_chunks, key=lambda c: int(c.chunk_id.split("_chunk_")[-1]))
        except Exception:
            pass
        selected_chunks = paper_chunks[:4]
        text_to_summarize = "\n\n".join(c.text for c in selected_chunks)

    prompt = (
        f"You are a research assistant. Provide a concise summary of the following research paper titled '{paper.title}'.\n"
        f"Format your response with exactly these sections:\n"
        f"1. **Core Objective**: What the paper aims to solve.\n"
        f"2. **Key Methodology**: How they solve it.\n"
        f"3. **Main Findings/Results**: What they achieved.\n\n"
        f"Context from paper:\n{text_to_summarize}"
    )

    summary = session.generator.llm_client.complete(prompt)

    return {
        "paper_id": request.paper_id,
        "title": paper.title,
        "summary": summary,
    }
