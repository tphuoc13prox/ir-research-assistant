from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.domain.query import Query
from src.api.rag_session import RagSessionManager

router = APIRouter()
session_manager = RagSessionManager()


class TopicRequest(BaseModel):
    topic: str = Field(min_length=1)
    max_papers: int = Field(default=100, ge=1, le=150)


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
    }


@router.get("/session/progress")
def session_progress() -> dict[str, object]:
    return session_manager.get_status()


@router.post("/session/start")
def start_session(request: TopicRequest) -> dict[str, str]:
    try:
        session_manager.start_background(
            request.topic,
            max_papers=request.max_papers,
        )
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
