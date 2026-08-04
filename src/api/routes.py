from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.domain.query import Query
from src.api.rag_session import RagSessionManager, TopicSession
from src.paper_discovery.discovery_service import DiscoveryService
from src.retrieval.hybrid_retriever import HybridRetriever

router = APIRouter()
session_manager = RagSessionManager()


class TopicRequest(BaseModel):
    topic: str = Field(min_length=1)
    papers: list[dict] = Field(default_factory=list)
    base_model_name: str | None = Field(default=None)


class SettingsRequest(BaseModel):
    hybrid_enabled: bool
    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    rrf_k: int
    dense_weight: float = Field(default=1.0)
    sparse_weight: float = Field(default=1.0)
    ranking_enabled: bool = Field(default=True)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    paper_id: str | None = Field(default=None)


class SourceResponse(BaseModel):
    chunk_id: str
    paper_id: str
    score: float
    text: str
    metadata: dict[str, str]


class DebugRankItem(BaseModel):
    chunk_id: str
    paper_id: str
    score: float
    rank: int | None = None


class RetrievalDebugInfo(BaseModel):
    dense_latency_ms: float
    sparse_latency_ms: float
    fusion_latency_ms: float
    total_latency_ms: float
    dense_results: list[DebugRankItem]
    sparse_results: list[DebugRankItem]
    fused_results: list[DebugRankItem]


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    debug_info: RetrievalDebugInfo | None = None


class SummarizeRequest(BaseModel):
    paper_id: str


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/session/search")
def search_papers(query: str, limit: int = 10) -> list[dict]:
    try:
        service = DiscoveryService()
        discovered = service.search_papers(query, limit=limit)
        return [
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "authors": p.authors,
                "abstract": p.abstract,
                "pdf_url": p.pdf_url,
                "published_year": p.published_year,
                "categories": p.categories,
                "doi": p.doi,
            }
            for p in discovered
        ]
    except Exception as exc:
        err_msg = str(exc)
        if "429" in err_msg or "too many requests" in err_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="arXiv rate-limit block (HTTP 429). Please wait a moment before trying again."
            )
        raise HTTPException(status_code=500, detail=err_msg)


@router.post("/session/start")
def start_session(request: TopicRequest) -> dict[str, str]:
    try:
        session_manager.start_background(
            request.topic,
            request.papers,
            base_model_name=request.base_model_name,
        )
        return {"status": "started"}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not prepare topic: {exc}") from exc


@router.get("/session/progress")
def session_progress() -> dict[str, object]:
    return session_manager.get_status()


@router.get("/session/status")
def session_status() -> dict[str, object]:
    session = session_manager.get_active()
    if session is None:
        return {"ready": False}
    
    stats = {
        "papers_count": len(session.papers),
        "chunks_count": session.chunks_count,
        "embedding_model": session_manager.model_name,
        "hybrid_enabled": session_manager.settings["hybrid_enabled"],
        "dense_index_status": "built" if (session.session_dir / "index" / "faiss.index").exists() else "none",
        "sparse_index_status": "built" if (session.session_dir / "index" / "bm25.pkl").exists() else "none",
    }

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
                "published_at": p.published_at or "",
            }
            for p in session.papers
        ],
        "stats": stats,
    }


@router.get("/settings")
def get_settings() -> dict[str, Any]:
    return {
        "settings": session_manager.settings,
        "embedding_model": session_manager.model_name,
        "base_model_name": session_manager.base_model_name,
    }


@router.post("/settings")
def update_settings(request: SettingsRequest) -> dict[str, str]:
    session_manager.settings.update({
        "hybrid_enabled": request.hybrid_enabled,
        "dense_top_k": request.dense_top_k,
        "sparse_top_k": request.sparse_top_k,
        "fusion_top_k": request.fusion_top_k,
        "rrf_k": request.rrf_k,
        "dense_weight": request.dense_weight,
        "sparse_weight": request.sparse_weight,
        "ranking_enabled": request.ranking_enabled,
    })
    session = session_manager.get_active()
    if session and hasattr(session.retriever, "fusion_strategy"):
        from src.retrieval.fusion.rrf import ReciprocalRankFusion
        session.retriever.fusion_strategy = ReciprocalRankFusion(
            k=request.rrf_k,
            weights=(request.dense_weight, request.sparse_weight)
        )
    return {"status": "updated"}


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session = session_manager.get_active()
    if session is None:
        raise HTTPException(
            status_code=409,
            detail="Please choose a research topic before chatting.",
        )

    settings = session_manager.settings
    hybrid_enabled = settings["hybrid_enabled"]

    filters = {}
    query = Query(
        text=request.question,
        top_k=settings["fusion_top_k"] if hybrid_enabled else settings["dense_top_k"],
        filters=filters,
        paper_id=request.paper_id,
    )

    import time
    start_time = time.time()

    if hybrid_enabled:
        results = session.retriever.retrieve(query)
        retriever_instance = session.retriever
    else:
        results = session.retriever.dense_retriever.retrieve(query)
        retriever_instance = session.retriever.dense_retriever

    # 1. Apply Field-Aware Ranking & Combination if enabled
    ranking_enabled = settings.get("ranking_enabled", True)
    if ranking_enabled:
        from src.retrieval.ranking.ranking_weights import RankingConfig
        from src.retrieval.ranking.field_aware_ranker import FieldAwareRanker
        from src.retrieval.ranking.score_combiner import ScoreCombiner
        
        rank_config = RankingConfig()
        ranker = FieldAwareRanker(rank_config)
        results = ranker.rank(query, results)
        
        combiner = ScoreCombiner(
            rrf_weight=rank_config.rrf_weight,
            field_weight=rank_config.field_weight
        )
        results = combiner.combine_list(results)

    total_latency_ms = (time.time() - start_time) * 1000

    debug_info = None
    if hybrid_enabled and isinstance(session.retriever, HybridRetriever):
        dense_dbg = [
            DebugRankItem(chunk_id=r.chunk_id, paper_id=r.paper_id, score=r.score)
            for r in getattr(session.retriever, "last_dense_results", [])
        ]
        sparse_dbg = [
            DebugRankItem(chunk_id=r.chunk_id, paper_id=r.paper_id, score=r.score)
            for r in getattr(session.retriever, "last_sparse_results", [])
        ]
        fused_dbg = [
            DebugRankItem(
                chunk_id=r.chunk_id,
                paper_id=r.paper_id,
                score=r.score,
                rank=rank,
            )
            for rank, r in enumerate(getattr(session.retriever, "last_fused_results", []), start=1)
        ]
        debug_info = RetrievalDebugInfo(
            dense_latency_ms=getattr(session.retriever, "last_dense_time", 0.0),
            sparse_latency_ms=getattr(session.retriever, "last_sparse_time", 0.0),
            fusion_latency_ms=getattr(session.retriever, "last_fusion_time", 0.0),
            total_latency_ms=total_latency_ms,
            dense_results=dense_dbg,
            sparse_results=sparse_dbg,
            fused_results=fused_dbg,
        )
    else:
        dense_results = getattr(retriever_instance, "last_dense_results", results) if hasattr(retriever_instance, "last_dense_results") else results
        dense_dbg = [
            DebugRankItem(chunk_id=r.chunk_id, paper_id=r.paper_id, score=r.score)
            for r in dense_results
        ]
        debug_info = RetrievalDebugInfo(
            dense_latency_ms=total_latency_ms,
            sparse_latency_ms=0.0,
            fusion_latency_ms=0.0,
            total_latency_ms=total_latency_ms,
            dense_results=dense_dbg,
            sparse_results=[],
            fused_results=[],
        )

    results = results[:request.top_k]
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
        debug_info=debug_info,
    )


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
            status_code=404,
            detail=f"Paper with ID '{request.paper_id}' not found in active session.",
        )

    paper_chunks = [
        chunk for chunk in session.retriever.dense_retriever.chunks.values()
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
