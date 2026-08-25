from typing import Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
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
    base_model_name: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
    relevance_threshold: float = Field(default=0.35)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    paper_id: str | None = Field(default=None)


class SourceResponse(BaseModel):
    chunk_id: str
    paper_id: str
    score: float
    text: str
    metadata: dict[str, Any]
    explanation: dict[str, Any] = Field(default_factory=dict)


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
        from src.query_understanding.query_parser import parse_query
        from src.domain.query import Query

        # Parse search query to extract constraints
        q_obj = Query(text=query, top_k=limit)
        parsed_q = parse_query(q_obj)
        
        # Build clean query for arXiv API
        topic_query = parsed_q.topic
        if parsed_q.boost_terms:
            topic_query += " " + " ".join(f'"{term}"' for term in parsed_q.boost_terms)
        
        if not topic_query.strip():
            topic_query = query

        # Fetch more candidates to allow filtering
        fetch_limit = max(limit * 3, 30)
        service = DiscoveryService()
        discovered = service.search_papers(topic_query, limit=fetch_limit)
        
        # Filter papers based on parsed year constraints and exclusions
        filtered = []
        year_constraints = parsed_q.metadata_constraints.get("year", {})
        
        for p in discovered:
            # 1. Check year constraints
            p_year = None
            try:
                p_year = int(p.published_year)
            except Exception:
                pass
                
            if p_year is not None:
                passed_year = True
                for op, val in year_constraints.items():
                    if op == ">" and not (p_year > val):
                        passed_year = False
                    elif op == ">=" and not (p_year >= val):
                        passed_year = False
                    elif op == "<" and not (p_year < val):
                        passed_year = False
                    elif op == "<=" and not (p_year <= val):
                        passed_year = False
                if not passed_year:
                    continue
            elif year_constraints:
                continue
                
            # 2. Check exclusion terms
            passed_exclusion = True
            text_for_checking = (p.title + " " + p.abstract + " " + " ".join(p.categories)).lower()
            for exc_term in parsed_q.exclude_terms:
                if exc_term in text_for_checking:
                    passed_exclusion = False
                    break
            if not passed_exclusion:
                continue
                
            filtered.append(p)
            
        # Return at most 'limit' results
        filtered = filtered[:limit]
        
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
            for p in filtered
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
        "base_model_name": request.base_model_name,
        "relevance_threshold": request.relevance_threshold,
    })
    session_manager.base_model_name = request.base_model_name
    session = session_manager.get_active()
    if session:
        if hasattr(session.retriever, "fusion_strategy"):
            from src.retrieval.fusion.rrf import ReciprocalRankFusion
            session.retriever.fusion_strategy = ReciprocalRankFusion(
                k=request.rrf_k,
                weights=(request.dense_weight, request.sparse_weight)
            )

        current_model = getattr(session.generator.llm_client, "base_model_name", None)
        if current_model != request.base_model_name:
            import gc
            import torch
            
            # Clean up VRAM/RAM of previous model
            if hasattr(session.generator, "llm_client"):
                if hasattr(session.generator.llm_client, "model"):
                    del session.generator.llm_client.model
                if hasattr(session.generator.llm_client, "tokenizer"):
                    del session.generator.llm_client.tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            from src.generation.llm_client import LocalFinetunedLLMClient
            from src.generation.generator import Generator
            from src.generation.prompt_builder import PromptBuilder

            session.generator = Generator(
                LocalFinetunedLLMClient(request.base_model_name, request.base_model_name),
                PromptBuilder(),
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
        top_k=30,
        filters=filters,
        paper_id=request.paper_id,
    )

    qu_enabled = settings.get("query_understanding", {}).get("enabled", True)
    if qu_enabled:
        from src.query_understanding.query_parser import parse_query
        query = parse_query(query)

    import time
    start_time = time.time()

    if hybrid_enabled:
        results = session.retriever.retrieve(query)
        retriever_instance = session.retriever
    else:
        results = session.retriever.dense_retriever.retrieve(query)
        retriever_instance = session.retriever.dense_retriever

    filter_enabled = settings.get("query_understanding", {}).get("metadata_filter", True)
    if qu_enabled and filter_enabled:
        from src.retrieval.filtering.metadata_filter import filter_results
        results = filter_results(results, query)

    # 1. Apply Field-Aware Ranking & Combination if enabled
    ranking_enabled = settings.get("ranking_enabled", True)
    if ranking_enabled:
        from src.retrieval.ranking.ranking_weights import RankingConfig
        from src.retrieval.ranking.field_aware_ranker import FieldAwareRanker
        from src.retrieval.ranking.score_combiner import ScoreCombiner
        
        rank_config = RankingConfig()
        ranker = FieldAwareRanker(rank_config)
        results = ranker.rank(query, results)
        
        intent_enabled = settings.get("query_understanding", {}).get("intent_ranking", True)
        if qu_enabled and intent_enabled:
            from src.retrieval.ranking.intent_ranker import IntentRanker
            intent_ranker = IntentRanker()
            results = intent_ranker.rank(query, results)
        
        combiner = ScoreCombiner(
            rrf_weight=rank_config.rrf_weight,
            field_weight=rank_config.field_weight
        )
        results = combiner.combine_list(results)

    from src.retrieval.ranking.explanation_generator import generate_explanation
    for r in results:
        r.explanation = generate_explanation(r, query)

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

    threshold = settings.get("relevance_threshold", 0.35)
    results = [r for r in results if r.score >= threshold]
    results = results[:30]
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
                explanation=getattr(result, "explanation", {}),
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


@router.get("/paper/{paper_id}/content")
def get_paper_content(paper_id: str) -> dict:
    session = session_manager.get_active()
    if session is None:
        raise HTTPException(
            status_code=409,
            detail="Please choose a research topic before viewing papers.",
        )

    paper = next((p for p in session.papers if p.paper_id == paper_id), None)
    if not paper:
        raise HTTPException(
            status_code=404,
            detail=f"Paper with ID '{paper_id}' not found in active session.",
        )

    # Gather chunks of this paper
    paper_chunks = [
        chunk for chunk in session.retriever.dense_retriever.chunks.values()
        if chunk.paper_id == paper_id
    ]

    # Sort chunks in original order
    def get_chunk_idx(c):
        cid = c.chunk_id
        try:
            if "_chunk_" in cid:
                return int(cid.split("_chunk_")[-1])
            if ":" in cid:
                parts = cid.split(":")
                if parts[-1].isdigit():
                    return int(parts[-1])
        except Exception:
            pass
        return 0

    paper_chunks.sort(key=get_chunk_idx)

    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": [a.get("name") if isinstance(a, dict) else str(a) for a in paper.authors] if paper.authors else [],
        "published_at": paper.published_at,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "page": getattr(c, "page", None),
                "section": getattr(c, "section", None),
            }
            for c in paper_chunks
        ]
    }


@router.get("/paper/{paper_id}/pdf")
def get_paper_pdf(paper_id: str) -> FileResponse:
    session = session_manager.get_active()
    if session is None:
        raise HTTPException(
            status_code=409,
            detail="Please choose a research topic before viewing PDFs.",
        )

    pdf_path = session.session_dir / "papers" / f"{paper_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF file for paper '{paper_id}' not found.",
        )

    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename={paper_id}.pdf"}
    )


import os
import sys
import signal
import threading
import time

last_heartbeat_time = time.time()

def monitor_heartbeat():
    global last_heartbeat_time
    # Give the browser 15 seconds to open and make the first heartbeat request
    time.sleep(15)
    while True:
        time.sleep(3)
        if time.time() - last_heartbeat_time > 8.0:
            print("No active browser connection detected. Automatically shutting down server...")
            try:
                os.kill(os.getppid(), signal.SIGTERM)
            except Exception:
                pass
            try:
                os.kill(os.getpid(), signal.SIGTERM)
            except Exception:
                pass
            os._exit(0)

# Avoid starting the monitor thread when running automated tests
is_testing = any("pytest" in arg.lower() or "unittest" in arg.lower() for arg in sys.argv)
if not is_testing:
    print("[SERVER] Starting connection heartbeat monitor...")
    threading.Thread(target=monitor_heartbeat, daemon=True).start()


@router.post("/heartbeat")
def heartbeat() -> dict[str, str]:
    global last_heartbeat_time
    last_heartbeat_time = time.time()
    return {"status": "ok"}
