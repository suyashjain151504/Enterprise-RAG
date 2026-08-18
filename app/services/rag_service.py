from __future__ import annotations

from loguru import logger

from app.config import settings
from app.models import (
    ChatResponse,
    ResponseMetadata,
    RetrievedChunk,
    RetrievedChunkPreview,
)
from app.security.spotlighting import build_spotlighted_context
from app.security.system_prompt import build_system_prompt
# from app.services.crag import crag_pipeline
from app.services.embedding_service import embed_texts
# from app.services.reranking import Reranker
from app.services.llm_service import generate
# from app.services.vector_store import search, hybrid_search, sparse_search
# from app.services.self_reflective import reflect_on_answer, should_regenerate
# from app.services.hyde import HyDERetriever
# from app.services.router_service import classify_intent
# from app.services.sql_service import SQLService
from app.services.query_cache_service import query_cache


def _retrieve(question: str , top_k: int =5) -> list[RetrievedChunk]:
    """Retrieve relevant chunks from the vector store based on the question."""
    
    embedding = embed_texts([question])
    return search(embedding[0], top_k=top_k)

def _generate(question: str, chunks: list[RetrievedChunk]) -> ChatResponse:
    """Generate an answer using the LLM based on the question and retrieved chunks."""
    spotlighted = build_spotlighted_context(chunks)
    system = build_system_prompt()
    user_msg = f"{spotlighted}\n\n Question: {question}"
    raw = generate(system, user_msg)["text"]
    chunks_previews = [
        RetrievedChunkPreview(text=c.text, source=c.source, score=c.score) for c in chunks
    ]
    
    return ChatResponse(
        answer=raw,
        sources=list({c.source for c in chunks}),
        confidence=0.7,
        metadata=ResponseMetadata(
            route= "rag",
            retrieved_chunks=chunks_previews
        )
    )
    
    
def _top_k_from_flags(flags:dict | int | None) -> int:
    if flags in None:
        return 5
    if isinstance(flags, int):
        return flags
    return int(flags.get("top_k", 5))



def run_rag(question : str, flags: dict | int | None = None) -> ChatResponse:
    """Run the RAG pipeline: retrieve relevant chunks and generate an answer."""
    top_k = _top_k_from_flags(flags)
    logger.info("L1 naive RAG | top_k={}", top_k)
    chunks = _retrieve(question, top_k=top_k)
    return _generate(question, chunks)

def run_rag_with_trace(question : str, flags: dict | int | None = None) -> tuple[ChatResponse, list[RetrievedChunk]]:
    top_k = _top_k_from_flags(flags)
    chunks = _retrieve(question, top_k=top_k)
    response = _generate(question, chunks)
    return response, chunks

run_rag_with_trace_no_cache = run_rag_with_trace


    