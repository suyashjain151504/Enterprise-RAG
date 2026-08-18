from __future__ import annotations

import sys
import types

# Compatibility shim for ragas 0.3.x / 0.4.x
# ragas hard-imports ChatVertexAI from a path that was removed in modern
# langchain-community. This lives in our project so it works in Docker too.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:
        from langchain_community.chat_models.vertexai import ChatVertexAI  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        _mod = types.ModuleType("langchain_community.chat_models.vertexai")
        _mod.ChatVertexAI = None  # type: ignore[attr-defined]
        sys.modules["langchain_community.chat_models.vertexai"] = _mod

import os
# ... rest of your imports

from datasets import Dataset
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.llms import llm_factory
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.config import settings


METRICS = [
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy,
]

# def _get_ragas_llm():
#     """Create a Ragas-compatible LLM using app settings."""
#     os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)
#     return llm_factory(settings.llm_model_grader)

def _get_ragas_llm():
    """Create a Ragas-compatible LLM using Gemini."""
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model_grader,          # e.g. "gemini-1.5-flash" or "gemini-2.0-flash"
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
    )
    return LangchainLLMWrapper(llm)

# def _get_ragas_embeddings():
#     """Create Ragas-compatible embeddings using app settings."""
#     lc_emb = OpenAIEmbeddings(
#         model=settings.embedding_model,
#         api_key=settings.openai_api_key,
#     )
#     return LangchainEmbeddingsWrapper(lc_emb)

# def _get_ragas_embeddings():
#     """Create Ragas-compatible embeddings using Gemini (Jio subscription)."""
#     lc_emb = GoogleGenerativeAIEmbeddings(
#         model=settings.embedding_model,          # e.g. "models/embedding-001" or "models/text-embedding-004"
#         google_api_key=settings.gemini_api_key,
#     )
#     return LangchainEmbeddingsWrapper(lc_emb)

def _get_ragas_embeddings():
    """Very lightweight embeddings using FastEmbed (ideal for Docker)."""
    lc_emb = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    return LangchainEmbeddingsWrapper(lc_emb)


def build_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "user_input": [r["question"] for r in rows],
            "response": [r["answer"] for r in rows],
            "retrieved_contexts": [r["contexts"] for r in rows],
            "reference": [r["ground_truth"] for r in rows],
        }
    )
    
def run(rows:list[dict]) -> list[dict]:
    if not rows:
        return []
    
    ds = build_dataset(rows)
    result = evaluate(
        ds,
        metrics=METRICS,
        llm=_get_ragas_llm(),
        embeddings=_get_ragas_embeddings(),
        show_progress=True
    )
    
    return result.to_pandas().to_dict(orient="records")