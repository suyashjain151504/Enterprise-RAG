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


from langchain_core.embeddings import Embeddings
from ragas.run_config import RunConfig


class _FastEmbedForRagas(Embeddings):
    """Expose .model as a str so ragas EmbeddingUsageEvent validation succeeds."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = model_name
        self._inner = FastEmbedEmbeddings(model_name=model_name)

    def embed_documents(self, texts):
        return self._inner.embed_documents(texts)

    def embed_query(self, text):
        return self._inner.embed_query(text)

    async def aembed_documents(self, texts):
        return await self._inner.aembed_documents(texts)

    async def aembed_query(self, text):
        return await self._inner.aembed_query(text)


# --- GEMINI grader. Uncomment this and comment out the local `_get_ragas_llm()` below. ---
# def _get_ragas_llm():
#     """Create a Ragas-compatible LLM using Gemini."""
#     llm = ChatGoogleGenerativeAI(
#         model=settings.llm_model_grader,          # e.g. "gemini-1.5-flash" or "gemini-2.0-flash"
#         google_api_key=settings.gemini_api_key,
#         temperature=0.0,
#     )
#     return LangchainLLMWrapper(llm)

# --- LOCAL llama.cpp grader (ACTIVE). Same GGUF as the answerer on :8080. ---
# --- Eval on the host uses 127.0.0.1. If you run eval inside Docker, use host.docker.internal. ---
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

def _get_ragas_llm():
    llm = ChatOpenAI(
        model="local",
        base_url="http://127.0.0.1:8080/v1",
        api_key="llamacpp",
        temperature=0.0,
        max_tokens=1024,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return LangchainLLMWrapper(llm)

def build_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "user_input": [r["question"] for r in rows],
            "response": [r["answer"] for r in rows],
            "retrieved_contexts": [r["contexts"] for r in rows],
            "reference": [r["ground_truth"] for r in rows],
        }
    )
    

def _get_ragas_embeddings():
    return LangchainEmbeddingsWrapper(_FastEmbedForRagas("BAAI/bge-small-en-v1.5"))


def run(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    answer_relevancy.strictness = 1  # Gemini cannot return n=3 completions

    ds = build_dataset(rows)
    result = evaluate(
        ds,
        metrics=METRICS,
        llm=_get_ragas_llm(),
        embeddings=_get_ragas_embeddings(),
        show_progress=True,
        run_config=RunConfig(
            max_workers=1,   # default is 16; free tier is 15 RPM
            timeout=180,
            max_retries=3,
        ),
    )
    return result.to_pandas().to_dict(orient="records")