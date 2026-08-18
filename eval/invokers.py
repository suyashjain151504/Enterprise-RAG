from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from app.config import settings
# from app.models import ChatResponse, RetrievedChunk
# from app.services.rag_service import run_rag_with_trace_no_cache

class SkippedIntent(Exception):
    """Raised when an intent is skipped due to missing configuration or other reasons."""
    pass

class Invoker(ABC):
    """Abstract base class for service invokers."""
    
    @abstractmethod
    def invoke(self, qustion:str, flags: dict, intent: str) -> tuple[Any, list]:
        ...
        
class ServiceInvoker(Invoker):
    """Invoker for the RAG service."""
    SUPPORTED_INTENTS = {"rag", "web_fallback"}
    
    def invoke(self, question:str, flags: dict, intent: str) -> tuple[Any, list]:
        raise NotImplementedError("RAG service is not implemented in lesson 0."
                                  "switch to lesson-1-naive branch to enable retrieval based evaluation.")

