import uuid
from app.config import settings
from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from app.middleware.rate_limiter import is_allowed_user
# from app.security.content_moderation import moderate_and_redact
# from app.security.input_guard import check_input_safe
# from app.security.input_restructuring import count_tokens, restructure_input
# from app.security.token_budget import check_budget, consume_budget
from app.services.rag_service import run_rag

# from app.core.graph import graph
from app.middleware.auth import User, get_current_user
from app.models import ChatResponse, QueryRequest, PendingSQLBlock


router = APIRouter(tags=["query"])

@router.post("/query", response_model=ChatResponse)
async def query(
    body: QueryRequest,
    user: User = Depends(get_current_user)
) -> ChatResponse:
    return run_rag(body.question, flags={"top_k": body.top_k})
    
    