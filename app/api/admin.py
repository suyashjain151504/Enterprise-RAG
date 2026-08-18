import asyncio
from typing import Any

from fastapi import APIRouter, Depends
from loguru import logger

from app.config import settings


from app.middleware.auth import User, require_admin


router = APIRouter(tags=["Admin"])

async def _ping_postgres() -> bool:
    try:
        import psycopg2

        conn = psycopg2.connect(settings.database_url, connect_timeout=2)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        return True
    except Exception as exc:
        logger.debug("Postgres health check failed: {}", exc)
        return False

async def _ping_qdrant() -> bool:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url, timeout=2)
        client.get_collections()
        return True
    except Exception as exc:
        logger.debug("Qdrant health check failed: {}", exc)
        return False

async def _ping_redis() -> bool:
    try:
        from upstash_redis import Redis

        redis = Redis(url=settings.upstash_redis_url, token=settings.upstash_redis_token)
        redis.ping()
        return True
    except Exception as exc:
        logger.debug("Redis health check failed: {}", exc)
        return False
    
async def _ping_gemini() -> bool:
    try:
        from google import genai

        client = genai.Client(api_key=settings.gemini_api_key)

        response = await client.aio.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents="ping",
            config={"max_output_tokens": 8},
        )

        if getattr(response, "text", None):
            return True
        if getattr(response, "candidates", None):
            return True
        return False

    except Exception as exc:
        logger.warning("Gemini health check failed: {}", exc)
        return False

# async def _ping_gemini() -> bool:
#     try:
#         import google.generativeai as genai

#         genai.configure(api_key=settings.gemini_api_key)
        
#         model = genai.GenerativeModel("gemini-3.1-flash-lite")  # or any model you use
#         response = await model.generate_content_async(
#             "ping",
#             generation_config={"max_output_tokens": 1}
#         )
        
#         return bool(response.text)
#     except Exception as exc:
#         logger.debug("Gemini health check failed: {}", exc)
#         return False

# async def _ping_tavily() -> bool:
#     try:
#         from app.services.web_search import search_web

#         search_web("health check")
#         return True
#     except ValueError:
#         # Tavily key not configured — still "up" if the module loads
#         return True
#     except Exception as exc:
#         logger.debug("Tavily health check failed: {}", exc)
#         return False

@router.get("/admin/health")
async def health_check() -> dict[str, Any]:
    
    results = await asyncio.gather(
        _ping_postgres(),
        _ping_qdrant(),
        _ping_redis(),
        _ping_gemini(),
        # _ping_tavily(),
        return_exceptions=True   
    )
    
    postgres_ok = bool(results[0]) if not isinstance(results[0], Exception) else False
    qdrant_ok = bool(results[1]) if not isinstance(results[1], Exception) else False
    redis_ok = bool(results[2]) if not isinstance(results[2], Exception) else False
    gemini_ok = bool(results[3]) if not isinstance(results[3], Exception) else False
    # tavily_ok = bool(results[4]) if not isinstance(results[4], Exception) else False
    
    all_ok = postgres_ok and qdrant_ok and redis_ok and gemini_ok
    
    status = "ok" if all_ok else "degraded"
    
    return {
        "status" : status,
        "postgres" : postgres_ok,
        "qdrant" : qdrant_ok,
        "redis" : redis_ok,
        "gemini" : gemini_ok,
        # "tavily" : tavily_ok        
    }
