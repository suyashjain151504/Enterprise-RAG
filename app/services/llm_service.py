"""Answer-generation client.

Switch answerer here (exactly one `generate()` may be defined):

- GEMINI (cloud): uncomment the Gemini `generate()` block. Comment out the
  local-llama.cpp `generate()` / `_llm_client()` block at the bottom.
- LOCAL llama.cpp (port 8080): keep the local `generate()` active (current).
  Docker must use LLM_BASE_URL=http://host.docker.internal:8080/v1
  (see docker-compose.yml). Host uv-run uses 127.0.0.1.

Gemini still needs GEMINI_API_KEY in .env. Local llama.cpp ignores that key.
"""
import re
import threading
import time

import google.generativeai as genai
from app.config import settings

# Used only by the Gemini generate() path below (harmless if that path is commented).
genai.configure(api_key=settings.gemini_api_key)

_lock = threading.Lock()
_last_call = 0.0
MIN_INTERVAL_SEC = 4.5   # 15 RPM free tier ≈ one call / 4s
MAX_RETRIES = 6


def _retry_sleep_seconds(exc: BaseException, attempt: int) -> float:
    text = str(exc)
    m = re.search(r"Please retry in ([\d.]+)s", text)
    if m:
        return float(m.group(1)) + 0.5
    m = re.search(r"retry_delay \{\s*seconds: (\d+)", text)
    if m:
        return float(m.group(1)) + 0.5
    return min(2 ** attempt * 5, 60)


def _is_rate_limit(exc: BaseException) -> bool:
    name = type(exc).__name__
    text = str(exc)
    return (
        "429" in text
        or "ResourceExhausted" in name
        or "rate" in text.lower() and "quota" in text.lower()
        or "quota" in text.lower() and "exceeded" in text.lower()
    )


def _paced_generate(factory):
    """factory() must call generate_content and return the response."""
    global _last_call
    last_exc: BaseException | None = None
    for attempt in range(MAX_RETRIES):
        with _lock:
            wait = MIN_INTERVAL_SEC - (time.monotonic() - _last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                response = factory()
            except Exception as e:
                _last_call = time.monotonic()
                last_exc = e
                if _is_rate_limit(e) and attempt < MAX_RETRIES - 1:
                    time.sleep(_retry_sleep_seconds(e, attempt))
                    continue
                raise
            _last_call = time.monotonic()
            return response
    assert last_exc is not None
    raise last_exc


# --- GEMINI answerer (cloud). Uncomment this whole function to use Gemini. ---
# --- Then comment out the local `generate()` below (Python allows only one). ---
def generate(system_prompt: str, user_message: str, model: str | None = None, temperature: float = 0.0) -> dict:
    if model is None:
        model = settings.llm_model_answer

    def _call():
        client = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature),
        )
        return client.generate_content(user_message)

    response = _paced_generate(_call)
    text = response.text or ""
    usage_metadata = getattr(response, "usage_metadata", None)
    usage = {
        "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0) or 0,
        "completion_tokens": getattr(usage_metadata, "candidates_token_count", 0) or 0,
        "total_tokens": getattr(usage_metadata, "total_token_count", 0) or 0,
    }
    return {"text": text, "usage": usage}


# --- LOCAL llama.cpp answerer (ACTIVE). llama.cpp OpenAI-compatible server on :8080. ---
# --- To go back to Gemini: comment out `_llm_client` + this `generate()`, uncomment Gemini `generate()` above. ---

# from openai import OpenAI
# from openai import APIConnectionError
# from app.config import settings


# def _llm_client() -> OpenAI:
#     return OpenAI(
#         base_url=settings.llm_base_url,
#         api_key="llamacpp",  # ignored by llama.cpp
#     )


# def generate(
#     system_prompt: str,
#     user_message: str,
#     model: str | None = None,
#     temperature: float = 0.0,
# ) -> dict:
#     try:
#         response = _llm_client().chat.completions.create(
#             model="local",  # llama.cpp uses whatever GGUF is already loaded
#             temperature=temperature,
#             max_tokens=1024,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_message},
#             ],
#             extra_body={"chat_template_kwargs": {"enable_thinking": False}},
#         )
#     except APIConnectionError as e:
#         raise ConnectionError(
#             f"Cannot reach local LLM at {settings.llm_base_url}. "
#             "From Docker this must be http://host.docker.internal:8080/v1 "
#             "(127.0.0.1 inside the container is not your Windows llama.cpp)."
#         ) from e
#     choice = response.choices[0].message
#     text = choice.content or ""
#     usage = {
#         "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
#         "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
#         "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
#     }
#     return {"text": text, "usage": usage}




# /////////////////////////// local llm 









# import google.generativeai as genai
# from app.config import settings

# # Configure the client once (Google AI Studio key)
# genai.configure(api_key=settings.gemini_api_key)  # or settings.google_api_key / whatever you named it


# def generate(
#     system_prompt: str,
#     user_message: str,
#     model: str | None = None,
#     temperature: float = 0.0,
# ) -> dict:
#     if model is None:
#         model = settings.llm_model_answer

#     # Gemini does not have a separate "system" role in the classic API.
#     # We put the system prompt as the first part of the conversation.
#     client = genai.GenerativeModel(
#         model_name=model,
#         system_instruction=system_prompt,
#         generation_config=genai.types.GenerationConfig(
#             temperature=temperature,
#         ),
#     )

#     response = client.generate_content(user_message)

#     text = response.text or ""

#     # Token usage (Gemini returns it under usage_metadata)
#     usage_metadata = getattr(response, "usage_metadata", None)
#     usage = {
#         "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0) or 0,
#         "completion_tokens": getattr(usage_metadata, "candidates_token_count", 0) or 0,
#         "total_tokens": getattr(usage_metadata, "total_token_count", 0) or 0,
#     }

#     return {"text": text, "usage": usage}


# def generate_with_json(
#     system_prompt: str,
#     user_message: str,
#     model: str | None = None,
#     temperature: float = 0.0,
# ) -> dict:
#     if model is None:
#         model = settings.llm_model_grader

#     client = genai.GenerativeModel(
#         model_name=model,
#         system_instruction=system_prompt,
#         generation_config=genai.types.GenerationConfig(
#             temperature=temperature,
#             response_mime_type="application/json",  # forces JSON output
#         ),
#     )

#     response = client.generate_content(user_message)

#     text = response.text or ""

#     usage_metadata = getattr(response, "usage_metadata", None)
#     usage = {
#         "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0) or 0,
#         "completion_tokens": getattr(usage_metadata, "candidates_token_count", 0) or 0,
#         "total_tokens": getattr(usage_metadata, "total_token_count", 0) or 0,
#     }

#     return {"text": text, "usage": usage}