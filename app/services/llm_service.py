import google.generativeai as genai
from app.config import settings

# Configure the client once (Google AI Studio key)
genai.configure(api_key=settings.gemini_api_key)  # or settings.google_api_key / whatever you named it


def generate(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    if model is None:
        model = settings.llm_model_answer

    # Gemini does not have a separate "system" role in the classic API.
    # We put the system prompt as the first part of the conversation.
    client = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
        ),
    )

    response = client.generate_content(user_message)

    text = response.text or ""

    # Token usage (Gemini returns it under usage_metadata)
    usage_metadata = getattr(response, "usage_metadata", None)
    usage = {
        "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0) or 0,
        "completion_tokens": getattr(usage_metadata, "candidates_token_count", 0) or 0,
        "total_tokens": getattr(usage_metadata, "total_token_count", 0) or 0,
    }

    return {"text": text, "usage": usage}


def generate_with_json(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    if model is None:
        model = settings.llm_model_grader

    client = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json",  # forces JSON output
        ),
    )

    response = client.generate_content(user_message)

    text = response.text or ""

    usage_metadata = getattr(response, "usage_metadata", None)
    usage = {
        "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0) or 0,
        "completion_tokens": getattr(usage_metadata, "candidates_token_count", 0) or 0,
        "total_tokens": getattr(usage_metadata, "total_token_count", 0) or 0,
    }

    return {"text": text, "usage": usage}