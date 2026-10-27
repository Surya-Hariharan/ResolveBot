"""The only file that talks to an LLM.

Groq and OpenAI both speak the OpenAI API, so one client with a different
base URL is enough. Temperature is always 0, as the assignment requires.
"""
from openai import OpenAI

from app.config import PROVIDERS

MAX_OUTPUT_TOKENS = 500


class LLMNotAvailable(Exception):
    """The LLM cannot be used (no key, unknown provider...). The message is safe to show."""


def chat(settings, messages, tools=None):
    """Send one request and return the model's reply message."""
    problem = settings.llm_problem()
    if problem:
        raise LLMNotAvailable(problem)

    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=PROVIDERS[settings.llm_provider]["base_url"],
        timeout=settings.llm_timeout_seconds,
        max_retries=1,
    )
    request = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    if tools:
        request["tools"] = tools
    response = client.chat.completions.create(**request)
    return response.choices[0].message
