"""The only file that talks to an LLM.

Groq, OpenAI, Anthropic and Google all offer an OpenAI-style chat API, so the openai client
pointed at the chosen provider's base URL (see app/config.py) is enough.
Temperature is always 0, as the assignment requires.
"""
from openai import BadRequestError, OpenAI

MAX_OUTPUT_TOKENS = 500


class LLMNotAvailable(Exception):
    """The LLM cannot be used (no key). The message is safe to show."""


def chat(settings, messages, tools=None, timeout=None):
    """Send one request and return the model's reply message.

    `timeout` (seconds) lets the caller shorten the wait to fit the 30-second reply limit.
    """
    problem = settings.llm_problem()
    if problem:
        raise LLMNotAvailable(problem)

    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=min(timeout or settings.llm_timeout_seconds, settings.llm_timeout_seconds),
        max_retries=0,  # the SDK would sleep through a rate-limit wait and miss the 30-second limit
    )
    request = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": 0,
        # OpenAI's newer models reject max_tokens; the other providers still use it
        ("max_completion_tokens" if settings.llm_provider == "openai" else "max_tokens"): MAX_OUTPUT_TOKENS,
    }
    if tools:
        request["tools"] = tools
    try:
        response = client.chat.completions.create(**request)
    except BadRequestError as problem:
        # Groq answers 400 "tool_use_failed" when the model wrote a malformed tool call. It is random,
        # so one more try is allowed; any other 400 (a bad request of ours) is not retried.
        if getattr(problem, "code", "") != "tool_use_failed":
            raise
        response = client.chat.completions.create(**request)
    return response.choices[0].message
