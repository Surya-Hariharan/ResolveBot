"""All settings, including the API-key lookup, live in this file.

No other module reads environment variables, so there is one place to look
when something is misconfigured.
"""
import os
from dataclasses import dataclass, field
from datetime import date

from dotenv import load_dotenv

# What we know about each provider. Groq (and OpenAI) are reached through the
# OpenAI-compatible API, so one small client covers both. Anthropic and Google
# are listed only so their key variables are recognised; there is no adapter yet.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_variable": "GROQ_API_KEY",
        "has_adapter": True,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_variable": "OPENAI_API_KEY",
        "has_adapter": True,
    },
    "anthropic": {"base_url": None, "key_variable": "ANTHROPIC_API_KEY", "has_adapter": False},
    "google": {"base_url": None, "key_variable": "GOOGLE_API_KEY", "has_adapter": False},
}


@dataclass
class Settings:
    policies_path: str
    manifest_path: str
    db_path: str
    app_today: date
    llm_provider: str
    llm_model: str
    llm_timeout_seconds: float
    # repr=False so the key can never end up in a log line or traceback by accident
    llm_api_key: str = field(default="", repr=False)

    def llm_problem(self):
        """Why the LLM cannot be used right now, or None if it can.

        The message names variables only, never key values.
        """
        provider = PROVIDERS.get(self.llm_provider)
        if provider is None:
            return f"LLM_PROVIDER '{self.llm_provider}' is not a known provider."
        if not provider["has_adapter"]:
            return f"LLM_PROVIDER '{self.llm_provider}' is recognised but not implemented yet."
        if not self.llm_api_key:
            return (
                f"No LLM API key is configured. Set LLM_API_KEY or {provider['key_variable']}."
            )
        return None


def find_api_key(env, provider_name):
    """LLM_API_KEY wins; otherwise the selected provider's own variable.

    Empty values count as unset (a blank line in .env is common). We never
    borrow another provider's key, so a key is not sent to the wrong service.
    """
    generic_key = env.get("LLM_API_KEY", "").strip()
    if generic_key:
        return generic_key
    provider = PROVIDERS.get(provider_name)
    if provider is None:
        return ""
    return env.get(provider["key_variable"], "").strip()


def load_settings(env=None):
    """Build Settings from the environment. Tests pass their own dict as env."""
    if env is None:
        load_dotenv()  # real environment variables still win over .env
        env = os.environ

    provider_name = env.get("LLM_PROVIDER", "groq").strip().lower()
    try:
        app_today = date.fromisoformat(env.get("APP_TODAY", "2026-07-01").strip())
    except ValueError:
        raise ValueError("APP_TODAY must look like 2026-07-01 (YYYY-MM-DD).")

    return Settings(
        policies_path=env.get("POLICIES_PATH", "./data/policies.md"),
        manifest_path=env.get("MANIFEST_PATH", "./data/manifest.json"),
        db_path=env.get("DB_PATH", "./data/shopsphere.db"),
        app_today=app_today,
        llm_provider=provider_name,
        llm_model=env.get("LLM_MODEL", "llama-3.3-70b-versatile").strip(),
        llm_timeout_seconds=float(env.get("LLM_TIMEOUT_SECONDS", "20")),
        llm_api_key=find_api_key(env, provider_name),
    )
