"""All settings, including the API-key lookup, live in this file.

No other module reads environment variables, so there is one place to look
when something is misconfigured.
"""
import os
from dataclasses import dataclass, field
from datetime import date

from dotenv import load_dotenv

# provider -> (base URL, old provider-specific key variable, default model).
# The key now comes from LLM_API_KEY; the old variable is only a fallback so an existing .env keeps working.
# All four offer an OpenAI-style chat API, so one client (the openai package) serves them all.
# Only Groq has been tested live; the others are untested without a key (see README).
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY", "claude-haiku-5-5"),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY", "gemini-2.0-flash"),
}
DEFAULT_PROVIDER = "groq"
GROQ_BASE_URL = PROVIDERS[DEFAULT_PROVIDER][0]


@dataclass
class Settings:
    policies_path: str
    manifest_path: str
    db_path: str
    app_today: date
    llm_model: str
    llm_timeout_seconds: float
    llm_provider: str = "groq"
    llm_base_url: str = GROQ_BASE_URL
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.3  # below this a paragraph is not relevant enough to use
    # repr=False so the key can never end up in a log line or traceback by accident
    llm_api_key: str = field(default="", repr=False)

    def llm_problem(self):
        """Why the LLM cannot be used right now, or None if it can.

        The message names the variable only, never the key value.
        """
        if self.llm_provider not in PROVIDERS:
            return f"LLM_PROVIDER must be one of: {', '.join(PROVIDERS)}."
        if not self.llm_api_key:
            return f"No API key is configured for {self.llm_provider}. Set LLM_API_KEY in .env."
        return None


def load_settings(env=None):
    """Build Settings from the environment. Tests pass their own dict as env."""
    if env is None:
        load_dotenv()  # real environment variables still win over .env
        env = os.environ

    try:
        app_today = date.fromisoformat(env.get("APP_TODAY", "2026-07-01").strip())
    except ValueError:
        raise ValueError("APP_TODAY must look like 2026-07-01 (YYYY-MM-DD).")

    # LLM_API_KEY is the key for whichever provider LLM_PROVIDER names (blank = groq).
    api_key = env.get("LLM_API_KEY", "").strip()  # a blank line in .env counts as unset
    provider = env.get("LLM_PROVIDER", "").strip().lower()
    if not provider:
        # Old .env files had no LLM_API_KEY: the provider was whichever one had its own key variable set.
        old_style = next((name for name, (_, key_name, _) in PROVIDERS.items() if env.get(key_name, "").strip()), None)
        provider = DEFAULT_PROVIDER if api_key or not old_style else old_style
    base_url, old_key_name, default_model = PROVIDERS.get(provider, PROVIDERS[DEFAULT_PROVIDER])

    return Settings(
        policies_path=env.get("POLICIES_PATH", "./data/policies.md"),
        manifest_path=env.get("MANIFEST_PATH", "./data/manifest.json"),
        db_path=env.get("DB_PATH", "./data/shopsphere.db"),
        app_today=app_today,
        llm_model=(env.get("LLM_MODEL", "").strip() or default_model),
        llm_timeout_seconds=float(env.get("LLM_TIMEOUT_SECONDS", "20")),
        llm_provider=provider,
        llm_base_url=base_url,
        embedding_model=env.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip(),
        retrieval_top_k=int(env.get("RETRIEVAL_TOP_K", "5")),
        retrieval_min_score=float(env.get("RETRIEVAL_MIN_SCORE", "0.3")),
        llm_api_key=api_key or env.get(old_key_name, "").strip(),
    )
