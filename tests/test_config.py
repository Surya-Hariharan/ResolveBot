import pytest

from app.agent import llm
from app.config import PROVIDERS, load_settings


def test_defaults_match_the_assignment():
    settings = load_settings({})
    assert settings.policies_path == "./data/policies.md"
    assert settings.manifest_path == "./data/manifest.json"
    assert settings.db_path == "./data/shopsphere.db"
    assert str(settings.app_today) == "2026-07-01"


def test_key_is_read_from_llm_api_key():
    settings = load_settings({"LLM_API_KEY": "specific"})
    assert settings.llm_api_key == "specific"
    assert settings.llm_problem() is None


def test_blank_values_count_as_unset():
    settings = load_settings({"LLM_API_KEY": "  "})
    assert settings.llm_api_key == ""
    assert "LLM_API_KEY" in settings.llm_problem()


def test_provider_is_chosen_by_llm_provider_and_uses_the_one_key():
    for name in PROVIDERS:
        settings = load_settings({"LLM_PROVIDER": name.upper(), "LLM_API_KEY": "k"})
        assert settings.llm_provider == name and settings.llm_api_key == "k"
        assert settings.llm_problem() is None and settings.llm_base_url.startswith("https://")


def test_blank_provider_means_groq():
    settings = load_settings({"LLM_API_KEY": "k"})
    assert settings.llm_provider == "groq" and settings.llm_model == "openai/gpt-oss-120b"
    assert load_settings({}).llm_provider == "groq"  # nothing set: the problem text names LLM_API_KEY


def test_missing_key_fails_safely_for_every_provider():
    for name in PROVIDERS:
        problem = load_settings({"LLM_PROVIDER": name}).llm_problem()
        assert "LLM_API_KEY" in problem and name in problem


def test_missing_key_stops_the_call_before_any_request(monkeypatch):
    monkeypatch.setattr(llm, "OpenAI", lambda **_: pytest.fail("no client should be built without a key"))
    with pytest.raises(llm.LLMNotAvailable, match="LLM_API_KEY"):
        llm.chat(load_settings({"LLM_PROVIDER": "openai"}), [{"role": "user", "content": "hi"}])


def test_unknown_provider_gives_a_clear_safe_problem_not_a_crash():
    settings = load_settings({"LLM_PROVIDER": "mystery", "LLM_API_KEY": "SECRET-123"})
    assert "LLM_PROVIDER must be one of" in settings.llm_problem() and "SECRET-123" not in settings.llm_problem()


def test_default_model_follows_the_provider_and_llm_model_overrides_it():
    assert load_settings({"LLM_PROVIDER": "openai"}).llm_model == "gpt-4o-mini"
    assert load_settings({"LLM_PROVIDER": "openai", "LLM_MODEL": "  "}).llm_model == "gpt-4o-mini"  # blank = default
    assert load_settings({}).llm_model == "openai/gpt-oss-120b"
    assert load_settings({"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-x"}).llm_model == "gpt-x"
    # no other provider's default is the groq model name
    assert [n for n, (_, _, model) in PROVIDERS.items() if model == PROVIDERS["groq"][2]] == ["groq"]


def test_old_provider_specific_variables_still_work_as_a_fallback():
    # An .env written before LLM_API_KEY existed keeps working without being edited.
    old = load_settings({"GROQ_API_KEY": "gq"})
    assert old.llm_provider == "groq" and old.llm_api_key == "gq" and old.llm_problem() is None
    assert load_settings({"OPENAI_API_KEY": "sk"}).llm_provider == "openai"
    assert load_settings({"LLM_PROVIDER": "google", "GOOGLE_API_KEY": "g"}).llm_api_key == "g"
    # the new variable wins when both are present, and a key for another provider is never used
    assert load_settings({"LLM_API_KEY": "new", "GROQ_API_KEY": "old"}).llm_api_key == "new"
    assert load_settings({"LLM_PROVIDER": "openai", "GROQ_API_KEY": "gq"}).llm_problem() is not None


def test_key_never_appears_in_repr_or_problem_text():
    settings = load_settings({"LLM_API_KEY": "SECRET-123"})
    assert "SECRET-123" not in repr(settings)
    assert settings.llm_problem() is None
    assert "SECRET-123" not in repr(load_settings({}).llm_problem())


def test_bad_app_today_fails_loudly():
    with pytest.raises(ValueError):
        load_settings({"APP_TODAY": "July 1"})
