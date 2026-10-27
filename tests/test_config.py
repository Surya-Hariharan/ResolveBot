import pytest

from app.config import load_settings


def test_defaults_match_the_assignment():
    settings = load_settings({})
    assert settings.policies_path == "./data/policies.md"
    assert settings.manifest_path == "./data/manifest.json"
    assert settings.db_path == "./data/shopsphere.db"
    assert str(settings.app_today) == "2026-07-01"
    assert settings.llm_provider == "groq"


def test_generic_key_wins_over_provider_key():
    settings = load_settings({"LLM_API_KEY": "generic", "GROQ_API_KEY": "specific"})
    assert settings.llm_api_key == "generic"


def test_provider_key_is_used_when_generic_is_unset():
    assert load_settings({"GROQ_API_KEY": "specific"}).llm_api_key == "specific"


def test_blank_values_count_as_unset():
    settings = load_settings({"LLM_API_KEY": "  ", "GROQ_API_KEY": ""})
    assert settings.llm_api_key == ""
    assert "No LLM API key" in settings.llm_problem()


def test_another_providers_key_is_not_borrowed():
    settings = load_settings({"LLM_PROVIDER": "groq", "OPENAI_API_KEY": "sk-other"})
    assert settings.llm_api_key == ""


def test_openai_provider_reads_openai_key():
    settings = load_settings({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-x"})
    assert settings.llm_api_key == "sk-x"
    assert settings.llm_problem() is None


def test_provider_without_adapter_is_reported_honestly():
    settings = load_settings({"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k"})
    assert "not implemented" in settings.llm_problem()


def test_unknown_provider_is_reported():
    assert "not a known provider" in load_settings({"LLM_PROVIDER": "nope"}).llm_problem()


def test_key_never_appears_in_repr_or_problem_text():
    settings = load_settings({"LLM_PROVIDER": "anthropic", "LLM_API_KEY": "SECRET-123"})
    assert "SECRET-123" not in repr(settings)
    assert "SECRET-123" not in settings.llm_problem()


def test_bad_app_today_fails_loudly():
    with pytest.raises(ValueError):
        load_settings({"APP_TODAY": "July 1"})
