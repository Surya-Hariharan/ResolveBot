from datetime import date
from pathlib import Path

import pytest

from app.config import load_settings
from app.retrieval import policy_loader
from app.retrieval.search import PolicySearch

FIXTURES = Path(__file__).parent / "fixtures"
POLICIES = FIXTURES / "policies.md"
MANIFEST = FIXTURES / "manifest.json"


def chunks_on(day):
    return policy_loader.load_chunks(POLICIES, MANIFEST, day)


def ids(chunks):
    return {(c.doc_id, c.section) for c in chunks}


# ------------------------------------------------------------ loading/filtering
def test_documents_are_split_at_each_title_and_matched_by_doc_id():
    docs = policy_loader.split_documents(POLICIES.read_text(encoding="utf-8"))
    assert [doc_id for doc_id, _ in docs] == [
        "returns_policy", "returns_policy_old", "shipping_policy", "shipping_update", "faq_general",
    ]


def test_sections_are_cut_at_the_numbered_headings():
    found = ids(chunks_on(date(2026, 7, 1)))
    assert ("returns_policy", "§2.1") in found and ("returns_policy", "§2.2") in found
    chunk = next(c for c in chunks_on(date(2026, 7, 1)) if c.section == "§2.1" and c.doc_id == "returns_policy")
    assert "21 days" in chunk.text and "Opened earbuds" not in chunk.text
    assert chunk.heading.startswith("§2.1") and chunk.parent_heading.startswith("§2")


def test_superseded_documents_never_appear():
    assert "returns_policy_old" not in {c.doc_id for c in chunks_on(date(2026, 7, 1))}
    assert "returns_policy_old" not in {c.doc_id for c in chunks_on(date(2030, 1, 1))}


def test_documents_not_yet_effective_are_marked_not_in_force():
    before = {c.doc_id: c.in_force for c in chunks_on(date(2026, 7, 1))}
    assert before["shipping_update"] is False and before["returns_policy"] is True
    on_the_day = {c.doc_id: c.in_force for c in chunks_on(date(2026, 9, 1))}
    assert on_the_day["shipping_update"] is True  # the effective date itself counts


def test_effective_date_follows_app_today_not_the_computer_clock():
    assert next(c for c in chunks_on(date(2026, 8, 31)) if c.doc_id == "shipping_update").in_force is False
    assert next(c for c in chunks_on(date(2026, 9, 2)) if c.doc_id == "shipping_update").in_force is True


def test_unknown_documents_without_a_manifest_entry_are_ignored(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    assert policy_loader.load_chunks(POLICIES, manifest, date(2026, 7, 1)) == []


def test_support_email_is_read_from_the_documents():
    assert policy_loader.find_support_email(chunks_on(date(2026, 7, 1))) == "help@shop.example.com"


def test_long_sections_are_split_without_crossing_sections():
    chunks = chunks_on(date(2026, 7, 1))
    pieces = policy_loader.split_long_chunks(chunks, lambda text: len(text.split()), max_tokens=8)
    assert len(pieces) > len(chunks)
    assert ids(pieces) == ids(chunks)  # every piece still belongs to a real section


def test_real_policies_parse_into_ten_documents_if_present():
    root = Path(__file__).resolve().parent.parent / "data"
    if not (root / "policies.md").exists():
        pytest.skip("data/ folder is not present")
    docs = policy_loader.split_documents((root / "policies.md").read_text(encoding="utf-8"))
    assert len(docs) == 10
    chunks = policy_loader.load_chunks(root / "policies.md", root / "manifest.json", date(2026, 7, 1))
    assert "returns_policy_2024" not in {c.doc_id for c in chunks}
    assert ("returns_policy", "§2.1") in ids(chunks)
    assert {c.doc_id for c in chunks if not c.in_force} == {"shipping_policy_update"}


# ----------------------------------------------------------------- ranking
def keyword_search(day=date(2026, 7, 1)):
    search = PolicySearch(chunks=chunks_on(day), model_name="keyword-only", top_k=5, min_score=0.3)
    search._ready.set()
    return search


def test_keyword_search_finds_the_right_section_first():
    top = keyword_search().search("return window days")[0]
    assert (top["doc_id"], top["section"]) == ("returns_policy", "§2.1")


def test_an_unrelated_question_returns_nothing():
    assert keyword_search().search("what is the weather on mars") == []


def test_results_carry_what_the_llm_needs_to_cite():
    for paragraph in keyword_search().search("return window days"):
        assert {"doc_id", "section", "title", "text"} <= set(paragraph)


def test_dense_search_with_the_real_embedding_model():
    search = PolicySearch(chunks=chunks_on(date(2026, 7, 1)), model_name="sentence-transformers/all-MiniLM-L6-v2",
                          top_k=5, min_score=0.3)
    search.warm_up()
    if search.load_error:
        pytest.skip(f"embedding model unavailable: {search.load_error}")
    assert search.method == "dense"
    top = search.search("How many days do I have to return something?")[0]
    assert (top["doc_id"], top["section"]) == ("returns_policy", "§2.1")
    assert all(c.doc_id != "returns_policy_old" for c, _ in search.rank("return window"))


def test_embedding_failure_falls_back_to_keywords():
    search = PolicySearch(chunks=chunks_on(date(2026, 7, 1)), model_name="no-such-org/no-such-model",
                          top_k=5, min_score=0.3)
    search.warm_up()
    assert search.load_error and search.method == "keyword"
    assert search.search("return window days")  # still answers


def test_bge_and_e5_get_their_required_prefixes():
    from app.retrieval.search import prefixes_for

    assert prefixes_for("intfloat/e5-small-v2") == {"query": "query: ", "passage": "passage: "}
    assert prefixes_for("BAAI/bge-small-en-v1.5")["query"].startswith("Represent this sentence")
    assert prefixes_for("sentence-transformers/all-MiniLM-L6-v2") == {"query": "", "passage": ""}


def test_search_settings_come_from_config():
    settings = load_settings({})
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.retrieval_top_k == 5
    custom = load_settings({"EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5", "RETRIEVAL_TOP_K": "3", "RETRIEVAL_MIN_SCORE": "0.4"})
    assert (custom.embedding_model, custom.retrieval_top_k, custom.retrieval_min_score) == ("BAAI/bge-small-en-v1.5", 3, 0.4)


def test_the_customers_own_words_rescue_a_poor_llm_query():
    search = keyword_search()
    assert search.search("xyzzy") == []
    rescued = search.search("xyzzy", extra_query="How many days can items be returned within?")
    assert (rescued[0]["doc_id"], rescued[0]["section"]) == ("returns_policy", "§2.1")
