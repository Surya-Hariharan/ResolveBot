"""End-to-end tests of /chat with a scripted LLM: tools, routes, citations, the yes-step."""
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app
from app.retrieval.policy_loader import load_chunks
from app.retrieval.search import PolicySearch
from tests.make_test_db import TODAY, count_rows, make_db
from tests.scripted_llm import FakeLLM, call, calls, say

FIXTURES = Path(__file__).parent / "fixtures"
NOT_FOUND_TEXT = "I couldn't find that order on your account."
RETURN_ARGS = {"order_id": "SS-1", "item_ids": ["SS-1-1"], "reason": "changed_mind"}


@pytest.fixture
def db(tmp_path):
    return make_db(tmp_path / "agent.db")


@pytest.fixture
def policy_search():
    chunks = load_chunks(FIXTURES / "policies.md", FIXTURES / "manifest.json", TODAY)
    search = PolicySearch(chunks=chunks, model_name="keyword-only", top_k=3, min_score=0.3)
    search._ready.set()  # no embedding model in unit tests: keyword matching is used
    return search


def make_client(db, policy_search, *steps):
    settings = load_settings({"DB_PATH": db, "LLM_API_KEY": "test-key-not-real"})
    fake = FakeLLM(*steps)
    return TestClient(create_app(settings, llm_chat=fake, policy_search=policy_search)), fake


def send(client, message, session="s1", customer="C1001"):
    response = client.post(
        "/chat", json={"session_id": session, "message": message}, headers={"X-Customer-Id": customer}
    )
    assert response.status_code == 200, response.text
    return response.json()


def proposal_steps(args=RETURN_ARGS):
    return [
        calls(call("get_orders", order_id=args["order_id"])),
        calls(call("check_return_eligibility", **args)),
        calls(call("create_return_request", **args)),
        say("You can return it. Shall I go ahead?\nROUTE: answer\nSOURCES: none"),
    ]


def tool_summary(reply):
    return [(t["name"], t["status"]) for t in reply["tool_calls"]]


# ------------------------------------------------------- propose, then confirm
def test_return_is_proposed_without_saving_then_saved_on_yes(db, policy_search):
    client, _ = make_client(db, policy_search, *proposal_steps())
    first = send(client, "Return the speaker from SS-1, I changed my mind.")
    assert first["route"] == "answer" and first["pending_confirmation"] is True
    assert tool_summary(first)[-1] == ("create_return_request", "pending_confirmation")
    assert count_rows(db, "return_requests") == 0  # nothing saved yet

    second = send(client, "Yes, go ahead.")
    assert second["pending_confirmation"] is False
    assert tool_summary(second) == [("create_return_request", "success")]
    assert count_rows(db, "return_requests") == 1
    saved = sqlite3.connect(db).execute("SELECT customer_id, item_ids, reason, status FROM return_requests").fetchone()
    assert saved == ("C1001", '["SS-1-1"]', "changed_mind", "requested")


def test_reply_has_all_required_fields_with_the_right_types(db, policy_search):
    client, _ = make_client(db, policy_search, *proposal_steps())
    reply = send(client, "Return the speaker from SS-1, I changed my mind.")
    assert set(reply) >= {"request_id", "answer", "route", "citations", "tool_calls", "pending_confirmation"}
    for entry in reply["tool_calls"]:
        assert set(entry) == {"name", "args", "status"}


def test_the_yes_step_does_not_call_the_llm(db, policy_search):
    client, fake = make_client(db, policy_search, *proposal_steps())
    send(client, "Return the speaker from SS-1, I changed my mind.")
    calls_before = len(fake.requests)
    send(client, "yes")
    assert len(fake.requests) == calls_before


@pytest.mark.parametrize("answer", ["no", "No thanks", "maybe", "ok", "hmm let me think", "yes but only the charger"])
def test_anything_but_a_clear_yes_saves_nothing(db, policy_search, answer):
    client, _ = make_client(db, policy_search, *proposal_steps(), say("No problem.\nROUTE: answer\nSOURCES: none"))
    send(client, "Return the speaker from SS-1, I changed my mind.")
    reply = send(client, answer)
    assert reply["pending_confirmation"] is False
    assert count_rows(db, "return_requests") == 0


def test_a_yes_after_a_no_does_not_revive_the_old_proposal(db, policy_search):
    client, _ = make_client(
        db, policy_search, *proposal_steps(),
        say("Okay, I won't do it.\nROUTE: answer\nSOURCES: none"),
        say("Which order do you mean?\nROUTE: clarify\nSOURCES: none"),
    )
    send(client, "Return the speaker from SS-1, I changed my mind.")
    send(client, "no")
    reply = send(client, "yes")  # nothing is waiting any more, so this goes to the LLM
    assert reply["route"] == "clarify"
    assert count_rows(db, "return_requests") == 0


def test_changing_the_details_makes_a_new_proposal(db, policy_search):
    new_args = {"order_id": "SS-4", "item_ids": ["SS-4-2"], "reason": "changed_mind"}
    client, _ = make_client(
        db, policy_search, *proposal_steps(),
        calls(call("check_return_eligibility", **new_args)),
        calls(call("create_return_request", **new_args)),
        say("Just the speaker from SS-4? Shall I go ahead?\nROUTE: answer\nSOURCES: none"),
    )
    send(client, "Return the speaker from SS-1, I changed my mind.")
    reply = send(client, "Actually, the speaker from SS-4 instead")
    assert reply["pending_confirmation"] is True
    assert count_rows(db, "return_requests") == 0
    send(client, "yes")
    row = sqlite3.connect(db).execute("SELECT order_id, item_ids FROM return_requests").fetchall()
    assert row == [("SS-4", '["SS-4-2"]')]


def test_just_do_it_with_nothing_waiting_saves_nothing(db, policy_search):
    client, _ = make_client(db, policy_search, say("What would you like me to do?\nROUTE: clarify\nSOURCES: none"))
    reply = send(client, "just do it")
    assert reply["route"] == "clarify" and count_rows(db, "return_requests") == 0


def test_eligibility_is_rechecked_when_the_customer_says_yes(db, policy_search):
    client, _ = make_client(db, policy_search, *proposal_steps())
    send(client, "Return the speaker from SS-1, I changed my mind.")
    # the data changes while the proposal is waiting: the window is now very short
    connection = sqlite3.connect(db)
    connection.execute("UPDATE policy_parameters SET value = '3' WHERE key = 'return_window_days'")
    connection.commit()
    connection.close()
    reply = send(client, "yes")
    assert tool_summary(reply) == [("create_return_request", "error")]
    assert "Nothing was saved" in reply["answer"]
    assert count_rows(db, "return_requests") == 0


def test_only_one_proposal_is_kept_per_message(db, policy_search):
    ticket = {"order_id": "SS-6", "ticket_type": "cancellation", "summary": "Cancel it"}
    client, _ = make_client(
        db, policy_search,
        calls(call("create_return_request", **RETURN_ARGS), call("create_support_ticket", **ticket)),
        say("I can start the return. Shall I go ahead?\nROUTE: answer\nSOURCES: none"),
    )
    reply = send(client, "Return SS-1 and cancel SS-6")
    assert tool_summary(reply) == [("create_return_request", "pending_confirmation"), ("create_support_ticket", "error")]
    send(client, "yes")
    assert count_rows(db, "return_requests") == 1 and count_rows(db, "support_tickets") == 0


def test_cancellation_and_warranty_tickets_are_proposed_then_saved(db, policy_search):
    ticket = {"order_id": "SS-6", "ticket_type": "cancellation", "summary": "Customer asked to cancel"}
    client, _ = make_client(
        db, policy_search,
        calls(call("create_support_ticket", **ticket)),
        say("I'll open a cancellation ticket for SS-6. Shall I go ahead?\nROUTE: answer\nSOURCES: none"),
    )
    reply = send(client, "Please cancel order SS-6")
    assert reply["pending_confirmation"] and count_rows(db, "support_tickets") == 0
    send(client, "yes please")
    assert count_rows(db, "support_tickets") == 1


def test_a_shipped_order_cannot_even_be_proposed_for_cancellation(db, policy_search):
    ticket = {"order_id": "SS-7", "ticket_type": "cancellation", "summary": "Cancel"}
    client, _ = make_client(
        db, policy_search,
        calls(call("create_support_ticket", **ticket)),
        say("SS-7 has already shipped so it can't be cancelled.\nROUTE: answer\nSOURCES: none"),
    )
    reply = send(client, "Please cancel order SS-7")
    assert reply["pending_confirmation"] is False
    assert tool_summary(reply) == [("create_support_ticket", "error")]
    assert count_rows(db, "support_tickets") == 0


def test_proposal_text_always_asks_for_confirmation(db, policy_search):
    client, _ = make_client(
        db, policy_search,
        calls(call("create_return_request", **RETURN_ARGS)),
        say("I will return the speaker.\nROUTE: answer\nSOURCES: none"),
    )
    assert "Shall I go ahead?" in send(client, "return SS-1")["answer"]


# ------------------------------------------------------------ other routes
def test_foreign_and_missing_orders_get_the_exact_same_reply(db, policy_search):
    replies = []
    for order_id in ("SS-9", "SS-999"):  # SS-9 belongs to C1002
        client, _ = make_client(
            db, policy_search,
            calls(call("get_orders", order_id=order_id)),
            say("Order not found, sorry!\nROUTE: answer\nSOURCES: none"),
        )
        replies.append(send(client, f"Where is order {order_id}?"))
    for reply in replies:
        assert reply["answer"] == NOT_FOUND_TEXT
        assert reply["route"] == "no_answer" and reply["citations"] == []
    # nothing else about the two replies differs, apart from the id the customer typed
    assert replies[0]["answer"] == replies[1]["answer"]
    assert [t["status"] for t in replies[0]["tool_calls"]] == [t["status"] for t in replies[1]["tool_calls"]] == ["error"]


def test_foreign_order_return_is_refused_and_nothing_is_saved(db, policy_search):
    args = {"order_id": "SS-9", "item_ids": ["SS-9-1"], "reason": "changed_mind"}
    client, _ = make_client(
        db, policy_search,
        calls(call("create_return_request", **args)),
        say("ok\nROUTE: answer\nSOURCES: none"),
    )
    reply = send(client, "return SS-9")
    assert reply["answer"] == NOT_FOUND_TEXT and reply["pending_confirmation"] is False
    send(client, "yes")
    assert count_rows(db, "return_requests") == 0


def test_the_llm_cannot_choose_the_customer(db, policy_search):
    client, _ = make_client(
        db, policy_search,
        calls(call("get_orders", order_id="SS-9", customer_id="C1002")),  # C1001 is calling
        say("x\nROUTE: answer\nSOURCES: none"),
    )
    reply = send(client, "show SS-9")
    assert reply["answer"] == NOT_FOUND_TEXT
    assert reply["tool_calls"][0]["args"] == {"order_id": "SS-9"}  # customer_id was dropped


def test_off_topic_message_is_out_of_scope_with_nothing_listed(db, policy_search):
    client, _ = make_client(
        db, policy_search, say("I can only help with ShopSphere orders and policies.\nROUTE: out_of_scope\nSOURCES: none")
    )
    reply = send(client, "Write me a poem about rain")
    assert reply["route"] == "out_of_scope"
    assert reply["citations"] == [] and reply["tool_calls"] == []


def test_missing_details_get_a_clarify_route(db, policy_search):
    client, _ = make_client(
        db, policy_search,
        calls(call("get_orders")),
        say("Which order and item? SS-1 (speaker) or SS-4 (earbuds)?\nROUTE: clarify\nSOURCES: none"),
    )
    reply = send(client, "return my charger")
    assert reply["route"] == "clarify" and reply["pending_confirmation"] is False


def test_order_lookup_is_a_normal_answer(db, policy_search):
    client, fake = make_client(
        db, policy_search,
        calls(call("get_orders", order_id="SS-7")),
        say("SS-7 is shipped, expected 4 July 2026.\nROUTE: answer\nSOURCES: none"),
    )
    reply = send(client, "Where is order SS-7?")
    assert reply["route"] == "answer" and reply["citations"] == []
    assert "ROUTE" not in reply["answer"] and "SOURCES" not in reply["answer"]
    tool_message = [m for m in fake.requests[-1]["messages"] if m["role"] == "tool"][0]
    assert "4 July 2026" in tool_message["content"]


# ------------------------------------------------------------------- policy
def test_policy_answer_cites_only_the_sections_used(db, policy_search):
    client, _ = make_client(
        db, policy_search,
        calls(call("search_policy", query="return window days")),
        say("You have 21 days.\nROUTE: answer\nSOURCES: returns_policy §2.1"),
    )
    reply = send(client, "How many days do I have to return something?")
    assert reply["citations"] == [{"doc_id": "returns_policy", "section": "§2.1"}]
    assert reply["tool_calls"][0]["name"] == "search_policy"


def test_made_up_citations_are_dropped(db, policy_search):
    client, _ = make_client(
        db, policy_search,
        calls(call("search_policy", query="return window days")),
        say("21 days.\nROUTE: answer\nSOURCES: returns_policy §2.1; invented_doc §9.9"),
    )
    reply = send(client, "return window?")
    assert reply["citations"] == [{"doc_id": "returns_policy", "section": "§2.1"}]


def test_policy_with_no_match_is_no_answer_with_support_email(db, policy_search):
    client, _ = make_client(
        db, policy_search,
        calls(call("search_policy", query="bitcoin payments crypto")),
        say("We take bitcoin!\nROUTE: answer\nSOURCES: none"),
    )
    reply = send(client, "Can I pay with bitcoin?")
    assert reply["route"] == "no_answer" and reply["citations"] == []
    assert "help@shop.example.com" in reply["answer"]


def test_search_only_returns_current_documents(policy_search):
    found = policy_search.search("return window days")
    assert {p["doc_id"] for p in found} <= {"returns_policy", "faq_general"}
    assert "returns_policy_old" not in {c.doc_id for c in policy_search.chunks}


def test_future_policy_is_marked_not_in_force(policy_search):
    found = policy_search.search("new shipping fee rupees per order")
    future = [p for p in found if p["doc_id"] == "shipping_update"]
    assert future and "NOT IN FORCE YET" in future[0]["note"] and "1 September 2026" in future[0]["note"]
    current = [p for p in found if p["doc_id"] == "shipping_policy"]
    assert current and "note" not in current[0]


def test_policy_paragraphs_come_before_faq_paragraphs(policy_search):
    found = policy_search.search("return items within days of delivery")
    types = [p["doc_id"] == "faq_general" for p in found]
    assert types == sorted(types)  # all policy paragraphs first


# -------------------------------------------------- limits and failure modes
def test_tool_calls_are_capped_at_six(db, policy_search):
    steps = [calls(call("get_orders")) for _ in range(10)] + [say("I did my best.\nROUTE: answer\nSOURCES: none")]
    client, fake = make_client(db, policy_search, *steps)
    reply = send(client, "do everything")
    assert len(reply["tool_calls"]) == 6
    assert reply["answer"]  # polite reply, not a crash
    assert fake.requests[-1]["tools"] is None  # the last request offered no tools


def test_a_model_that_never_stops_asking_for_tools_still_gets_a_reply(db, policy_search):
    client, _ = make_client(db, policy_search, *[calls(call("get_orders")) for _ in range(20)])
    reply = send(client, "loop forever")
    assert len(reply["tool_calls"]) == 6 and "one thing at a time" in reply["answer"]


def test_unknown_tool_and_bad_arguments_do_not_crash(db, policy_search):
    bad_json = calls(SimpleCall("get_orders", "{not json"))
    client, _ = make_client(
        db, policy_search,
        calls(call("delete_everything")), bad_json,
        say("Sorry.\nROUTE: answer\nSOURCES: none"),
    )
    reply = send(client, "hello")
    assert reply["route"] == "answer"
    assert [t["name"] for t in reply["tool_calls"]] == ["get_orders"]  # unknown tool is not listed


def SimpleCall(name, raw_arguments):
    from types import SimpleNamespace

    return SimpleNamespace(id="call_x", function=SimpleNamespace(name=name, arguments=raw_arguments))


@pytest.mark.parametrize("problem", [TimeoutError("slow"), RuntimeError("boom"), ValueError("secret-test-key-not-real")])
def test_llm_failures_give_a_controlled_200_reply(db, policy_search, problem):
    client, _ = make_client(db, policy_search, problem)
    reply = send(client, "hello")
    assert reply["route"] == "no_answer" and reply["answer"]
    assert "test-key-not-real" not in reply["answer"]


def test_provider_errors_are_described_without_raw_details(db, policy_search):
    class RateLimitError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    client, _ = make_client(db, policy_search, RateLimitError("429 with details"))
    assert "busy" in send(client, "hi")["answer"]
    client, _ = make_client(db, policy_search, APITimeoutError("timed out"))
    assert "too long" in send(client, "hi")["answer"]


def test_llm_receives_a_time_budget_under_thirty_seconds(db, policy_search):
    client, fake = make_client(db, policy_search, say("hi\nROUTE: out_of_scope\nSOURCES: none"))
    send(client, "hi")
    assert 0 < fake.requests[0]["timeout"] < 30


def test_session_memory_is_passed_to_the_llm(db, policy_search):
    client, fake = make_client(
        db, policy_search,
        say("Which order?\nROUTE: clarify\nSOURCES: none"),
        say("Got it.\nROUTE: answer\nSOURCES: none"),
    )
    send(client, "return my charger")
    send(client, "SS-6")
    contents = [m["content"] for m in fake.requests[1]["messages"] if m["role"] in ("user", "assistant")]
    assert contents == ["return my charger", "Which order?", "SS-6"]  # tags were stripped from memory


def test_system_prompt_contains_no_policy_numbers(db, policy_search):
    client, fake = make_client(db, policy_search, say("hi\nROUTE: out_of_scope\nSOURCES: none"))
    send(client, "hi")
    prompt = fake.requests[0]["messages"][0]["content"]
    for number in ("30", "12 months", "24", "99", "2999", "48"):
        assert number not in prompt


def test_a_waiting_proposal_can_be_saved_even_if_the_llm_key_disappears(db, policy_search):
    # the yes-step is plain Python, so it does not need the LLM
    settings = load_settings({"DB_PATH": db, "LLM_API_KEY": "test-key-not-real"})
    client = TestClient(create_app(settings, llm_chat=FakeLLM(*proposal_steps()), policy_search=policy_search))
    send(client, "Return the speaker from SS-1, I changed my mind.")
    settings.llm_api_key = ""
    assert send(client, "confirm")["tool_calls"][0]["status"] == "success"
    assert count_rows(db, "return_requests") == 1
    assert "API key" in send(client, "anything else?")["answer"]  # other messages get the controlled reply


def test_customers_have_separate_sessions_and_data(db, policy_search):
    client, _ = make_client(
        db, policy_search,
        calls(call("get_orders")),
        say("Here are your orders.\nROUTE: answer\nSOURCES: none"),
    )
    reply = send(client, "my orders", session="shared", customer="C1002")
    assert reply["tool_calls"][0]["status"] == "success"
    forbidden = client.post(
        "/chat", json={"session_id": "shared", "message": "hi"}, headers={"X-Customer-Id": "C1001"}
    )
    assert forbidden.status_code == 403


def test_lookalike_hyphens_and_spaces_in_the_answer_are_made_ordinary(db, policy_search):
    # Regression: the real model wrote "SS\u201110512" (non-breaking hyphen) and "3\u202fJuly", which broke plain-text matching.
    text = "Order SS\u20111 arrives on 3\u202fJuly\u00a02026.\nROUTE: answer\nSOURCES: none"
    client, _ = make_client(db, policy_search, say(text))
    reply = send(client, "Where is my order?")
    assert reply["answer"] == "Order SS-1 arrives on 3 July 2026."


# ---- regressions found with the real model (gpt-oss) ----
@pytest.mark.parametrize("invented", ["SS-9999", ""])  # the real model sent "" over HTTP for "return something"
def test_an_order_id_the_customer_never_gave_is_not_looked_up(db, policy_search, invented):
    # The model once called get_orders with an id it made up, so the customer was told "order not found".
    client, _ = make_client(
        db, policy_search,
        calls(call("get_orders", order_id=invented)),
        say("Which order would you like to return?\nROUTE: clarify\nSOURCES: none"),
    )
    reply = send(client, "I want to return something.")
    assert tool_summary(reply) == [("get_orders", "error")]
    assert reply["route"] == "clarify"
    assert NOT_FOUND_TEXT not in reply["answer"]


def test_an_order_id_from_an_earlier_tool_result_may_be_used(db, policy_search):
    client, _ = make_client(
        db, policy_search,
        calls(call("get_orders")),
        calls(call("get_orders", order_id="SS-1")),  # SS-1 was shown by the first result
        say("Here it is.\nROUTE: answer\nSOURCES: none"),
    )
    reply = send(client, "What orders do I have?")
    assert tool_summary(reply) == [("get_orders", "success"), ("get_orders", "success")]


@pytest.mark.parametrize("text, route", [
    ("I'm sorry, I can't help with that.", "out_of_scope"),
    ("Which order do you mean?", "clarify"),
])
def test_a_missing_route_tag_without_tools_is_guessed_safely(db, policy_search, text, route):
    # gpt-oss-20b sometimes left the ROUTE line out of a refusal.
    client, _ = make_client(db, policy_search, say(text))
    reply = send(client, "What's the weather today?")
    assert reply["route"] == route and reply["tool_calls"] == [] and reply["citations"] == []


def test_a_missing_route_tag_after_tools_is_an_answer(db, policy_search):
    client, _ = make_client(db, policy_search, calls(call("get_orders")), say("You have one order."))
    assert send(client, "What orders do I have?")["route"] == "answer"


def test_groq_tool_use_failed_is_retried_once_but_other_400s_are_not(monkeypatch):
    from types import SimpleNamespace

    from openai import BadRequestError

    from app.agent import llm

    def make_error(code):
        error = BadRequestError.__new__(BadRequestError)
        error.code = code
        return error

    class FakeCompletions:
        def __init__(self, outcomes):
            self.outcomes, self.calls = list(outcomes), 0

        def create(self, **request):
            self.calls += 1
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return SimpleNamespace(choices=[SimpleNamespace(message="ok")])

    def run(outcomes):
        completions = FakeCompletions(outcomes)
        monkeypatch.setattr(llm, "OpenAI", lambda **_: SimpleNamespace(chat=SimpleNamespace(completions=completions)))
        settings = load_settings({"LLM_API_KEY": "test-key-not-real"})
        try:
            return llm.chat(settings, [{"role": "user", "content": "hi"}]), completions.calls
        except BadRequestError:
            return "raised", completions.calls

    assert run([make_error("tool_use_failed"), None]) == ("ok", 2)
    assert run([make_error("tool_use_failed"), make_error("tool_use_failed")]) == ("raised", 2)
    assert run([make_error("context_length_exceeded")]) == ("raised", 1)


def test_a_failed_reply_does_not_claim_a_proposal_is_waiting(db, policy_search):
    # Seen live: the proposal tool ran, then the provider returned 429 on the last call.
    class RateLimitError(Exception):
        pass

    steps = proposal_steps()[:3] + [RateLimitError("429")]
    client, _ = make_client(db, policy_search, *steps)
    reply = send(client, "Return the speaker from SS-1, I changed my mind.")
    assert reply["pending_confirmation"] is False and reply["route"] == "no_answer"
    assert tool_summary(reply)[-1] == ("create_return_request", "error")
    assert send(client, "yes")["tool_calls"] == []  # nothing was waiting
    assert count_rows(db, "return_requests") == 0


@pytest.mark.parametrize("provider, base_url_part, token_parameter, default_model", [
    ("groq", "groq.com", "max_tokens", "openai/gpt-oss-120b"),
    ("openai", "api.openai.com", "max_completion_tokens", "gpt-4o-mini"),
    ("anthropic", "anthropic.com", "max_tokens", "claude-haiku-5-5"),
    ("google", "generativelanguage.googleapis.com", "max_tokens", "gemini-2.0-flash"),
])
def test_each_provider_gets_its_own_url_model_and_parameters(monkeypatch, provider, base_url_part, token_parameter, default_model):
    # Mocked: no network. This shows the request is built correctly, not that the provider accepts it.
    from types import SimpleNamespace

    from app.agent import llm

    seen = {}

    class FakeCompletions:
        def create(self, **request):
            seen["request"] = request
            return SimpleNamespace(choices=[SimpleNamespace(message="ok")])

    def fake_client(**kwargs):
        seen["client"] = kwargs
        return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    monkeypatch.setattr(llm, "OpenAI", fake_client)
    settings = load_settings({"LLM_PROVIDER": provider, "LLM_API_KEY": "test-key-not-real"})
    assert llm.chat(settings, [{"role": "user", "content": "hi"}]) == "ok"
    assert seen["client"]["api_key"] == "test-key-not-real" and base_url_part in seen["client"]["base_url"]
    assert seen["client"]["max_retries"] == 0
    assert seen["request"]["model"] == default_model  # blank LLM_MODEL: this provider's own default
    assert llm.chat(load_settings({"LLM_PROVIDER": provider, "LLM_API_KEY": "k", "LLM_MODEL": "my-model"}),
                    [{"role": "user", "content": "hi"}]) == "ok"
    assert seen["request"]["model"] == "my-model"  # an explicit override is passed through unchanged
    assert seen["request"]["temperature"] == 0 and seen["request"][token_parameter] == llm.MAX_OUTPUT_TOKENS
    assert ("max_tokens" in seen["request"]) != (token_parameter == "max_completion_tokens")


@pytest.mark.parametrize("typed", ["return something from order ss1", "order SS 1 please", "where is order #ss-1?"])
def test_an_order_id_typed_loosely_by_the_customer_still_counts_as_mentioned(db, policy_search, typed):
    client, _ = make_client(db, policy_search, calls(call("get_orders", order_id="SS-1")),
                            say("Here.\nROUTE: answer\nSOURCES: none"))
    assert tool_summary(send(client, typed)) == [("get_orders", "success")]


def test_was_mentioned_accepts_loose_forms_but_not_unrelated_numbers():
    from app.agent.orchestrator import was_mentioned

    text = "WHERE IS ORDER 10512? ALSO SS 10458"
    assert was_mentioned("SS-10512", text) and was_mentioned("ss10458", text)
    assert not was_mentioned("SS-10513", text) and not was_mentioned("SS-1051", text) and not was_mentioned("", text)


def test_an_unexpected_crash_is_a_controlled_200_reply_without_details(db, policy_search, monkeypatch):
    # Only 400/401/403 are allowed errors, so even a bug in saving must not become a 500 or leak details.
    client, _ = make_client(db, policy_search, *proposal_steps())
    send(client, "Return the speaker from SS-1, I changed my mind.")

    def broken(*args, **kwargs):
        raise RuntimeError("database is locked: /secret/path.db")

    monkeypatch.setitem(__import__("app.tools", fromlist=["x"]).TOOL_FUNCTIONS, "create_return_request", broken)
    reply = send(client, "Yes, go ahead.")
    assert reply["route"] == "no_answer" and "secret" not in reply["answer"] and ".db" not in reply["answer"]
    assert reply["pending_confirmation"] is False and count_rows(db, "return_requests") == 0
