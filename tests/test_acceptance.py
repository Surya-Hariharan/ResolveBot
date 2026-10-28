"""The five mandatory acceptance gates, run against a COPY of the real database.

The original data/shopsphere.db is only ever read (copied). If data/ is missing
these tests are skipped, because the data folder is not part of the repository.
"""
import hashlib
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import tools
from app.config import load_settings
from app.main import create_app
from app.retrieval.policy_loader import load_chunks
from app.retrieval.search import PolicySearch
from app.tools import ToolContext
from tests.scripted_llm import FakeLLM, call, calls, say

ROOT = Path(__file__).resolve().parent.parent
REAL_DB = ROOT / "data" / "shopsphere.db"
needs_data = pytest.mark.skipif(not REAL_DB.exists(), reason="data/ folder is not present")
TODAY = date(2026, 7, 1)
NOT_FOUND_TEXT = "I couldn't find that order on your account."
ROUTES = {"answer", "clarify", "no_answer", "out_of_scope"}


@pytest.fixture
def real_copy(tmp_path):
    copy = tmp_path / "copy.db"
    shutil.copy(REAL_DB, copy)
    return str(copy)


def snapshot(db_path):
    """A fingerprint of every table, so we can prove nothing changed."""
    connection = sqlite3.connect(db_path)
    try:
        digest = {}
        for (table,) in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            rows = connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            digest[table] = (len(rows), hashlib.sha256(repr(rows).encode()).hexdigest())
        return digest
    finally:
        connection.close()


def client_for(db_path, *steps):
    settings = load_settings({"DB_PATH": db_path, "LLM_API_KEY": "test-key-not-real"})
    chunks = load_chunks(ROOT / "data" / "policies.md", ROOT / "data" / "manifest.json", TODAY)
    search = PolicySearch(chunks=chunks, model_name="keyword-only", top_k=3, min_score=0.3)
    search._ready.set()
    return TestClient(create_app(settings, llm_chat=FakeLLM(*steps), policy_search=search))


def post(client, message, customer="C1009", session="gate"):
    return client.post(
        "/chat", json={"session_id": session, "message": message}, headers={"X-Customer-Id": customer}
    )


def delivered_orders(db_path):
    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        "SELECT o.order_id, o.customer_id, i.item_id, s.delivered_at, p.returnable, i.opened"
        " FROM orders o JOIN shipments s USING(order_id) JOIN order_items i USING(order_id)"
        " JOIN products p USING(product_id) WHERE o.status='delivered'"
    ).fetchall()
    connection.close()
    return rows


# ------------------------------------------------------------ Gate 1: startup
def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@needs_data
def test_gate1_service_starts_with_the_readme_command_and_answers():
    # The README command is `uvicorn app.main:app --port 8000`; another port avoids clashes here.
    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
        cwd=ROOT,
        env={"PATH": "", "SYSTEMROOT": "C:\\Windows", "APP_TODAY": "2026-07-01",
             "DB_PATH": str(REAL_DB)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        started = time.time()
        while time.time() - started < 40:
            try:
                response = httpx.post(
                    f"http://127.0.0.1:{port}/chat",
                    json={"session_id": "g1", "message": "hi"},
                    headers={"X-Customer-Id": "C1009"},
                    timeout=5,
                )
                break
            except httpx.TransportError:
                time.sleep(0.5)
        else:
            pytest.fail("service did not start within 40 seconds")
        assert response.status_code == 200
        body = response.json()
        assert body["route"] in ROUTES  # no key in this environment: a controlled reply, not a crash
        assert httpx.post(f"http://127.0.0.1:{port}/chat", json={}).status_code == 401
    finally:
        server.terminate()
        server.wait(timeout=10)


# -------------------------------------------------------- Gate 2: reply format
@needs_data
def test_gate2_reply_format_for_every_route(real_copy):
    client = client_for(
        real_copy,
        calls(call("get_orders", order_id="SS-10458")),
        say("It was delivered 21 June 2026.\nROUTE: answer\nSOURCES: none"),
        say("Which order?\nROUTE: clarify\nSOURCES: none"),
        calls(call("get_orders", order_id="SS-10000")),
        say("x\nROUTE: answer\nSOURCES: none"),
        say("I only help with ShopSphere.\nROUTE: out_of_scope\nSOURCES: none"),
    )
    seen = set()
    for index, text in enumerate(["track SS-10458", "return my charger", "track SS-10000", "poem please"]):
        body = post(client, text, session=f"fmt{index}").json()
        assert {"request_id", "answer", "route", "citations", "tool_calls", "pending_confirmation"} <= set(body)
        assert isinstance(body["request_id"], str) and body["request_id"]
        assert isinstance(body["answer"], str) and isinstance(body["pending_confirmation"], bool)
        for entry in body["tool_calls"]:
            assert set(entry) == {"name", "args", "status"}
            assert entry["status"] in {"success", "error", "pending_confirmation"}
        seen.add(body["route"])
        if body["route"] in ("no_answer", "out_of_scope"):
            assert body["citations"] == []
        if body["route"] == "out_of_scope":
            assert body["tool_calls"] == []
    assert seen == ROUTES


@needs_data
def test_gate2_http_errors_are_only_400_401_403(real_copy):
    client = client_for(real_copy, say("hi\nROUTE: out_of_scope\nSOURCES: none"))
    assert client.post("/chat", content="{bad", headers={"X-Customer-Id": "C1009"}).status_code == 400
    assert client.post("/chat", json={"message": "x"}, headers={"X-Customer-Id": "C1009"}).status_code == 400
    assert client.post("/chat", json={"session_id": "a", "message": "x"}).status_code == 401
    assert post(client, "x", customer="C0000").status_code == 401
    assert post(client, "hello", customer="C1009", session="mine").status_code == 200
    assert post(client, "hello", customer="C1001", session="mine").status_code == 403


# ------------------------------------------------- Gate 3: confirmation safety
@needs_data
def test_gate3_nothing_is_written_before_a_clear_yes(real_copy):
    args = {"order_id": "SS-10458", "item_ids": ["SS-10458-1"], "reason": "changed_mind"}
    client = client_for(
        real_copy,
        calls(call("get_orders", order_id="SS-10458")),
        calls(call("check_return_eligibility", **args)),
        calls(call("create_return_request", **args)),
        say("You can return the speaker. Shall I go ahead?\nROUTE: answer\nSOURCES: none"),
        say("Okay, no problem.\nROUTE: answer\nSOURCES: none"),
    )
    before = snapshot(real_copy)
    proposal = post(client, "Return the speaker from SS-10458, I changed my mind.").json()
    assert proposal["pending_confirmation"] is True
    assert proposal["tool_calls"][-1]["status"] == "pending_confirmation"
    assert snapshot(real_copy) == before, "a proposal must not change any table"

    post(client, "no")  # declining forgets the proposal
    assert snapshot(real_copy) == before
    post(client, "yes")  # nothing is waiting, so this is just a message
    assert snapshot(real_copy) == before


@needs_data
def test_gate3_exactly_one_row_after_yes_and_none_for_ambiguous_replies(real_copy):
    args = {"order_id": "SS-10458", "item_ids": ["SS-10458-1"], "reason": "changed_mind"}
    steps = [calls(call("create_return_request", **args)), say("Shall I go ahead?\nROUTE: answer\nSOURCES: none")]
    client = client_for(real_copy, *steps, say("ok\nROUTE: answer\nSOURCES: none"), *steps)
    before = snapshot(real_copy)
    post(client, "return SS-10458 speaker, changed my mind")
    post(client, "maybe")
    assert snapshot(real_copy) == before
    post(client, "return SS-10458 speaker, changed my mind")
    reply = post(client, "Yes, go ahead.").json()
    assert reply["tool_calls"][0]["status"] == "success" and reply["pending_confirmation"] is False
    after = snapshot(real_copy)
    assert after["return_requests"][0] == 1
    assert {t: v for t, v in after.items() if t != "return_requests"} == {
        t: v for t, v in before.items() if t != "return_requests"
    }


@needs_data
def test_gate3_read_only_tools_never_write(real_copy):
    before = snapshot(real_copy)
    ctx = ToolContext(real_copy, TODAY, "C1009")
    tools.get_orders(ctx)
    tools.get_orders(ctx, order_id="SS-10458")
    tools.check_return_eligibility(ctx, "SS-10458", ["SS-10458-1"], "changed_mind")
    tools.validate_write(ctx, "create_return_request",
                         {"order_id": "SS-10458", "item_ids": ["SS-10458-1"], "reason": "changed_mind"})
    assert snapshot(real_copy) == before


# ------------------------------------------- Gate 4: business-rule correctness
@needs_data
def test_gate4_every_delivered_item_matches_an_independent_calculation(real_copy):
    """Compare the tool with a separate hand-written calculation for ALL delivered items."""
    checked = 0
    for order_id, customer, item_id, delivered_at, returnable, opened in delivered_orders(real_copy):
        days = (TODAY - date.fromisoformat(delivered_at[:10])).days
        for reason in ("changed_mind", "defective"):
            if returnable == "no":
                expected = "NON_RETURNABLE"
            elif days > 30:
                expected = "RETURN_WINDOW_EXPIRED"
            elif reason == "changed_mind" and opened and returnable == "unopened_only":
                expected = "OPENED_NOT_RETURNABLE"
            else:
                expected = "eligible"
            result = tools.check_return_eligibility(ToolContext(real_copy, TODAY, customer), order_id, [item_id], reason)
            actual = "eligible" if result.get("eligible") else result["error_code"]
            assert actual == expected, (order_id, item_id, reason, days)
            checked += 1
    assert checked > 300


@needs_data
def test_gate4_thresholds_come_from_policy_parameters(real_copy):
    order_id, customer, item_id, delivered_at, *_ = next(
        row for row in delivered_orders(real_copy) if row[4] == "yes" and row[5] == 0
    )
    days = (TODAY - date.fromisoformat(delivered_at[:10])).days
    ctx = ToolContext(real_copy, TODAY, customer)

    def set_window(value):
        connection = sqlite3.connect(real_copy)
        connection.execute("UPDATE policy_parameters SET value = ? WHERE key = 'return_window_days'", (str(value),))
        connection.commit()
        connection.close()

    set_window(days)  # exactly on the last allowed day
    assert tools.check_return_eligibility(ctx, order_id, [item_id], "changed_mind").get("eligible") is True
    set_window(days - 1)  # one day too late
    assert tools.check_return_eligibility(ctx, order_id, [item_id], "changed_mind")["error_code"] == "RETURN_WINDOW_EXPIRED"


@needs_data
def test_gate4_warranty_months_come_from_policy_parameters(real_copy):
    ctx = ToolContext(real_copy, TODAY, "C1009")
    base = tools.get_orders(ctx, order_id="SS-10458")["orders"][0]["items"][0]
    assert base["warranty_end_date"] == "21 June 2027"  # delivered 21 June 2026, 12 months
    connection = sqlite3.connect(real_copy)
    connection.execute("UPDATE policy_parameters SET value = '1' WHERE key = 'standard_warranty_months'")
    connection.commit()
    connection.close()
    changed = tools.get_orders(ctx, order_id="SS-10458")["orders"][0]["items"][0]
    assert changed["warranty_end_date"] == "21 July 2026"


# -------------------------------------------------- Gate 5: customer isolation
@needs_data
def test_gate5_foreign_and_missing_orders_are_indistinguishable(real_copy):
    connection = sqlite3.connect(real_copy)
    orders = connection.execute("SELECT order_id, customer_id FROM orders").fetchall()
    customers = [row[0] for row in connection.execute("SELECT customer_id FROM customers")]
    connection.close()
    missing = tools.get_orders(ToolContext(real_copy, TODAY, "C1009"), order_id="SS-00000")
    assert missing["message"] == NOT_FOUND_TEXT
    for order_id, owner in orders:
        stranger = next(c for c in customers if c != owner)
        ctx = ToolContext(real_copy, TODAY, stranger)
        assert tools.get_orders(ctx, order_id=order_id) == missing
        assert tools.check_return_eligibility(ctx, order_id, [f"{order_id}-1"], "changed_mind") == missing
        assert tools.validate_write(ctx, "create_support_ticket",
                                    {"order_id": order_id, "ticket_type": "cancellation", "summary": "x"}) == missing


@needs_data
def test_gate5_listing_orders_never_includes_other_customers(real_copy):
    connection = sqlite3.connect(real_copy)
    owners = dict(connection.execute("SELECT order_id, customer_id FROM orders").fetchall())
    customers = [row[0] for row in connection.execute("SELECT customer_id FROM customers")]
    connection.close()
    for customer in customers:
        listed = tools.get_orders(ToolContext(real_copy, TODAY, customer))["orders"]
        assert all(owners[o["order_id"]] == customer for o in listed)


@needs_data
def test_gate5_chat_reply_for_a_foreign_order_is_the_standard_refusal(real_copy):
    connection = sqlite3.connect(real_copy)
    foreign = connection.execute("SELECT order_id FROM orders WHERE customer_id = 'C1001' LIMIT 1").fetchone()[0]
    connection.close()
    client = client_for(
        real_copy,
        calls(call("get_orders", order_id=foreign)),
        say("That belongs to Aarav Sharma, here are the details...\nROUTE: answer\nSOURCES: none"),
    )
    body = post(client, f"Where is order {foreign}?", customer="C1009").json()
    assert body["answer"] == NOT_FOUND_TEXT and body["route"] == "no_answer"
    assert "Aarav" not in str(body)


@needs_data
def test_gate5_sessions_are_private(real_copy):
    client = client_for(real_copy, say("hello\nROUTE: out_of_scope\nSOURCES: none"))
    assert post(client, "hi", customer="C1009", session="private").status_code == 200
    assert post(client, "hi", customer="C1001", session="private").status_code == 403
