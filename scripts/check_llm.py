"""Verify the live LLM connection (the provider chosen in .env) and a full tool-calling round trip.

    python -m scripts.check_llm

Prints PASS / FAIL / BLOCKED for each step. It never prints the key, and error
text is passed through a filter that removes it. Needs LLM_API_KEY in .env
or the environment.
"""
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from openai import OpenAI

from app import tools
from app.agent import llm
from app.config import load_settings
from app.main import create_app

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")  # model text can hold characters a Windows console cannot print


def safe(text, settings):
    """Remove the key from any text before it is printed."""
    return str(text).replace(settings.llm_api_key, "<key>") if settings.llm_api_key else str(text)


def main():
    settings = load_settings()
    problem = settings.llm_problem()
    if problem:
        print(f"BLOCKED: {problem}")
        return 2
    print(f"provider={settings.llm_provider} model={settings.llm_model}")

    # 1. Which models can this account use? Is the configured one among them?
    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url, timeout=20, max_retries=0)
    try:
        ids = sorted(m.id for m in client.models.list().data)
    except Exception as error:
        print("FAIL list models:", type(error).__name__, getattr(error, "status_code", ""), safe(error, settings)[:200])
        return 1
    print(f"PASS list models: {len(ids)} available; configured model available: {settings.llm_model in ids}")
    if settings.llm_model not in ids:
        print("FAIL configured model not available. Available:", ", ".join(ids))
        return 1

    # 2. A plain completion at temperature 0
    try:
        reply = llm.chat(settings, [{"role": "user", "content": "Reply with exactly the word: pong"}])
        print("PASS plain completion:", repr((reply.content or "").strip()[:40]))
    except Exception as error:
        print("FAIL plain completion:", type(error).__name__, safe(error, settings)[:200])
        return 1

    # 3. Does the model call our tool with sensible arguments?
    messages = [
        {"role": "system", "content": "You are a support assistant. Use tools to look up orders."},
        {"role": "user", "content": "Where is order SS-10512?"},
    ]
    try:
        first = llm.chat(settings, messages, tools.TOOL_SCHEMAS)
    except Exception as error:
        print("FAIL tool request:", type(error).__name__, safe(error, settings)[:300])
        return 1
    if not first.tool_calls:
        print("FAIL the model did not call a tool:", repr((first.content or "")[:100]))
        return 1
    call = first.tool_calls[0]
    print("PASS tool requested:", call.function.name, call.function.arguments)

    # 4. Return the tool result to the model: does it produce a final answer from it?
    with tempfile.TemporaryDirectory() as folder:
        copy = str(Path(folder) / "copy.db")
        shutil.copy(ROOT / "data" / "shopsphere.db", copy)
        ctx = tools.ToolContext(copy, settings.app_today, "C1009")
        result = tools.get_orders(ctx, **json.loads(call.function.arguments))
    messages.append({"role": "assistant", "content": None, "tool_calls": [
        {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}}]})
    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})
    try:
        final = llm.chat(settings, messages, tools.TOOL_SCHEMAS)
    except Exception as error:
        print("FAIL tool result round trip:", type(error).__name__, safe(error, settings)[:300])
        return 1
    # the model sometimes writes narrow no-break spaces; treat them as normal spaces
    mentions_eta = "3 July 2026" in (final.content or "").replace(" ", " ").replace(" ", " ")
    print(f"{'PASS' if mentions_eta else 'FAIL'} tool result used in the answer (mentions '3 July 2026'):", repr((final.content or "")[:160]))

    # 5. The same thing through the whole application
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        copy = str(Path(folder) / "copy.db")
        shutil.copy(ROOT / "data" / "shopsphere.db", copy)
        settings.db_path = copy
        app_client = TestClient(create_app(settings))
        body = app_client.post("/chat", json={"session_id": "check", "message": "Where is order SS-10512?"},
                               headers={"X-Customer-Id": "C1009"}).json()
        connection = sqlite3.connect(copy)
        rows = connection.execute("SELECT (SELECT COUNT(*) FROM return_requests) + (SELECT COUNT(*) FROM support_tickets)").fetchone()[0]
        connection.close()
    print("app reply:", json.dumps({k: body[k] for k in ("route", "tool_calls", "citations", "pending_confirmation")}), "| rows saved:", rows)
    print("answer:", body["answer"][:200])
    return 0 if mentions_eta and body["route"] == "answer" and rows == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
