"""FastAPI app: POST /chat with the HTTP errors the assignment asks for.

Order of checks: who is calling (401), is the body valid (400), does the
session belong to them (403). Everything else answers 200.
"""
import json
import uuid

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app import database
from app.agent.orchestrator import handle_message
from app.config import load_settings
from app.models import ChatResponse
from app.sessions import SessionOwnerError, SessionStore


def error_response(status_code, message):
    return JSONResponse(status_code=status_code, content={"detail": message})


def read_chat_body(raw_body):
    """Return (session_id, message), or None if the body is not valid."""
    try:
        body = json.loads(raw_body)
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    session_id = body.get("session_id")
    message = body.get("message")
    if not isinstance(session_id, str) or not session_id.strip():
        return None
    if not isinstance(message, str) or not message.strip():
        return None
    return session_id, message


def create_app(settings=None):
    settings = settings or load_settings()
    sessions = SessionStore()
    app = FastAPI(title="ResolveBot")

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: Request):
        # The customer comes from the header only, never from the message text
        customer_id = (request.headers.get("X-Customer-Id") or "").strip()
        if not customer_id or not database.customer_exists(settings.db_path, customer_id):
            return error_response(401, "Missing or unknown X-Customer-Id.")

        parsed = read_chat_body(await request.body())
        if parsed is None:
            return error_response(400, "Body must be JSON with non-empty session_id and message.")
        session_id, message = parsed

        try:
            session = sessions.get_or_create(session_id, customer_id)
        except SessionOwnerError:
            return error_response(403, "This session belongs to another customer.")

        answer, route, citations, tool_calls, pending = await run_in_threadpool(
            handle_message, settings, customer_id, session, message
        )
        return ChatResponse(
            request_id=str(uuid.uuid4()),
            answer=answer,
            route=route,
            citations=citations,
            tool_calls=tool_calls,
            pending_confirmation=pending,
        )

    return app


app = create_app()
