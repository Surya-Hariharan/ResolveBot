"""FastAPI app: POST /chat with the HTTP errors the assignment asks for.

Order of checks: who is calling (401), is the body valid (400), does the
session belong to them (403). Everything else answers 200.
"""
import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app import database
from app.agent import llm
from app.agent.orchestrator import Reply, handle_message
from app.config import load_settings
from app.models import ChatResponse
from app.retrieval.search import PolicySearch
from app.sessions import SessionOwnerError, SessionStore

logger = logging.getLogger("resolvebot")


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


def load_policy_search(settings):
    """Parse the policy files. The service still starts if they are missing;
    search_policy then reports that the documents are unavailable.
    """
    try:
        return PolicySearch(settings)
    except (OSError, ValueError, KeyError) as problem:
        logger.warning("Policy documents not loaded: %s", type(problem).__name__)
        return None


def create_app(settings=None, llm_chat=None, policy_search=None):
    """llm_chat and policy_search can be replaced in tests."""
    settings = settings or load_settings()
    sessions = SessionStore()
    if policy_search is None:
        policy_search = load_policy_search(settings)
    chat_function = llm_chat or llm.chat

    @asynccontextmanager
    async def lifespan(_app):
        # Load the embedding model in the background so the server is up immediately
        if policy_search is not None:
            policy_search.start_warm_up_in_background()
        yield

    app = FastAPI(title="ResolveBot", lifespan=lifespan)

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

        try:
            reply = await run_in_threadpool(
                handle_message, settings, customer_id, session, message, policy_search, chat_function
            )
        except Exception as problem:  # last resort: the assignment allows only 400/401/403 as errors
            logger.error("Unexpected error while answering: %s", type(problem).__name__)
            reply = Reply("Sorry, something went wrong on our side. Nothing was changed. Please try again.", "no_answer")
        return ChatResponse(
            request_id=str(uuid.uuid4()),
            answer=reply.answer,
            route=reply.route,
            citations=reply.citations,
            tool_calls=reply.tool_calls,
            pending_confirmation=reply.pending_confirmation,
        )

    return app


app = create_app()
