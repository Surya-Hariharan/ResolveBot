"""Turns one customer message into one reply.

Order of events for every message:
  1. If a proposal is waiting and the message is a clear yes, Python saves it (no LLM).
  2. Otherwise the waiting proposal is forgotten and the LLM/tool loop runs.
  3. Write tools never save inside the loop; they only record a proposal.
  4. The route, citations and tool_calls of the reply are decided in Python.
"""
import json
import logging
import re
import time
from dataclasses import dataclass, field

from app import tools
from app.agent import llm
from app.agent.prompts import build_system_prompt
from app.confirmation import is_clear_yes
from app.tools import ORDER_NOT_FOUND_TEXT, ToolContext

logger = logging.getLogger("resolvebot")

MAX_TOOL_CALLS = 6
REPLY_DEADLINE_SECONDS = 27  # the assignment allows 30; keep a margin for the HTTP layer
MAX_HISTORY_MESSAGES = 12
ROUTES = ("answer", "clarify", "no_answer", "out_of_scope")


@dataclass
class Reply:
    answer: str
    route: str
    citations: list = field(default_factory=list)  # [{"doc_id", "section"}]
    tool_calls: list = field(default_factory=list)  # [{"name", "args", "status"}]
    pending_confirmation: bool = False


@dataclass
class TurnState:
    """Everything that happened while answering this one message."""

    tool_calls: list = field(default_factory=list)
    results: list = field(default_factory=list)  # (tool name, result dict)
    retrieved: dict = field(default_factory=dict)  # (doc_id, section) -> paragraph
    searched: bool = False
    proposal: dict | None = None  # {"name": ..., "args": ...}
    calls_made: int = 0


# ---------------------------------------------------------------- entry point
def handle_message(settings, customer_id, session, message, policy_search=None, llm_chat=llm.chat):
    ctx = ToolContext(
        db_path=settings.db_path,
        today=settings.app_today,
        customer_id=customer_id,
        policy_search=(lambda query: policy_search.search(query, extra_query=message)) if policy_search else None,
    )

    # The waiting proposal is used up by whatever the customer says next
    waiting = session.pending_proposal
    session.pending_proposal = None

    if waiting and is_clear_yes(message):
        reply = save_confirmed_action(ctx, waiting)
        remember(session, message, reply.answer)
        return reply

    problem = settings.llm_problem()
    if problem:
        return Reply(f"The assistant is not available right now. {problem}", "no_answer")

    support_email = policy_search.support_email if policy_search else None
    started = time.monotonic()
    state = TurnState()
    try:
        text = run_tool_loop(settings, ctx, session, message, state, llm_chat, support_email, started)
    except llm.LLMNotAvailable as problem:
        return Reply(f"The assistant is not available right now. {problem}", "no_answer")
    except TurnTimeout:
        return Reply("Sorry, that is taking too long. Please try again in a moment.", "no_answer", tool_calls=unfinished(state.tool_calls))
    except Exception as problem:  # provider errors (rate limit, network, bad request...)
        logger.warning("LLM call failed: %s %s %s", type(problem).__name__, getattr(problem, "status_code", ""), getattr(problem, "code", ""))
        return Reply(provider_error_text(problem), "no_answer", tool_calls=unfinished(state.tool_calls))

    reply = build_reply(text, state, support_email)
    if reply.pending_confirmation:
        session.pending_proposal = state.proposal
    remember(session, message, reply.answer)
    return reply


def unfinished(tool_calls):
    """The reply failed, so no proposal is kept: a call that was waiting for a yes is reported as an error."""
    return [{**c, "status": "error"} if c["status"] == "pending_confirmation" else c for c in tool_calls]


class TurnTimeout(Exception):
    pass


def remember(session, user_message, assistant_answer):
    session.history.append({"role": "user", "content": user_message})
    session.history.append({"role": "assistant", "content": assistant_answer})
    del session.history[:-MAX_HISTORY_MESSAGES]


def provider_error_text(problem):
    name = type(problem).__name__
    if "Timeout" in name:
        return "Sorry, the assistant took too long to respond. Please try again."
    if "RateLimit" in name:
        return "The assistant is busy right now. Please try again in a moment."
    if "Authentication" in name or "PermissionDenied" in name:
        return "The assistant could not sign in to its language model. Please contact the service owner."
    return "The assistant ran into a problem reaching its language model. Please try again."


# ------------------------------------------------------------- the yes step
def save_confirmed_action(ctx, waiting):
    """Run the proposed write now. The tool re-checks the rules before saving."""
    name, args = waiting["name"], waiting["args"]
    result = tools.TOOL_FUNCTIONS[name](ctx, **args)
    if "error_code" in result:
        answer = f"I couldn't do that after all: {result['message']} Nothing was saved."
        status = "error"
    elif name == "create_return_request":
        answer = f"Done. Your return request has been created (reference {result['request_id']})."
        status = "success"
    else:
        answer = f"Done. Your {result['ticket_type']} ticket has been opened (reference {result['ticket_id']})."
        status = "success"
    return Reply(answer, "answer", tool_calls=[{"name": name, "args": args, "status": status}])


# ---------------------------------------------------------------- tool loop
def run_tool_loop(settings, ctx, session, message, state, llm_chat, support_email, started):
    messages = [{"role": "system", "content": build_system_prompt(ctx.today, support_email)}]
    messages += session.history[-MAX_HISTORY_MESSAGES:]
    messages.append({"role": "user", "content": message})

    offer_tools = True
    for _round in range(MAX_TOOL_CALLS + 3):  # hard stop even if the model keeps asking for tools
        remaining = REPLY_DEADLINE_SECONDS - (time.monotonic() - started)
        if remaining < 2:
            raise TurnTimeout()
        reply = llm_chat(settings, messages, tools.TOOL_SCHEMAS if offer_tools else None, timeout=remaining)
        if not reply.tool_calls:
            return reply.content or ""

        messages.append(assistant_message_as_dict(reply))
        for call in reply.tool_calls:
            if state.calls_made >= MAX_TOOL_CALLS:
                result = tools.error("TOOL_LIMIT_REACHED", "No more tool calls are allowed for this message.")
                offer_tools = False
            else:
                state.calls_made += 1
                result = run_tool(ctx, state, call.function.name, call.function.arguments, conversation_text(messages))
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False, default=str)}
            )
        if state.calls_made >= MAX_TOOL_CALLS:
            offer_tools = False
            messages.append(
                {"role": "user", "content": "(System note: no more tool calls are allowed. Reply now with what you have, "
                 "or ask the customer to continue in a new message.)"}
            )
    return "I wasn't able to finish that in one go. Could you ask for one thing at a time?"


def assistant_message_as_dict(reply):
    return {
        "role": "assistant",
        "content": reply.content or None,
        "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in reply.tool_calls
        ],
    }


def conversation_text(messages):
    """Everything said or returned so far, upper-cased: an order id must appear in it to be used."""
    return " ".join(str(m.get("content") or "") for m in messages).upper()


def was_mentioned(order_id, conversation):
    """Is this order id in the conversation? 'ss10512', 'SS 10512' and just '10512' all count."""
    squeezed = re.sub(r"[^A-Z0-9]", "", order_id.upper())
    if not squeezed:
        return False
    if squeezed in re.sub(r"[^A-Z0-9]", "", conversation):
        return True
    number = re.search(r"\d{3,}", squeezed)
    return bool(number and re.search(rf"(?<!\d){number.group()}(?!\d)", conversation))


def run_tool(ctx, state, name, raw_arguments, conversation=""):
    """Run one tool the LLM asked for and record it. Always returns a dict, never raises."""
    if name not in tools.TOOL_FUNCTIONS:
        return tools.error("UNKNOWN_TOOL", f"There is no tool called {name}.")
    try:
        asked = json.loads(raw_arguments) if raw_arguments else {}
        if not isinstance(asked, dict):
            raise ValueError
    except ValueError:
        return record(state, name, {}, tools.error("INVALID_INPUT", "Tool arguments must be a JSON object."))

    # Keep only the arguments the tool declares. A customer_id sent by the LLM is dropped here.
    args = {key: asked[key] for key in tools.TOOL_ARGUMENTS[name] if asked.get(key) is not None}

    # The model must not guess an order id the customer never gave (and no tool result ever showed)
    guessed = args.get("order_id")
    if isinstance(guessed, str) and not was_mentioned(guessed, conversation):
        return record(state, name, args, tools.error(
            "INVALID_INPUT", "The customer has not said which order. Ask them which order they mean."))

    if name in tools.WRITE_TOOLS:
        result = propose_write(ctx, state, name, args)
    else:
        try:
            result = tools.TOOL_FUNCTIONS[name](ctx, **args)
        except Exception as problem:  # a bug in a tool must not crash the chat
            logger.error("Tool %s failed: %s", name, type(problem).__name__)
            result = tools.error("TOOL_FAILED", "That lookup failed. Please try again.")
    if name == "search_policy" and "paragraphs" in result:
        state.searched = True
        for paragraph in result["paragraphs"]:
            state.retrieved.setdefault((paragraph["doc_id"], paragraph["section"]), paragraph)
    return record(state, name, args, result)


def propose_write(ctx, state, name, args):
    """A write tool called by the LLM only proposes. Nothing is saved here."""
    if state.proposal is not None:
        return tools.error(
            "ONE_PROPOSAL_AT_A_TIME",
            "Only one action can wait for confirmation. Finish this one first, then offer the other.",
        )
    try:
        problem = tools.validate_write(ctx, name, args)
    except Exception as bad_input:
        logger.error("Proposal check failed: %s", type(bad_input).__name__)
        return tools.error("TOOL_FAILED", "That check failed. Please try again.")
    if problem:
        return problem
    state.proposal = {"name": name, "args": args}
    return {
        "status": "pending_confirmation",
        "message": "NOT saved yet. Tell the customer exactly what you will do and ask 'Shall I go ahead?'. "
        "It is saved only if their next message is a clear yes.",
    }


def record(state, name, args, result):
    if result.get("status") == "pending_confirmation":
        status = "pending_confirmation"
    else:
        status = "error" if "error_code" in result else "success"
    state.tool_calls.append({"name": name, "args": args, "status": status})
    state.results.append((name, result))
    return result


# ------------------------------------------------------ building the reply
TAG_ROUTE = re.compile(r"^\W*ROUTE\W*:\W*([a-z_]+)\W*$", re.IGNORECASE | re.MULTILINE)
TAG_SOURCES = re.compile(r"^\W*SOURCES\W*:(.*)$", re.IGNORECASE | re.MULTILINE)
SOURCE_ITEM = re.compile(r"([A-Za-z0-9_]+)\s*(§[\d.]+)?")


# The model sometimes writes non-breaking hyphens and spaces (like in "SS-10512"), which break plain-text matching.
# Each is replaced by the ordinary character: hyphen variants U+2010 to U+2012, and the spaces U+00A0, U+2009, U+202F.
LOOKALIKES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u00a0": " ", "\u2009": " ", "\u202f": " "})


def build_reply(text, state, support_email):
    answer, route, cited = split_tags(text.translate(LOOKALIKES))
    if route is None:  # the model forgot the tag: no tools + a question back = clarify, no tools otherwise = off-topic
        if state.tool_calls:
            route = "answer"
        else:
            route = "clarify" if "?" in answer else "out_of_scope"
    failures = [r for _, r in state.results if r.get("error_code") == "ORDER_NOT_FOUND"]
    worked = [r for name, r in state.results if "error_code" not in r and name != "search_policy"]
    searches = [r for name, r in state.results if name == "search_policy" and "paragraphs" in r]

    if state.proposal:
        route = "answer"
        if not answer.rstrip().endswith("?"):  # the proposal must end by asking for a yes
            answer = f"{answer} Shall I go ahead?".strip()
        return Reply(answer, route, citations_for(state, cited), state.tool_calls, True)

    if failures:
        # Same words whether the order is missing or belongs to someone else
        if not worked:
            return Reply(ORDER_NOT_FOUND_TEXT, "no_answer", [], state.tool_calls)
        if ORDER_NOT_FOUND_TEXT not in answer:
            answer = f"{ORDER_NOT_FOUND_TEXT} {answer}".strip()

    if searches and not worked and not any(s["paragraphs"] for s in searches):
        return Reply(no_policy_text(support_email), "no_answer", [], state.tool_calls)

    if not answer.strip():
        answer = "Sorry, I couldn't put together an answer. Could you rephrase that?"
        route = "clarify"

    if route == "out_of_scope" and state.tool_calls:
        route = "answer"  # it used tools, so it was about the store
    if route == "no_answer":
        if support_email and support_email not in answer:
            answer = f"{answer} You can email {support_email}."
        return Reply(answer, "no_answer", [], state.tool_calls)
    if route == "out_of_scope":
        return Reply(answer, "out_of_scope", [], [])
    return Reply(answer, route, citations_for(state, cited), state.tool_calls)


def split_tags(text):
    """Remove the ROUTE and SOURCES lines the LLM adds. Returns (answer, route, cited); route is None if missing."""
    route_match = TAG_ROUTE.search(text)
    route = route_match.group(1).lower() if route_match and route_match.group(1).lower() in ROUTES else None
    sources_match = TAG_SOURCES.search(text)
    cited = None  # None = the LLM did not say; [] = it said none
    if sources_match:
        body = sources_match.group(1).strip()
        cited = [] if body.lower().startswith("none") or not body else SOURCE_ITEM.findall(body)
    cleaned = TAG_SOURCES.sub("", TAG_ROUTE.sub("", text)).strip()
    return cleaned, route, cited


def citations_for(state, cited):
    """Only paragraphs that search_policy really returned can be cited."""
    if not state.retrieved:
        return []
    if cited is None:  # the LLM gave no list: cite the best match rather than nothing
        first = next(iter(state.retrieved))
        return [{"doc_id": first[0], "section": first[1]}]
    result = []
    for doc_id, section in cited:
        if not section:  # only the document was named: use its first retrieved section
            section = next((s for d, s in state.retrieved if d == doc_id), "")
        if (doc_id, section) in state.retrieved and {"doc_id": doc_id, "section": section} not in result:
            result.append({"doc_id": doc_id, "section": section})
    return result


def no_policy_text(support_email):
    contact = f"emailing {support_email}" if support_email else "contacting our support team"
    return f"Sorry, our current policies don't cover that. I'd suggest {contact}."
