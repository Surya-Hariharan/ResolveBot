"""Turns one customer message into one reply.

STAGE 1 PLACEHOLDER: the tool loop, routing and confirmation are not built yet
(Stages 4-5). For now this only checks that an LLM is usable, so the service
shows a controlled message instead of crashing when no key is configured.
"""


def handle_message(settings, customer_id, session, message):
    """Return (answer, route, citations, tool_calls, pending_confirmation)."""
    problem = settings.llm_problem()
    if problem:
        return (f"The assistant is not available right now. {problem}", "no_answer", [], [], False)

    return ("The assistant logic is not built yet (foundation stage).", "no_answer", [], [], False)
