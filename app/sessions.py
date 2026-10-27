"""In-memory sessions. A session belongs to the customer who started it."""
from dataclasses import dataclass, field


class SessionOwnerError(Exception):
    """Someone tried to use a session that another customer started."""


@dataclass
class Session:
    owner_id: str
    history: list = field(default_factory=list)  # chat messages, filled in by the agent
    pending_proposal: dict | None = None  # the one action waiting for a "yes"


class SessionStore:
    def __init__(self):
        self._sessions = {}

    def get_or_create(self, session_id, customer_id):
        session = self._sessions.get(session_id)
        if session is None:
            session = Session(owner_id=customer_id)
            self._sessions[session_id] = session
        elif session.owner_id != customer_id:
            raise SessionOwnerError(session_id)
        return session
