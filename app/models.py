"""The shape of a /chat reply, exactly as the assignment defines it."""
from typing import Literal

from pydantic import BaseModel


class Citation(BaseModel):
    doc_id: str
    section: str


class ToolCall(BaseModel):
    name: str
    args: dict
    status: Literal["success", "error", "pending_confirmation"]


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    route: Literal["answer", "clarify", "no_answer", "out_of_scope"]
    citations: list[Citation] = []
    tool_calls: list[ToolCall] = []
    pending_confirmation: bool = False
