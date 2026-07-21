"""Wire types mirroring the PharosOne dialog ingest API contract.

The keys are exactly the JSON wire keys (snake_case): `args_preview`,
`result_preview`, `message_id`, `tool_call`, and so on. `TypedDict` is used so
plain dict literals type-check without any runtime dependency.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, TypedDict, Union

Role = Literal["user", "bot", "tool"]
ToolStatus = Literal["ok", "denied", "error", "pending"]


class _ToolCallRequired(TypedDict):
    name: str
    label: str
    status: ToolStatus


class ToolCall(_ToolCallRequired, total=False):
    """A tool invocation attached to a message (usually role="tool").

    `args_preview` and `result_preview` are optional short previews of the
    tool arguments and result.
    """

    args_preview: str
    result_preview: str


class _MessageRequired(TypedDict):
    role: Role
    text: str


class Message(_MessageRequired, total=False):
    """One dialog turn for send_dialog snapshots.

    `ts` accepts a `datetime` (serialized as RFC 3339 UTC; naive values are
    taken as UTC) or an already-formatted RFC 3339 string. `message_id` is the
    client-side idempotency key for per-message upserts.
    """

    ts: Union[datetime, str]
    message_id: str
    tool_call: ToolCall


class EndUser(TypedDict, total=False):
    """Optional end-user context (`end_user` on send_message / send_dialog)."""

    external_id: str
    email: str
    name: str
    ip: str
    user_agent: str
    locale: str
    timezone: str
    referrer: str
