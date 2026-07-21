"""HTTP client for the PharosOne dialog ingest API — stdlib only."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional, Sequence, Union

from ._version import __version__
from .types import EndUser, Message, ToolCall

__all__ = ["PharosOne", "PharosOneError"]

_DEFAULT_TIMEOUT = 15.0
# get_analysis blocks while the server computes the deep analysis (up to ~75s
# worst case), so it needs a longer per-call timeout than regular ingest calls.
_DEFAULT_ANALYSIS_TIMEOUT = 90.0
_USER_AGENT = f"pharosone-dialogs-python/{__version__}"


class PharosOneError(Exception):
    """Raised for non-2xx API responses.

    `status` is the HTTP status code; `detail` is the huma error `detail`
    field when the body is JSON with one, otherwise the raw body text.
    """

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def _rfc3339_utc(ts: Union[datetime, str]) -> str:
    """Serialize a timestamp as RFC 3339 UTC (naive datetimes are taken as UTC)."""
    if isinstance(ts, str):
        return ts
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _encode_message(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = dict(message)
    ts = out.get("ts")
    if isinstance(ts, datetime):
        out["ts"] = _rfc3339_utc(ts)
    return out


def _extract_detail(raw: bytes) -> str:
    text = raw.decode("utf-8", "replace")
    try:
        parsed = json.loads(text)
    except ValueError:
        return text
    if isinstance(parsed, dict):
        detail = parsed.get("detail")
        if isinstance(detail, str) and detail:
            return detail
    return text


class PharosOne:
    """Client for the PharosOne dialog ingest API.

    Args:
        base_url: API origin, e.g. "https://pharosone.example.com". Falls back
            to the PHAROSONE_BASE_URL env var; explicit argument wins.
        api_key: Ingest API key sent as "Authorization: Bearer <key>". Falls
            back to the PHAROSONE_API_KEY env var; explicit argument wins.
        timeout: Per-request timeout in seconds (default 15.0).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if base_url is None:
            base_url = os.environ.get("PHAROSONE_BASE_URL")
        if api_key is None:
            api_key = os.environ.get("PHAROSONE_API_KEY")
        if not base_url:
            raise ValueError(
                "PharosOne base_url is required: pass base_url= or set PHAROSONE_BASE_URL"
            )
        if not api_key:
            raise ValueError(
                "PharosOne api_key is required: pass api_key= or set PHAROSONE_API_KEY"
            )
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout = timeout

    def upsert_agent(
        self,
        agent_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        goal: Optional[str] = None,
    ) -> dict[str, Any]:
        """POST /api/v1/upsert-agent — create or update an agent.

        `agent_id` is the internal id or the agent name; missing agents are
        created (name defaults to `agent_id`). Only the provided fields are
        updated. Returns the agent as a dict (id, name, description, goal, ...).
        """
        body: dict[str, Any] = {"agent_id": agent_id}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if goal is not None:
            body["goal"] = goal
        return self._post("/api/v1/upsert-agent", body)

    def send_message(
        self,
        agent_id: str,
        session_id: str,
        role: str,
        text: str,
        *,
        ts: Optional[Union[datetime, str]] = None,
        message_id: Optional[str] = None,
        tool_call: Optional[ToolCall] = None,
        end_user: Optional[EndUser] = None,
    ) -> dict[str, Any]:
        """POST /api/v1/send-message — stream a single dialog message.

        Appends one message to the dialog identified by (agent_id, session_id).
        Pass `message_id` to make the call idempotent: re-sending the same id
        updates the existing row instead of appending. Returns a dict with
        status, dialog_id, message_index, created, and the synchronous fast
        verdict: `flagged` (bool) and `fast_scan` ("ok" | "failed").
        `fast_scan == "failed"` means the fast scan did not run — there is NO
        verdict; never treat `flagged == False` as clean in that case. For the
        detailed finding, call :meth:`get_analysis`.
        """
        body: dict[str, Any] = {
            "agent_id": agent_id,
            "session_id": session_id,
            "role": role,
            "text": text,
        }
        if ts is not None:
            body["ts"] = _rfc3339_utc(ts)
        if message_id is not None:
            body["message_id"] = message_id
        if tool_call is not None:
            body["tool_call"] = dict(tool_call)
        if end_user is not None:
            body["end_user"] = dict(end_user)
        return self._post("/api/v1/send-message", body)

    def send_dialog(
        self,
        agent_id: str,
        session_id: str,
        messages: Sequence[Message],
        *,
        end_user: Optional[EndUser] = None,
    ) -> dict[str, Any]:
        """POST /api/v1/send-dialog — replace a dialog with a full snapshot.

        Re-sending the same (agent_id, session_id) replaces all stored
        messages with this snapshot. Returns a dict with status, dialog_id,
        and the synchronous fast verdict: `flagged` (bool) and `fast_scan`
        ("ok" | "failed"). `fast_scan == "failed"` means the fast scan did not
        run — there is NO verdict; never treat `flagged == False` as clean in
        that case. For the detailed finding, call :meth:`get_analysis`.
        """
        body: dict[str, Any] = {
            "agent_id": agent_id,
            "session_id": session_id,
            "messages": [_encode_message(m) for m in messages],
        }
        if end_user is not None:
            body["end_user"] = dict(end_user)
        return self._post("/api/v1/send-dialog", body)

    def get_analysis(
        self,
        *,
        dialog_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """POST /api/v1/dialog-analysis — fetch the detailed verdict on demand.

        Select the dialog with `dialog_id`, or with `agent_id` and
        `session_id` together — exactly one form; anything else raises
        `ValueError` before any request is made.

        The call is synchronous: the server computes the deep analysis while
        the request blocks (up to ~75s worst case). This method therefore uses
        a per-call timeout of `max(self.timeout, 90.0)` seconds instead of the
        constructor timeout; pass `timeout=` to override it for one call.

        Returns a dict with dialog_id, status ("live" | "clean" | "flagged"),
        analysis_status ("pending" | "running" | "done" | "failed"), flagged
        (bool), flag (dict or None) and effectiveness (dict or None). While
        analysis_status is not "done", flag and effectiveness may still be
        None — re-call later to retry (a "failed" analysis is retried by the
        server on the next call).
        """
        by_dialog = dialog_id is not None
        if by_dialog and (agent_id is not None or session_id is not None):
            raise ValueError(
                "get_analysis: pass either dialog_id or agent_id + session_id, not both"
            )
        if not by_dialog and (agent_id is None or session_id is None):
            raise ValueError(
                "get_analysis: pass dialog_id, or both agent_id and session_id"
            )
        if by_dialog:
            body: dict[str, Any] = {"dialog_id": dialog_id}
        else:
            body = {"agent_id": agent_id, "session_id": session_id}
        if timeout is None:
            timeout = max(self.timeout, _DEFAULT_ANALYSIS_TIMEOUT)
        return self._post("/api/v1/dialog-analysis", body, timeout=timeout)

    def _post(
        self, path: str, body: dict[str, Any], *, timeout: Optional[float] = None
    ) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": _USER_AGENT,
            },
        )
        if timeout is None:
            timeout = self.timeout
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.status
                raw = response.read()
        except urllib.error.HTTPError as err:
            raise PharosOneError(err.code, _extract_detail(err.read())) from err
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise PharosOneError(status, f"invalid JSON in response body: {raw[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise PharosOneError(status, f"unexpected non-object response: {raw[:200]!r}")
        return parsed
