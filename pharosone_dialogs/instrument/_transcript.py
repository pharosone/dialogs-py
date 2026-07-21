"""Shared transcript rebuilding: provider payloads -> PharosOne dialog messages.

Single source of truth for the OpenAI- and Anthropic-shaped mappings used by
``pharosone_dialogs.instrument`` (wrap_openai / wrap_anthropic) and by
``pharosone_dialogs.integrations`` (LangChain, LiteLLM).

Everything is duck-typed: payload values may be plain dicts (wire payloads,
LiteLLM) or attribute objects (openai / anthropic SDK models) — never import
the provider packages here.

Mapping (pinned by docs/superpowers/specs/2026-07-21-provider-sdk-instrumentation.md):

OpenAI Chat Completions (request ``messages[]`` + ``choices[0].message``):
- ``system`` / ``developer``      -> skipped (feeds the agent description).
- ``user``                        -> ``{role:"user", text}``.
- ``assistant`` text              -> ``{role:"bot", text}``.
- ``assistant.tool_calls[]``      -> one ``role:"tool"`` entry each, keyed by
  the tool-call id (``message_id``), ``status:"pending"``,
  ``args_preview`` = the function arguments.
- ``tool`` result                 -> resolves the pending entry in place
  (``status`` ok/error best-effort, ``result_preview``); no separate message.

Anthropic Messages (``system`` param + ``messages[]`` + response ``content``):
- ``system`` param                -> skipped.
- user text blocks                -> ``user``; ``tool_result`` blocks resolve
  the pending entry by ``tool_use_id`` (``is_error`` -> status).
- assistant text blocks           -> ``bot``; ``tool_use`` blocks -> pending
  ``role:"tool"`` entries (``message_id`` = block id,
  ``args_preview`` = json(input)).

Content that is a list of parts (vision, blocks) -> text parts joined with
newlines, ``[non-text content]`` noted for the rest.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger("pharosone_dialogs.instrument")

# args_preview / result_preview cap (~500 chars per the spec).
PREVIEW_CAP = 500
# Message text cap: the server rejects >= 20000 chars.
TEXT_CAP = 19_999

_SKIPPED_ROLES = frozenset({"system", "developer"})

Redactor = Callable[[str], str]


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Read `name` from a dict or an attribute object (duck-typed access)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[: cap - 1] + "…"


def to_text(content: Any) -> str:
    """Normalize provider message content (str | list of parts | None) to text.

    List parts: text parts are joined with newlines; every non-text part is
    noted as ``[non-text content]``.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            text = get_field(part, "text")
            if isinstance(text, str):
                parts.append(text)
            else:
                parts.append("[non-text content]")
        return "\n".join(p for p in parts if p)
    return str(content)


def make_preview(value: Any, redact: Optional[Redactor]) -> str:
    """Build an args/result preview: stringify, redact (optional), cap ~500."""
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    if redact is not None:
        try:
            text = redact(text)
        except Exception:
            # Never leak the unredacted preview if the caller's hook breaks.
            logger.exception("pharosone: redact hook raised; preview replaced")
            text = "[redaction failed]"
    return truncate(text, PREVIEW_CAP)


def looks_like_error(text: str) -> bool:
    """Best-effort: does a tool result read as an error? Defaults to no."""
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if lowered.startswith(("error", "exception", "traceback")):
        return True
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except ValueError:
            return False
        if isinstance(parsed, dict) and (parsed.get("error") or parsed.get("is_error")):
            return True
    return False


def _text_message(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "text": truncate(text, TEXT_CAP)}


def _pending_tool_entry(
    name: Any, args: Any, entry_id: Any, redact: Optional[Redactor]
) -> dict[str, Any]:
    name = name or "tool"
    entry: dict[str, Any] = {
        "role": "tool",
        "text": "",
        "tool_call": {
            "name": name,
            "label": name,
            "status": "pending",
            "args_preview": make_preview(args, redact),
        },
    }
    if entry_id:
        entry["message_id"] = str(entry_id)
    return entry


def _resolve_tool_entry(
    entry: dict[str, Any], is_error: bool, result: Any, redact: Optional[Redactor]
) -> None:
    entry["tool_call"]["status"] = "error" if is_error else "ok"
    preview_source = result if isinstance(result, str) else to_text(result)
    entry["tool_call"]["result_preview"] = make_preview(preview_source, redact)


def openai_transcript(
    messages: Optional[Sequence[Any]],
    response_message: Any = None,
    *,
    redact: Optional[Redactor] = None,
) -> list[dict[str, Any]]:
    """Rebuild the dialog from OpenAI-shaped request messages + response message."""
    out: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}

    def handle(msg: Any) -> None:
        role = get_field(msg, "role")
        if role in _SKIPPED_ROLES:
            return
        if role == "user":
            out.append(_text_message("user", to_text(get_field(msg, "content"))))
            return
        if role == "assistant":
            text = to_text(get_field(msg, "content"))
            if text:
                out.append(_text_message("bot", text))
            for tc in get_field(msg, "tool_calls") or ():
                func = get_field(tc, "function")
                tc_id = get_field(tc, "id")
                entry = _pending_tool_entry(
                    get_field(func, "name"), get_field(func, "arguments"), tc_id, redact
                )
                if tc_id:
                    pending[str(tc_id)] = entry
                out.append(entry)
            return
        if role in ("tool", "function"):
            tc_id = get_field(msg, "tool_call_id")
            content = get_field(msg, "content")
            entry = pending.pop(str(tc_id), None) if tc_id is not None else None
            if entry is None:
                # Orphan result (history truncated upstream): best-effort entry.
                entry = _pending_tool_entry(get_field(msg, "name"), None, tc_id, redact)
                del entry["tool_call"]["args_preview"]
                out.append(entry)
            explicit = get_field(msg, "is_error")
            if explicit is None:
                is_error = looks_like_error(to_text(content))
            else:
                is_error = bool(explicit)
            _resolve_tool_entry(entry, is_error, content, redact)
            return
        # Unknown roles are ignored (best-effort snapshot).

    for msg in messages or ():
        handle(msg)
    if response_message is not None:
        handle(response_message)
    return out


def openai_system_prompt(messages: Optional[Sequence[Any]]) -> Optional[str]:
    """First system/developer message text, for sync_agent descriptions."""
    for msg in messages or ():
        if get_field(msg, "role") in _SKIPPED_ROLES:
            text = to_text(get_field(msg, "content"))
            if text.strip():
                return text
    return None


def anthropic_transcript(
    messages: Optional[Sequence[Any]],
    response: Any = None,
    *,
    redact: Optional[Redactor] = None,
) -> list[dict[str, Any]]:
    """Rebuild the dialog from Anthropic-shaped request messages + response.

    `response` is the response message itself (``role`` + ``content`` blocks),
    or a synthetic ``{"role": "assistant", "content": [...]}`` dict from the
    streaming accumulator (where tool_use ``input`` may be a raw JSON string).
    """
    out: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}

    def flush_text(role: str, texts: list[str]) -> None:
        text = "\n".join(t for t in texts if t)
        if text:
            out.append(_text_message(role, text))
        texts.clear()

    def handle_message(role: Any, content: Any) -> None:
        out_role = "user" if role == "user" else "bot"
        if content is None or isinstance(content, str):
            if content:
                out.append(_text_message(out_role, content))
            return
        texts: list[str] = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
                continue
            btype = get_field(block, "type")
            if btype == "text":
                text = get_field(block, "text")
                if isinstance(text, str) and text:
                    texts.append(text)
            elif btype == "tool_use":
                flush_text(out_role, texts)
                block_id = get_field(block, "id")
                entry = _pending_tool_entry(
                    get_field(block, "name"), get_field(block, "input"), block_id, redact
                )
                if block_id:
                    pending[str(block_id)] = entry
                out.append(entry)
            elif btype == "tool_result":
                flush_text(out_role, texts)
                tu_id = get_field(block, "tool_use_id")
                entry = pending.pop(str(tu_id), None) if tu_id is not None else None
                if entry is None:
                    entry = _pending_tool_entry(None, None, tu_id, redact)
                    del entry["tool_call"]["args_preview"]
                    out.append(entry)
                _resolve_tool_entry(
                    entry,
                    bool(get_field(block, "is_error")),
                    get_field(block, "content"),
                    redact,
                )
            else:
                texts.append("[non-text content]")
        flush_text(out_role, texts)

    for msg in messages or ():
        role = get_field(msg, "role")
        if role in _SKIPPED_ROLES:
            continue
        handle_message(role, get_field(msg, "content"))
    if response is not None:
        handle_message(get_field(response, "role") or "assistant", get_field(response, "content"))
    return out


def anthropic_system_text(system: Any) -> Optional[str]:
    """Normalize the Anthropic `system` param (str | list of text blocks)."""
    if system is None:
        return None
    text = to_text(system)
    return text if text.strip() else None


def first_user_text(transcript: Sequence[dict[str, Any]]) -> str:
    for msg in transcript:
        if msg.get("role") == "user":
            return msg.get("text") or ""
    return ""


def fallback_session_id(transcript: Sequence[dict[str, Any]]) -> str:
    """Best-effort stable session id: sha256 of the first user message text."""
    return hashlib.sha256(first_user_text(transcript).encode("utf-8")).hexdigest()


def copy_transcript(transcript: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shallow-copy a transcript so later in-place tool resolution can't race
    with a snapshot already queued for the background worker."""
    out: list[dict[str, Any]] = []
    for msg in transcript:
        copied = dict(msg)
        tool_call = copied.get("tool_call")
        if isinstance(tool_call, dict):
            copied["tool_call"] = dict(tool_call)
        out.append(copied)
    return out
