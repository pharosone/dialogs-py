"""LangChain integration: a callback handler that mirrors every chat-model run
into PharosOne as a full-dialog snapshot.

Usage::

    from pharosone_dialogs import PharosOne
    from pharosone_dialogs.integrations.langchain import PharosCallbackHandler

    handler = PharosCallbackHandler(pharos, "support-bot", session_id="sess-42")
    llm.invoke(messages, config={"callbacks": [handler]})

``langchain-core`` is imported lazily: this module imports fine without it,
and a clear ImportError is raised only when :func:`PharosCallbackHandler` is
actually called. The returned object subclasses
``langchain_core.callbacks.BaseCallbackHandler``.

LangChain messages are converted to OpenAI-shaped dicts and fed through the
same transcript mapping as ``wrap_openai`` (one shared internal module —
``pharosone_dialogs.instrument._transcript``).
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Optional

from ..instrument import _transcript as _t
from ..instrument._session import current_session_id, resolve_session_id
from ..instrument._worker import DEFAULT_MAX_QUEUE, PharosInstrumentation

__all__ = ["PharosCallbackHandler"]

logger = logging.getLogger("pharosone_dialogs.integrations.langchain")


def _lc_to_openai(message: Any) -> Optional[dict[str, Any]]:
    """Convert a LangChain BaseMessage (duck-typed) to an OpenAI-shaped dict."""
    mtype = _t.get_field(message, "type") or _t.get_field(message, "role")
    content = _t.get_field(message, "content")
    if mtype in ("system", "developer"):
        return {"role": "system", "content": content}
    if mtype in ("human", "user"):
        return {"role": "user", "content": content}
    if mtype in ("ai", "assistant"):
        out: dict[str, Any] = {"role": "assistant", "content": content}
        tool_calls: list[Any] = []
        for tc in _t.get_field(message, "tool_calls") or ():
            args = _t.get_field(tc, "args")
            if not isinstance(args, str):
                try:
                    args = json.dumps(args, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    args = str(args)
            tool_calls.append(
                {
                    "id": _t.get_field(tc, "id"),
                    "type": "function",
                    "function": {
                        "name": _t.get_field(tc, "name") or "tool",
                        "arguments": args,
                    },
                }
            )
        if not tool_calls:
            extra = _t.get_field(message, "additional_kwargs") or {}
            tool_calls = list(_t.get_field(extra, "tool_calls") or ())
        if tool_calls:
            out["tool_calls"] = tool_calls
        return out
    if mtype == "tool":
        out = {
            "role": "tool",
            "tool_call_id": _t.get_field(message, "tool_call_id"),
            "content": content,
        }
        name = _t.get_field(message, "name")
        if name:
            out["name"] = name
        if _t.get_field(message, "status") == "error":
            out["is_error"] = True
        return out
    return None  # unknown message types are skipped (best-effort)


def _lc_response_message(response: Any) -> Optional[dict[str, Any]]:
    """Extract the assistant message from an LLMResult (duck-typed)."""
    generations = _t.get_field(response, "generations") or ()
    first_batch = generations[0] if generations else ()
    first = first_batch[0] if first_batch else None
    if first is None:
        return None
    message = _t.get_field(first, "message")
    if message is not None:
        converted = _lc_to_openai(message)
        if converted is not None:
            return converted
    text = _t.get_field(first, "text")
    if text:
        return {"role": "assistant", "content": text}
    return None


class _PharosCallbackHandlerImpl:
    """All handler logic, langchain-free. :func:`PharosCallbackHandler` mixes
    this into ``langchain_core.callbacks.BaseCallbackHandler``."""

    def __init__(
        self,
        pharos: Any,
        agent_id: str,
        session_id: Optional[str] = None,
        *,
        sync_agent: bool = False,
        on_result: Optional[Callable[[dict[str, Any]], Any]] = None,
        redact: Optional[Callable[[str], str]] = None,
        max_queue: int = DEFAULT_MAX_QUEUE,
    ) -> None:
        try:
            super().__init__()
        except Exception:  # pragma: no cover - permissive toward exotic bases
            pass
        self.instrumentation = PharosInstrumentation(
            pharos,
            agent_id,
            session_id=session_id,
            sync_agent=sync_agent,
            on_result=on_result,
            redact=redact,
            max_queue=max_queue,
        )
        self._lock = threading.Lock()
        self._runs: dict[str, list[dict[str, Any]]] = {}  # run_id -> request msgs
        self._sessions: dict[str, list[dict[str, Any]]] = {}  # session -> transcript
        self._tool_runs: dict[str, tuple[str, dict[str, Any]]] = {}
        self._last_session: Optional[str] = None

    # -- lifecycle -----------------------------------------------------------

    def drain(self, timeout: Optional[float] = None) -> bool:
        return self.instrumentation.drain(timeout)

    def close(self, timeout: Optional[float] = 5.0) -> None:
        self.instrumentation.close(timeout)

    # -- LLM callbacks ---------------------------------------------------------

    def on_chat_model_start(
        self, serialized: Any, messages: Any, *, run_id: Any = None, **kwargs: Any
    ) -> None:
        try:
            batch = messages[0] if messages else []
            converted = [
                m for m in (_lc_to_openai(msg) for msg in batch) if m is not None
            ]
            with self._lock:
                self._runs[str(run_id)] = converted
        except Exception:
            logger.exception("pharosone: on_chat_model_start failed")

    def on_llm_end(self, response: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        try:
            with self._lock:
                request_messages = self._runs.pop(str(run_id), [])
            redact = self.instrumentation.redact
            transcript = _t.openai_transcript(
                request_messages, _lc_response_message(response), redact=redact
            )
            session_id = resolve_session_id(
                None,
                current_session_id(),
                self.instrumentation.default_session_id,
                transcript,
            )
            system_prompt = _t.openai_system_prompt(request_messages)
            with self._lock:
                self._sessions[session_id] = transcript
                self._last_session = session_id
                snapshot = _t.copy_transcript(transcript)
            self.instrumentation.submit(
                session_id=session_id, messages=snapshot, system_prompt=system_prompt
            )
        except Exception:
            logger.exception("pharosone: on_llm_end failed")

    def on_llm_error(self, error: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        with self._lock:
            self._runs.pop(str(run_id), None)

    # -- tool callbacks -----------------------------------------------------------

    def on_tool_start(
        self, serialized: Any, input_str: Any, *, run_id: Any = None, **kwargs: Any
    ) -> None:
        try:
            with self._lock:
                session_id = (
                    current_session_id()
                    or self.instrumentation.default_session_id
                    or self._last_session
                )
                if session_id is None:
                    return  # no dialog to attach the tool run to yet
                transcript = self._sessions.setdefault(session_id, [])
                name = (_t.get_field(serialized, "name") or "tool") if serialized else "tool"
                entry: dict[str, Any] = {
                    "role": "tool",
                    "text": "",
                    "message_id": f"lc-tool-{run_id}",
                    "tool_call": {
                        "name": name,
                        "label": name,
                        "status": "pending",
                        "args_preview": _t.make_preview(
                            input_str, self.instrumentation.redact
                        ),
                    },
                }
                transcript.append(entry)
                self._tool_runs[str(run_id)] = (session_id, entry)
                snapshot = _t.copy_transcript(transcript)
            self.instrumentation.submit(session_id=session_id, messages=snapshot)
        except Exception:
            logger.exception("pharosone: on_tool_start failed")

    def on_tool_end(self, output: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        self._resolve_tool(run_id, "ok", output)

    def on_tool_error(self, error: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        self._resolve_tool(run_id, "error", error)

    def _resolve_tool(self, run_id: Any, status: str, output: Any) -> None:
        try:
            with self._lock:
                bound = self._tool_runs.pop(str(run_id), None)
                if bound is None:
                    return
                session_id, entry = bound
                entry["tool_call"]["status"] = status
                entry["tool_call"]["result_preview"] = _t.make_preview(
                    output if isinstance(output, str) else str(output),
                    self.instrumentation.redact,
                )
                snapshot = _t.copy_transcript(self._sessions.get(session_id, []))
            self.instrumentation.submit(session_id=session_id, messages=snapshot)
        except Exception:
            logger.exception("pharosone: tool callback failed")


def _load_base_callback_handler() -> type:
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError as exc:
        raise ImportError(
            "PharosCallbackHandler requires langchain-core; "
            "install it with: pip install langchain-core"
        ) from exc
    return BaseCallbackHandler


_handler_cls: Optional[type] = None


def PharosCallbackHandler(  # noqa: N802 - factory named like the class it builds
    pharos: Any,
    agent_id: str,
    session_id: Optional[str] = None,
    *,
    sync_agent: bool = False,
    on_result: Optional[Callable[[dict[str, Any]], Any]] = None,
    redact: Optional[Callable[[str], str]] = None,
    max_queue: int = DEFAULT_MAX_QUEUE,
) -> Any:
    """Build a PharosOne LangChain callback handler.

    The returned object subclasses ``langchain_core.callbacks.BaseCallbackHandler``
    (imported lazily — raises a clear ImportError here if langchain-core is
    missing) and implements ``on_chat_model_start`` / ``on_llm_end`` (snapshot
    per run, keyed by LangChain ``run_id``) plus ``on_tool_start`` /
    ``on_tool_end`` / ``on_tool_error`` (pending -> ok/error tool entries).

    Session resolution per snapshot: the innermost ``pharos_session(...)``
    scope, else `session_id`, else a stable sha256 of the first user message
    text. Flushing is fire-and-forget via a background worker; use
    ``handler.drain()`` / ``handler.close()`` (or ``handler.instrumentation``)
    for deterministic flushing. `on_result` receives each send_dialog result
    dict from the worker.
    """
    global _handler_cls
    base = _load_base_callback_handler()
    if _handler_cls is None or not issubclass(_handler_cls, base):
        _handler_cls = type(
            "PharosCallbackHandler",
            (_PharosCallbackHandlerImpl, base),
            {"__module__": __name__},
        )
    return _handler_cls(
        pharos,
        agent_id,
        session_id,
        sync_agent=sync_agent,
        on_result=on_result,
        redact=redact,
        max_queue=max_queue,
    )
