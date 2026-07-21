"""Transparent provider-client proxies: wrap_openai / wrap_anthropic.

Pure duck-typing — neither `openai` nor `anthropic` is imported (they are not
dependencies of this package; any object with the right method surface works,
which is also what makes OpenAI-compatible endpoints — Ollama, vLLM,
OpenRouter, ... — work for free).

Only the completion method is instrumented:

- OpenAI:    ``chat.completions.create``  (sync + async, ``stream=True``)
- Anthropic: ``messages.create``          (sync + async, ``stream=True``;
  the ``messages.stream()`` helper context manager is NOT instrumented yet —
  it passes through untouched)

Every other attribute access is delegated to the wrapped client unchanged,
and the instrumented method returns the provider's exact response object.
Instrumentation failures are logged and never raised into the caller.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, Optional

from . import _transcript as _t
from ._session import current_session_id, resolve_session_id
from ._worker import DEFAULT_MAX_QUEUE, PharosInstrumentation

__all__ = ["wrap_openai", "wrap_anthropic"]

logger = logging.getLogger("pharosone_dialogs.instrument")

_OPENAI = "openai"
_ANTHROPIC = "anthropic"
_HOOK_PATHS = {
    _OPENAI: ("chat", "completions", "create"),
    _ANTHROPIC: ("messages", "create"),
}


# -- request/response capture --------------------------------------------------


class _Capture:
    """Request-side snapshot taken before the provider call; `finish()` builds
    the transcript and hands it to the background worker (at most once)."""

    __slots__ = ("inst", "provider", "messages", "system", "per_call", "scoped", "done")

    def __init__(
        self,
        inst: PharosInstrumentation,
        kwargs: dict[str, Any],
        per_call_session: Optional[str],
        provider: str,
    ) -> None:
        self.inst = inst
        self.provider = provider
        self.per_call = per_call_session
        # Read the contextvar in the caller's context, not the worker thread.
        self.scoped = current_session_id()
        messages = kwargs.get("messages")
        self.messages = list(messages) if messages else []
        if provider == _OPENAI:
            self.system = _t.openai_system_prompt(self.messages)
        else:
            self.system = _t.anthropic_system_text(kwargs.get("system"))
        self.done = False

    def finish(self, response_message: Any) -> None:
        if self.done:
            return
        self.done = True
        try:
            if self.provider == _OPENAI:
                transcript = _t.openai_transcript(
                    self.messages, response_message, redact=self.inst.redact
                )
            else:
                transcript = _t.anthropic_transcript(
                    self.messages, response_message, redact=self.inst.redact
                )
            session_id = resolve_session_id(
                self.per_call, self.scoped, self.inst.default_session_id, transcript
            )
            self.inst.submit(
                session_id=session_id, messages=transcript, system_prompt=self.system
            )
        except Exception:
            logger.exception("pharosone: failed to build/submit dialog snapshot")


def _response_message(result: Any, provider: str) -> Any:
    if provider == _OPENAI:
        try:
            choices = _t.get_field(result, "choices") or ()
            return _t.get_field(choices[0], "message") if choices else None
        except Exception:
            return None
    return result  # Anthropic: the response IS the assistant message.


# -- streaming accumulation ------------------------------------------------------


class _OpenAIAccumulator:
    """Accumulate chat-completion chunks into an assistant message dict."""

    def __init__(self) -> None:
        self._text: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}

    def feed(self, chunk: Any) -> None:
        choices = _t.get_field(chunk, "choices") or ()
        if not choices:
            return
        delta = _t.get_field(choices[0], "delta")
        if delta is None:
            return
        content = _t.get_field(delta, "content")
        if isinstance(content, str) and content:
            self._text.append(content)
        for tc in _t.get_field(delta, "tool_calls") or ():
            index = _t.get_field(tc, "index")
            slot = self._tool_calls.setdefault(
                0 if index is None else index, {"id": None, "name": "", "arguments": ""}
            )
            tc_id = _t.get_field(tc, "id")
            if tc_id:
                slot["id"] = tc_id
            func = _t.get_field(tc, "function")
            if func is not None:
                name = _t.get_field(func, "name")
                if name:
                    slot["name"] += name
                args = _t.get_field(func, "arguments")
                if args:
                    slot["arguments"] += args

    def response_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": "".join(self._text)}
        if self._tool_calls:
            message["tool_calls"] = [
                {
                    "id": slot["id"],
                    "type": "function",
                    "function": {
                        "name": slot["name"] or "tool",
                        "arguments": slot["arguments"],
                    },
                }
                for _, slot in sorted(self._tool_calls.items())
            ]
        return message


class _AnthropicAccumulator:
    """Accumulate messages.create(stream=True) events into an assistant message.

    tool_use input stays the accumulated raw JSON string — exactly what
    args_preview wants."""

    def __init__(self) -> None:
        self._blocks: dict[int, dict[str, Any]] = {}

    def feed(self, event: Any) -> None:
        etype = _t.get_field(event, "type")
        if etype == "content_block_start":
            index = _t.get_field(event, "index")
            block = _t.get_field(event, "content_block")
            self._blocks[0 if index is None else index] = {
                "type": _t.get_field(block, "type"),
                "id": _t.get_field(block, "id"),
                "name": _t.get_field(block, "name"),
                "text": _t.get_field(block, "text") or "",
                "input_json": "",
            }
        elif etype == "content_block_delta":
            index = _t.get_field(event, "index")
            slot = self._blocks.setdefault(
                0 if index is None else index,
                {"type": "text", "id": None, "name": None, "text": "", "input_json": ""},
            )
            delta = _t.get_field(event, "delta")
            dtype = _t.get_field(delta, "type")
            if dtype == "text_delta":
                text = _t.get_field(delta, "text")
                if text:
                    slot["text"] += text
            elif dtype == "input_json_delta":
                partial = _t.get_field(delta, "partial_json")
                if partial:
                    slot["input_json"] += partial

    def response_message(self) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for _, slot in sorted(self._blocks.items()):
            if slot["type"] == "tool_use":
                content.append(
                    {
                        "type": "tool_use",
                        "id": slot["id"],
                        "name": slot["name"],
                        "input": slot["input_json"],
                    }
                )
            elif slot["text"]:
                content.append({"type": "text", "text": slot["text"]})
        return {"role": "assistant", "content": content}


_ACCUMULATORS = {_OPENAI: _OpenAIAccumulator, _ANTHROPIC: _AnthropicAccumulator}


class _InstrumentedStream:
    """Wraps a provider stream: yields chunks unchanged, accumulates them, and
    flushes the snapshot when the stream completes. A stream that is never
    fully consumed flushes what was seen on close()/context-exit/GC."""

    def __init__(self, inner: Any, accumulator: Any, capture: _Capture) -> None:
        self._pharos_inner = inner
        self._pharos_iter = iter(inner)
        self._pharos_acc = accumulator
        self._pharos_capture = capture

    def _pharos_finish(self) -> None:
        capture = self._pharos_capture
        if capture is None or capture.done:
            return
        try:
            capture.finish(self._pharos_acc.response_message())
        except Exception:
            logger.exception("pharosone: stream flush failed")

    def __iter__(self) -> "_InstrumentedStream":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._pharos_iter)
        except BaseException:
            self._pharos_finish()
            raise
        try:
            self._pharos_acc.feed(chunk)
        except Exception:
            logger.exception("pharosone: stream accumulation failed")
        return chunk

    def close(self) -> Any:
        self._pharos_finish()
        close = getattr(self._pharos_inner, "close", None)
        if close is not None:
            return close()
        return None

    def __enter__(self) -> "_InstrumentedStream":
        enter = getattr(self._pharos_inner, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, *exc_info: Any) -> Any:
        self._pharos_finish()
        exit_ = getattr(self._pharos_inner, "__exit__", None)
        if exit_ is not None:
            return exit_(*exc_info)
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pharos_inner, name)

    def __del__(self) -> None:
        try:
            self._pharos_finish()
        except Exception:
            pass


class _InstrumentedAsyncStream:
    """Async-iterator counterpart of `_InstrumentedStream`."""

    def __init__(self, inner: Any, accumulator: Any, capture: _Capture) -> None:
        self._pharos_inner = inner
        self._pharos_iter = aiter(inner)
        self._pharos_acc = accumulator
        self._pharos_capture = capture

    def _pharos_finish(self) -> None:
        capture = self._pharos_capture
        if capture is None or capture.done:
            return
        try:
            capture.finish(self._pharos_acc.response_message())
        except Exception:
            logger.exception("pharosone: stream flush failed")

    def __aiter__(self) -> "_InstrumentedAsyncStream":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._pharos_iter.__anext__()
        except BaseException:
            self._pharos_finish()
            raise
        try:
            self._pharos_acc.feed(chunk)
        except Exception:
            logger.exception("pharosone: stream accumulation failed")
        return chunk

    async def aclose(self) -> Any:
        self._pharos_finish()
        aclose = getattr(self._pharos_inner, "aclose", None)
        if aclose is not None:
            return await aclose()
        return None

    async def __aenter__(self) -> "_InstrumentedAsyncStream":
        enter = getattr(self._pharos_inner, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, *exc_info: Any) -> Any:
        self._pharos_finish()
        exit_ = getattr(self._pharos_inner, "__aexit__", None)
        if exit_ is not None:
            return await exit_(*exc_info)
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pharos_inner, name)

    def __del__(self) -> None:
        try:
            self._pharos_finish()
        except Exception:
            pass


# -- the instrumented completion method ------------------------------------------


def _make_wrapped_create(
    create: Callable[..., Any], inst: PharosInstrumentation, provider: str
) -> Callable[..., Any]:
    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        # Strip pharos_session_id unconditionally — it must NEVER reach the
        # provider, even if the capture below fails.
        per_call_session = kwargs.pop("pharos_session_id", None)
        capture: Optional[_Capture] = None
        try:
            capture = _Capture(inst, kwargs, per_call_session, provider)
        except Exception:
            logger.exception("pharosone: request capture failed")
        streaming = bool(kwargs.get("stream"))
        result = create(*args, **kwargs)
        if inspect.isawaitable(result):
            return _finish_async(result, capture, streaming, provider)
        return _finish_sync(result, capture, streaming, provider)

    try:
        functools.update_wrapper(wrapped_create, create)
    except Exception:
        pass
    return wrapped_create


def _finish_sync(
    result: Any, capture: Optional[_Capture], streaming: bool, provider: str
) -> Any:
    if capture is None:
        return result
    try:
        if streaming:
            return _InstrumentedStream(result, _ACCUMULATORS[provider](), capture)
        capture.finish(_response_message(result, provider))
    except Exception:
        logger.exception("pharosone: response capture failed")
    return result


async def _finish_async(
    awaitable: Any, capture: Optional[_Capture], streaming: bool, provider: str
) -> Any:
    result = await awaitable
    if capture is None:
        return result
    try:
        if streaming:
            return _InstrumentedAsyncStream(result, _ACCUMULATORS[provider](), capture)
        capture.finish(_response_message(result, provider))
    except Exception:
        logger.exception("pharosone: response capture failed")
    return result


# -- the transparent proxy ---------------------------------------------------------


class _ChildProxy:
    """Intermediate node on the way to the instrumented method
    (e.g. ``client.chat`` / ``client.chat.completions``)."""

    __slots__ = ("_target", "_path", "_hooks", "_prefixes")

    def __init__(
        self,
        target: Any,
        path: tuple[str, ...],
        hooks: dict[tuple[str, ...], Callable[[Any], Any]],
        prefixes: frozenset[tuple[str, ...]],
    ) -> None:
        self._target = target
        self._path = path
        self._hooks = hooks
        self._prefixes = prefixes

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        path = self._path + (name,)
        hook = self._hooks.get(path)
        if hook is not None:
            return hook(attr)
        if path in self._prefixes:
            return _ChildProxy(attr, path, self._hooks, self._prefixes)
        return attr

    def __repr__(self) -> str:
        return repr(self._target)


class InstrumentedClientProxy:
    """Transparent duck-typed proxy over a provider client.

    Everything is delegated to the wrapped client (reads AND writes); only
    the hooked method path returns an instrumented callable. The owning
    :class:`PharosInstrumentation` is exposed as ``.pharos_instrumentation``
    (``drain()`` / ``close()`` / ``atexit`` flush live there).
    """

    def __init__(
        self,
        target: Any,
        hooks: dict[tuple[str, ...], Callable[[Any], Any]],
        instrumentation: PharosInstrumentation,
    ) -> None:
        prefixes = frozenset(
            path[:i] for path in hooks for i in range(1, len(path))
        )
        object.__setattr__(self, "_pharos_target", target)
        object.__setattr__(self, "_pharos_hooks", dict(hooks))
        object.__setattr__(self, "_pharos_prefixes", prefixes)
        object.__setattr__(self, "pharos_instrumentation", instrumentation)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._pharos_target, name)
        path = (name,)
        hook = self._pharos_hooks.get(path)
        if hook is not None:
            return hook(attr)
        if path in self._pharos_prefixes:
            return _ChildProxy(attr, path, self._pharos_hooks, self._pharos_prefixes)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._pharos_target, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._pharos_target, name)

    def __dir__(self) -> list[str]:
        return dir(self._pharos_target)

    def __repr__(self) -> str:
        return f"<pharos-instrumented {self._pharos_target!r}>"

    def __enter__(self) -> "InstrumentedClientProxy":
        self._pharos_target.__enter__()
        return self

    def __exit__(self, *exc_info: Any) -> Any:
        return self._pharos_target.__exit__(*exc_info)

    async def __aenter__(self) -> "InstrumentedClientProxy":
        await self._pharos_target.__aenter__()
        return self

    async def __aexit__(self, *exc_info: Any) -> Any:
        return await self._pharos_target.__aexit__(*exc_info)


# -- public factories ------------------------------------------------------------


def _wrap(
    client: Any,
    provider: str,
    pharos: Any,
    agent_id: str,
    session_id: Optional[str],
    sync_agent: bool,
    on_result: Optional[Callable[[dict[str, Any]], Any]],
    redact: Optional[Callable[[str], str]],
    max_queue: int,
) -> InstrumentedClientProxy:
    inst = PharosInstrumentation(
        pharos,
        agent_id,
        session_id=session_id,
        sync_agent=sync_agent,
        on_result=on_result,
        redact=redact,
        max_queue=max_queue,
    )
    hooks = {
        _HOOK_PATHS[provider]: (
            lambda create: _make_wrapped_create(create, inst, provider)
        )
    }
    return InstrumentedClientProxy(client, hooks, inst)


def wrap_openai(
    client: Any,
    *,
    pharos: Any,
    agent_id: str,
    session_id: Optional[str] = None,
    sync_agent: bool = False,
    on_result: Optional[Callable[[dict[str, Any]], Any]] = None,
    redact: Optional[Callable[[str], str]] = None,
    max_queue: int = DEFAULT_MAX_QUEUE,
) -> Any:
    """Wrap an OpenAI (or OpenAI-compatible) client; returns a transparent proxy.

    Instruments ``chat.completions.create`` on both ``OpenAI`` and
    ``AsyncOpenAI`` clients, including ``stream=True``: on every completed
    call the full transcript (request messages + response) is rebuilt and
    sent to PharosOne as a ``send_dialog`` snapshot from a background thread.
    The provider response is returned unchanged; the wrapped call never
    blocks on — or raises from — PharosOne.

    Because the wrapping is pure duck-typing, any client with the same
    surface works: Ollama, vLLM, OpenRouter, Azure OpenAI, ... via
    ``OpenAI(base_url=...)``.

    Args:
        client: the provider client to wrap (returned object proxies it).
        pharos: a :class:`pharosone_dialogs.PharosOne` instance.
        agent_id: the PharosOne agent this bot reports as (fixed at wrap time).
        session_id: default dialog session id. Per-call
            ``pharos_session_id=`` (stripped before the provider) and the
            ``pharos_session(...)`` scope override it; with none of the
            three, a stable sha256 of the first user message text is used
            (best-effort).
        sync_agent: when True and a system prompt is present, call
            ``upsert_agent(agent_id, description=<system prompt>)`` once per
            process (dedup by agent_id). Off by default.
        on_result: optional callback invoked from the background worker with
            each send_dialog result dict (``flagged``, ``fast_scan``, ...).
        redact: optional ``str -> str`` hook applied to tool args/result
            previews before they are sent.
        max_queue: bound of the background queue (drop-oldest on overflow).
    """
    return _wrap(
        client, _OPENAI, pharos, agent_id, session_id, sync_agent, on_result, redact, max_queue
    )


def wrap_anthropic(
    client: Any,
    *,
    pharos: Any,
    agent_id: str,
    session_id: Optional[str] = None,
    sync_agent: bool = False,
    on_result: Optional[Callable[[dict[str, Any]], Any]] = None,
    redact: Optional[Callable[[str], str]] = None,
    max_queue: int = DEFAULT_MAX_QUEUE,
) -> Any:
    """Wrap an Anthropic client; returns a transparent proxy.

    Instruments ``messages.create`` on both ``Anthropic`` and
    ``AsyncAnthropic`` clients, including ``create(stream=True)``. The
    ``messages.stream()`` helper context manager is NOT instrumented yet
    (TODO) — it passes through untouched. See :func:`wrap_openai` for the
    shared parameter semantics.
    """
    return _wrap(
        client, _ANTHROPIC, pharos, agent_id, session_id, sync_agent, on_result, redact, max_queue
    )
