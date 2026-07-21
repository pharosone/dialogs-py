"""Scoped session binding for instrumented provider calls (contextvars)."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence

from ._transcript import fallback_session_id

__all__ = ["pharos_session", "current_session_id", "resolve_session_id"]

_session_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "pharos_session_id", default=None
)


@contextmanager
def pharos_session(session_id: str) -> Iterator[str]:
    """Bind `session_id` for every instrumented call inside the `with` block.

    Backed by `contextvars`, so it is safe across threads and asyncio tasks:
    the binding only applies to the current execution context. Nesting is
    allowed; the innermost binding wins. A per-call `pharos_session_id=`
    kwarg on the instrumented method still overrides this scope.
    """
    token = _session_var.set(session_id)
    try:
        yield session_id
    finally:
        _session_var.reset(token)


def current_session_id() -> Optional[str]:
    """The session id bound by the innermost `pharos_session`, if any."""
    return _session_var.get()


def resolve_session_id(
    per_call: Optional[str],
    scoped: Optional[str],
    default: Optional[str],
    transcript: Sequence[dict[str, Any]],
) -> str:
    """Pick the session id for a snapshot. Precedence (explicit wins):

    1. per-call ``pharos_session_id=`` kwarg (stripped before the provider),
    2. the innermost ``pharos_session(...)`` scope,
    3. the ``session_id=`` fixed at wrap/handler-construction time,
    4. best-effort fallback: sha256 of the first user message text.
    """
    if per_call:
        return per_call
    if scoped:
        return scoped
    if default:
        return default
    return fallback_session_id(transcript)
