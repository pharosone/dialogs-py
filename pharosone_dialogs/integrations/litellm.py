"""LiteLLM integration: a ``success_callback``-compatible function that mirrors
every completed LiteLLM call into PharosOne as a full-dialog snapshot.

Usage::

    import litellm
    from pharosone_dialogs import PharosOne
    from pharosone_dialogs.integrations.litellm import pharos_litellm_callback

    litellm.success_callback = [pharos_litellm_callback(pharos, "support-bot")]

``litellm`` itself is never imported here — the callback simply matches the
custom-callback signature ``(kwargs, completion_response, start_time,
end_time)``. LiteLLM speaks the OpenAI shape, so the payload goes through the
same transcript mapping as ``wrap_openai``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from ..instrument import _transcript as _t
from ..instrument._session import current_session_id, resolve_session_id
from ..instrument._worker import DEFAULT_MAX_QUEUE, PharosInstrumentation

__all__ = ["pharos_litellm_callback"]

logger = logging.getLogger("pharosone_dialogs.integrations.litellm")


def _metadata_session(kwargs: Any) -> Optional[str]:
    """Per-call override: metadata={"pharos_session_id": ...} on the request."""
    litellm_params = _t.get_field(kwargs, "litellm_params") or {}
    metadata = (
        _t.get_field(litellm_params, "metadata")
        or _t.get_field(kwargs, "metadata")
        or {}
    )
    value = _t.get_field(metadata, "pharos_session_id")
    return value if isinstance(value, str) and value else None


def pharos_litellm_callback(
    pharos: Any,
    agent_id: str,
    *,
    session_id: Optional[str] = None,
    sync_agent: bool = False,
    on_result: Optional[Callable[[dict[str, Any]], Any]] = None,
    redact: Optional[Callable[[str], str]] = None,
    max_queue: int = DEFAULT_MAX_QUEUE,
) -> Callable[..., None]:
    """Build a LiteLLM success callback bound to one PharosOne agent.

    The returned callable accepts LiteLLM's custom-callback signature
    ``(kwargs, completion_response, start_time, end_time)`` — assign it to
    ``litellm.success_callback``. It reads ``kwargs["messages"]`` plus the
    OpenAI-shaped ``completion_response`` (falling back to
    ``kwargs["completion_response"]`` / ``kwargs["response_obj"]``), rebuilds
    the transcript, and queues a fire-and-forget ``send_dialog`` snapshot.
    It never raises into LiteLLM.

    Session resolution: request ``metadata={"pharos_session_id": ...}``, else
    the innermost ``pharos_session(...)`` scope, else `session_id`, else a
    stable sha256 of the first user message text. The worker is exposed as
    ``callback.instrumentation`` (``drain()`` / ``close()``).
    """
    inst = PharosInstrumentation(
        pharos,
        agent_id,
        session_id=session_id,
        sync_agent=sync_agent,
        on_result=on_result,
        redact=redact,
        max_queue=max_queue,
    )

    def callback(
        kwargs: Any,
        completion_response: Any = None,
        start_time: Any = None,
        end_time: Any = None,
    ) -> None:
        try:
            kwargs = kwargs or {}
            messages = _t.get_field(kwargs, "messages") or []
            response_obj = completion_response
            if response_obj is None:
                response_obj = _t.get_field(kwargs, "completion_response") or _t.get_field(
                    kwargs, "response_obj"
                )
            response_message = None
            if response_obj is not None:
                choices = _t.get_field(response_obj, "choices") or ()
                if choices:
                    response_message = _t.get_field(choices[0], "message")
            transcript = _t.openai_transcript(
                messages, response_message, redact=inst.redact
            )
            resolved = resolve_session_id(
                _metadata_session(kwargs),
                current_session_id(),
                inst.default_session_id,
                transcript,
            )
            inst.submit(
                session_id=resolved,
                messages=transcript,
                system_prompt=_t.openai_system_prompt(messages),
            )
        except Exception:
            logger.exception("pharosone: litellm callback failed")

    callback.instrumentation = inst  # type: ignore[attr-defined]
    return callback
