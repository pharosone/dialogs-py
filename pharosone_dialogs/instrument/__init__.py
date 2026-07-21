"""Sit-on-the-wire instrumentation: wrap an existing OpenAI / Anthropic (or
OpenAI-compatible) client and mirror every completed chat call into PharosOne
as a full-dialog ``send_dialog`` snapshot — no manual send calls.

Quickstart::

    from pharosone_dialogs import PharosOne
    from pharosone_dialogs.instrument import wrap_openai, pharos_session

    pharos = PharosOne(base_url="https://pharosone.example.com", api_key="sk-...")
    client = wrap_openai(OpenAI(), pharos=pharos, agent_id="support-bot")

    with pharos_session("sess-42"):
        client.chat.completions.create(model="gpt-5.5", messages=[...])

Zero new dependencies: ``openai`` / ``anthropic`` are never imported — the
wrappers are duck-typed proxies. Flushing happens fire-and-forget on a daemon
thread (``wrapped.pharos_instrumentation.drain()`` / ``.close()`` to flush
deterministically); PharosOne errors are logged, never raised into the caller.
"""

from ._session import current_session_id, pharos_session
from ._worker import PharosInstrumentation
from ._wrappers import wrap_anthropic, wrap_openai

__all__ = [
    "wrap_openai",
    "wrap_anthropic",
    "pharos_session",
    "current_session_id",
    "PharosInstrumentation",
]
