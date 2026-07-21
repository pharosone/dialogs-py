"""Background flush worker: fire-and-forget send_dialog on a daemon thread.

The hot path (the caller's LLM call) never blocks on PharosOne and never sees
a PharosOne error: snapshots go into a bounded in-memory queue serviced by a
daemon thread; on overflow the OLDEST snapshot is dropped with a logged
warning (later snapshots supersede earlier ones anyway — send_dialog has
replace semantics).
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Optional, Sequence

from ._transcript import truncate

__all__ = ["PharosInstrumentation", "DEFAULT_MAX_QUEUE"]

logger = logging.getLogger("pharosone_dialogs.instrument")

DEFAULT_MAX_QUEUE = 256
_AGENT_DESCRIPTION_CAP = 2000

# sync_agent dedup: one upsert_agent attempt per agent_id per process.
_synced_agents: set[str] = set()
_synced_lock = threading.Lock()


class PharosInstrumentation:
    """Owns the background flush pipeline for one wrapped client / integration.

    Exposed on wrapped clients as ``wrapped.pharos_instrumentation`` and on
    the integrations as ``.instrumentation``.

    - ``submit(...)`` queues a snapshot; never blocks, never raises.
    - ``drain(timeout=None)`` blocks until everything queued is flushed
      (returns False on timeout). Registered with ``atexit`` so pending
      snapshots are flushed at interpreter shutdown (bounded wait).
    - ``close(timeout=5.0)`` drains, then stops the worker thread; later
      submissions are dropped.
    - PharosOne errors (and any other flush error) are logged, never raised.
    - ``on_result`` (if given) is called from the worker thread with the
      send_dialog result dict (``flagged`` / ``fast_scan`` / ``dialog_id``).
    """

    def __init__(
        self,
        pharos: Any,
        agent_id: str,
        *,
        session_id: Optional[str] = None,
        sync_agent: bool = False,
        on_result: Optional[Callable[[dict[str, Any]], Any]] = None,
        redact: Optional[Callable[[str], str]] = None,
        max_queue: int = DEFAULT_MAX_QUEUE,
    ) -> None:
        if max_queue < 1:
            raise ValueError("max_queue must be >= 1")
        self.pharos = pharos
        self.agent_id = agent_id
        self.default_session_id = session_id
        self.sync_agent = sync_agent
        self.on_result = on_result
        self.redact = redact
        self._max_queue = max_queue
        self._cond = threading.Condition()
        self._jobs: deque[tuple[str, list[dict[str, Any]], Optional[str]]] = deque()
        self._pending = 0  # queued + in-flight
        self._closed = False
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name=f"pharosone-instrument-{agent_id}", daemon=True
        )
        self._thread.start()
        atexit.register(self._atexit)

    def submit(
        self,
        *,
        session_id: str,
        messages: Sequence[dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> None:
        """Queue one dialog snapshot. Never blocks, never raises."""
        try:
            with self._cond:
                if self._closed:
                    logger.debug("pharosone: instrumentation closed; dropping snapshot")
                    return
                if len(self._jobs) >= self._max_queue:
                    self._jobs.popleft()
                    self._pending -= 1
                    logger.warning(
                        "pharosone: flush queue full (max %d); dropping oldest snapshot",
                        self._max_queue,
                    )
                self._jobs.append((session_id, list(messages), system_prompt))
                self._pending += 1
                self._cond.notify_all()
        except Exception:  # pragma: no cover - defensive, must never propagate
            logger.exception("pharosone: failed to queue snapshot")

    def drain(self, timeout: Optional[float] = None) -> bool:
        """Block until all queued snapshots are flushed. True when drained."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._pending > 0:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    def close(self, timeout: Optional[float] = 5.0) -> None:
        """Drain (bounded by `timeout`), then stop the worker thread."""
        with self._cond:
            self._closed = True
        self.drain(timeout)
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        self._thread.join(timeout)
        try:
            atexit.unregister(self._atexit)
        except Exception:  # pragma: no cover
            pass

    # -- worker ---------------------------------------------------------------

    def _atexit(self) -> None:
        try:
            self.close(timeout=10.0)
        except Exception:  # pragma: no cover
            pass

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._jobs and not self._stopping:
                    self._cond.wait()
                if self._jobs:
                    job = self._jobs.popleft()
                else:  # stopping and queue empty
                    return
            try:
                self._process(*job)
            except Exception:  # pragma: no cover - _process guards internally
                logger.exception("pharosone: snapshot flush failed")
            finally:
                with self._cond:
                    self._pending -= 1
                    self._cond.notify_all()

    def _process(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        system_prompt: Optional[str],
    ) -> None:
        if self.sync_agent and system_prompt:
            self._maybe_sync_agent(system_prompt)
        try:
            result = self.pharos.send_dialog(self.agent_id, session_id, messages)
        except Exception as exc:
            logger.warning(
                "pharosone: send_dialog failed for agent=%s session=%s: %s",
                self.agent_id,
                session_id,
                exc,
            )
            return
        if self.on_result is not None:
            try:
                self.on_result(result)
            except Exception:
                logger.exception("pharosone: on_result callback raised")

    def _maybe_sync_agent(self, system_prompt: str) -> None:
        with _synced_lock:
            if self.agent_id in _synced_agents:
                return
            _synced_agents.add(self.agent_id)  # one attempt per process
        try:
            self.pharos.upsert_agent(
                self.agent_id,
                description=truncate(system_prompt.strip(), _AGENT_DESCRIPTION_CAP),
            )
        except Exception as exc:
            logger.warning(
                "pharosone: upsert_agent failed for agent=%s: %s", self.agent_id, exc
            )
