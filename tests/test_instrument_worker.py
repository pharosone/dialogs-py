"""PharosInstrumentation worker tests: bounded queue, drain, close semantics."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace as NS

from pharosone_dialogs.instrument import PharosInstrumentation


class WorkerTestCase(unittest.TestCase):
    def make_blocking_pharos(self):
        started = threading.Event()
        release = threading.Event()
        processed = []

        def send_dialog(agent_id, session_id, messages, *, end_user=None):
            processed.append(session_id)
            started.set()
            release.wait(5)
            return {"status": "received", "dialog_id": "d", "flagged": False, "fast_scan": "ok"}

        return NS(send_dialog=send_dialog), started, release, processed

    def test_overflow_drops_oldest_with_warning(self) -> None:
        pharos, started, release, processed = self.make_blocking_pharos()
        inst = PharosInstrumentation(pharos, "bot", max_queue=1)
        self.addCleanup(inst.close, 5.0)
        try:
            inst.submit(session_id="s1", messages=[])
            self.assertTrue(started.wait(5))  # s1 is in flight, queue is empty
            inst.submit(session_id="s2", messages=[])  # queued
            with self.assertLogs("pharosone_dialogs.instrument", level="WARNING") as logs:
                inst.submit(session_id="s3", messages=[])  # drops s2
            self.assertTrue(any("dropping oldest" in line for line in logs.output))
        finally:
            release.set()
        self.assertTrue(inst.drain(5.0))
        self.assertEqual(processed, ["s1", "s3"])

    def test_drain_times_out_then_succeeds(self) -> None:
        pharos, started, release, _ = self.make_blocking_pharos()
        inst = PharosInstrumentation(pharos, "bot")
        self.addCleanup(inst.close, 5.0)
        inst.submit(session_id="s1", messages=[])
        self.assertTrue(started.wait(5))
        self.assertFalse(inst.drain(timeout=0.05))  # still blocked
        release.set()
        self.assertTrue(inst.drain(5.0))

    def test_close_flushes_then_drops_later_submissions(self) -> None:
        processed = []
        pharos = NS(
            send_dialog=lambda agent_id, session_id, messages, **kw: processed.append(session_id)
        )
        inst = PharosInstrumentation(pharos, "bot")
        inst.submit(session_id="s1", messages=[])
        inst.close(timeout=5.0)
        self.assertEqual(processed, ["s1"])
        inst.submit(session_id="s2", messages=[])  # after close: dropped, no error
        self.assertEqual(processed, ["s1"])
        self.assertFalse(inst._thread.is_alive())

    def test_on_result_exception_is_swallowed(self) -> None:
        pharos = NS(
            send_dialog=lambda agent_id, session_id, messages, **kw: {"fast_scan": "ok"}
        )

        def on_result(result):
            raise RuntimeError("callback exploded")

        inst = PharosInstrumentation(pharos, "bot", on_result=on_result)
        self.addCleanup(inst.close, 5.0)
        with self.assertLogs("pharosone_dialogs.instrument", level="ERROR"):
            inst.submit(session_id="s1", messages=[])
            self.assertTrue(inst.drain(5.0))


if __name__ == "__main__":
    unittest.main()
