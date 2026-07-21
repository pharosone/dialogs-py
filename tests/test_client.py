"""Smoke tests for the PharosOne client against a stdlib http.server."""

from __future__ import annotations

import json
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from pharosone_dialogs import PharosOne, PharosOneError


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        self.server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": json.loads(raw) if raw else None,
            }
        )
        if self.server.responses:
            status, payload = self.server.responses.pop(0)
        else:
            status, payload = 200, b"{}"
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # silence request logging
        pass


class ClientTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.requests = []
        self.server.responses = []
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.client = PharosOne(base_url=self.base_url, api_key="test-key")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    # --- construction / configuration -------------------------------------

    def test_missing_base_url_raises(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "PHAROSONE_BASE_URL"):
                PharosOne(api_key="k")

    def test_missing_api_key_raises(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "PHAROSONE_API_KEY"):
                PharosOne(base_url="http://x")

    def test_env_fallbacks_are_used(self) -> None:
        env = {"PHAROSONE_BASE_URL": self.base_url, "PHAROSONE_API_KEY": "env-key"}
        with mock.patch.dict("os.environ", env, clear=True):
            client = PharosOne()
        client.upsert_agent("a-1")
        request = self.server.requests[0]
        self.assertEqual(request["headers"]["authorization"], "Bearer env-key")

    def test_explicit_args_win_over_env(self) -> None:
        env = {"PHAROSONE_BASE_URL": "http://unused.invalid", "PHAROSONE_API_KEY": "env-key"}
        with mock.patch.dict("os.environ", env, clear=True):
            client = PharosOne(base_url=self.base_url, api_key="explicit-key")
        client.upsert_agent("a-1")
        request = self.server.requests[0]
        self.assertEqual(request["headers"]["authorization"], "Bearer explicit-key")

    def test_trailing_slash_base_url_is_normalized(self) -> None:
        client = PharosOne(base_url=self.base_url + "/", api_key="test-key")
        client.upsert_agent("a-1")
        self.assertEqual(self.server.requests[0]["path"], "/api/v1/upsert-agent")

    # --- upsert_agent -------------------------------------------------------

    def test_upsert_agent_full(self) -> None:
        agent = {
            "id": "b9a9...",
            "name": "Support Bot",
            "description": "Handles support",
            "goal": "Resolve tickets",
            "agent_context_json": {},
        }
        self.server.responses.append((200, json.dumps(agent).encode()))
        result = self.client.upsert_agent(
            "support-bot",
            name="Support Bot",
            description="Handles support",
            goal="Resolve tickets",
        )
        self.assertEqual(result, agent)
        request = self.server.requests[0]
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["path"], "/api/v1/upsert-agent")
        self.assertEqual(request["headers"]["authorization"], "Bearer test-key")
        self.assertEqual(request["headers"]["content-type"], "application/json")
        self.assertEqual(
            request["body"],
            {
                "agent_id": "support-bot",
                "name": "Support Bot",
                "description": "Handles support",
                "goal": "Resolve tickets",
            },
        )

    def test_upsert_agent_minimal_omits_absent_fields(self) -> None:
        self.client.upsert_agent("support-bot")
        self.assertEqual(self.server.requests[0]["body"], {"agent_id": "support-bot"})

    # --- send_message -------------------------------------------------------

    def test_send_message_full(self) -> None:
        response = {
            "status": "received",
            "dialog_id": "d-1",
            "message_index": 3,
            "created": True,
            "flagged": False,
            "fast_scan": "ok",
        }
        self.server.responses.append((202, json.dumps(response).encode()))
        tool_call = {
            "name": "search_kb",
            "label": "Search knowledge base",
            "status": "ok",
            "args_preview": '{"query": "refund"}',
            "result_preview": "3 articles",
        }
        ts = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone(timedelta(hours=2)))
        result = self.client.send_message(
            "support-bot",
            "sess-1",
            "tool",
            "",
            ts=ts,
            message_id="m-3",
            tool_call=tool_call,
            end_user={"external_id": "u-1", "locale": "en-US"},
        )
        self.assertEqual(result, response)
        request = self.server.requests[0]
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["path"], "/api/v1/send-message")
        self.assertEqual(request["headers"]["authorization"], "Bearer test-key")
        self.assertEqual(request["headers"]["content-type"], "application/json")
        self.assertEqual(
            request["body"],
            {
                "agent_id": "support-bot",
                "session_id": "sess-1",
                "role": "tool",
                "text": "",
                "ts": "2026-07-20T10:00:00Z",  # +02:00 input converted to UTC
                "message_id": "m-3",
                "tool_call": tool_call,
                "end_user": {"external_id": "u-1", "locale": "en-US"},
            },
        )

    def test_send_message_minimal_omits_optional_keys(self) -> None:
        self.client.send_message("support-bot", "sess-1", "user", "Hi!")
        self.assertEqual(
            self.server.requests[0]["body"],
            {"agent_id": "support-bot", "session_id": "sess-1", "role": "user", "text": "Hi!"},
        )

    def test_send_message_naive_datetime_is_utc(self) -> None:
        self.client.send_message("a", "s", "user", "x", ts=datetime(2026, 7, 20, 9, 58, 1))
        self.assertEqual(self.server.requests[0]["body"]["ts"], "2026-07-20T09:58:01Z")

    def test_send_message_string_ts_passthrough(self) -> None:
        self.client.send_message("a", "s", "user", "x", ts="2026-07-20T09:58:01Z")
        self.assertEqual(self.server.requests[0]["body"]["ts"], "2026-07-20T09:58:01Z")

    def test_send_message_returns_fast_verdict(self) -> None:
        response = {
            "status": "received",
            "dialog_id": "d-1",
            "message_index": 0,
            "created": True,
            "flagged": True,
            "fast_scan": "ok",
        }
        self.server.responses.append((202, json.dumps(response).encode()))
        result = self.client.send_message("a", "s", "user", "ATTACK")
        self.assertIs(result["flagged"], True)
        self.assertEqual(result["fast_scan"], "ok")

    def test_send_message_fast_scan_failed_passthrough(self) -> None:
        response = {
            "status": "received",
            "dialog_id": "d-1",
            "message_index": 0,
            "created": True,
            "flagged": False,
            "fast_scan": "failed",  # NO verdict: flagged=False must not be read as clean
        }
        self.server.responses.append((202, json.dumps(response).encode()))
        result = self.client.send_message("a", "s", "user", "x")
        self.assertEqual(result["fast_scan"], "failed")
        self.assertIs(result["flagged"], False)

    # --- send_dialog ----------------------------------------------------------

    def test_send_dialog_snapshot(self) -> None:
        response = {"status": "received", "dialog_id": "d-1", "flagged": False, "fast_scan": "ok"}
        self.server.responses.append((202, json.dumps(response).encode()))
        messages = [
            {
                "role": "user",
                "text": "hi",
                "ts": datetime(2026, 7, 20, 9, 58, tzinfo=timezone.utc),
                "message_id": "m-1",
            },
            {"role": "bot", "text": "hello"},
            {
                "role": "tool",
                "text": "",
                "tool_call": {"name": "lookup", "label": "Lookup order", "status": "pending"},
            },
        ]
        result = self.client.send_dialog(
            "support-bot", "sess-1", messages, end_user={"email": "u@example.com"}
        )
        self.assertEqual(result, response)
        request = self.server.requests[0]
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["path"], "/api/v1/send-dialog")
        self.assertEqual(request["headers"]["authorization"], "Bearer test-key")
        self.assertEqual(
            request["body"],
            {
                "agent_id": "support-bot",
                "session_id": "sess-1",
                "messages": [
                    {
                        "role": "user",
                        "text": "hi",
                        "ts": "2026-07-20T09:58:00Z",
                        "message_id": "m-1",
                    },
                    {"role": "bot", "text": "hello"},
                    {
                        "role": "tool",
                        "text": "",
                        "tool_call": {
                            "name": "lookup",
                            "label": "Lookup order",
                            "status": "pending",
                        },
                    },
                ],
                "end_user": {"email": "u@example.com"},
            },
        )
        # the caller's message dicts must not be mutated by ts serialization
        self.assertIsInstance(messages[0]["ts"], datetime)

    # --- get_analysis -----------------------------------------------------------

    _ANALYSIS_RESPONSE = {
        "dialog_id": "d-1",
        "status": "flagged",
        "analysis_status": "done",
        "flagged": True,
        "flag": {
            "category": "prompt-injection",
            "title": "Prompt injection attempt",
            "severity": "high",
            "summary": "The user tried to override the system prompt.",
            "mappings": [
                {"framework": "owasp-llm", "code": "LLM01", "name": "Prompt Injection", "detail": None}
            ],
        },
        "effectiveness": {"score": 42, "label": "poor", "summary": "Goal not reached."},
    }

    def test_get_analysis_by_dialog_id(self) -> None:
        self.server.responses.append((200, json.dumps(self._ANALYSIS_RESPONSE).encode()))
        result = self.client.get_analysis(dialog_id="d-1")
        self.assertEqual(result, self._ANALYSIS_RESPONSE)
        request = self.server.requests[0]
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["path"], "/api/v1/dialog-analysis")
        self.assertEqual(request["headers"]["authorization"], "Bearer test-key")
        self.assertEqual(request["body"], {"dialog_id": "d-1"})

    def test_get_analysis_by_agent_and_session(self) -> None:
        self.server.responses.append((200, json.dumps(self._ANALYSIS_RESPONSE).encode()))
        result = self.client.get_analysis(agent_id="support-bot", session_id="sess-42")
        self.assertEqual(result, self._ANALYSIS_RESPONSE)
        self.assertEqual(
            self.server.requests[0]["body"],
            {"agent_id": "support-bot", "session_id": "sess-42"},
        )

    def test_get_analysis_selector_xor_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "not both"):
            self.client.get_analysis(dialog_id="d-1", agent_id="a", session_id="s")
        with self.assertRaisesRegex(ValueError, "not both"):
            self.client.get_analysis(dialog_id="d-1", agent_id="a")
        with self.assertRaisesRegex(ValueError, "both agent_id and session_id"):
            self.client.get_analysis()
        with self.assertRaisesRegex(ValueError, "both agent_id and session_id"):
            self.client.get_analysis(agent_id="a")
        with self.assertRaisesRegex(ValueError, "both agent_id and session_id"):
            self.client.get_analysis(session_id="s")
        self.assertEqual(self.server.requests, [])  # rejected before any HTTP call

    def test_get_analysis_404_maps_to_error(self) -> None:
        payload = json.dumps(
            {"title": "Not Found", "status": 404, "detail": "dialog not found"}
        ).encode()
        self.server.responses.append((404, payload))
        with self.assertRaises(PharosOneError) as ctx:
            self.client.get_analysis(dialog_id="missing")
        self.assertEqual(ctx.exception.status, 404)
        self.assertEqual(ctx.exception.detail, "dialog not found")

    def test_get_analysis_uses_longer_timeout(self) -> None:
        # The server computes the analysis synchronously (up to ~75s), so the
        # default 15s per-request timeout must be stretched for this call only.
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.status = 200
            urlopen.return_value.__enter__.return_value.read.return_value = b"{}"
            self.client.get_analysis(dialog_id="d-1")
            self.assertEqual(urlopen.call_args.kwargs["timeout"], 90.0)
            self.client.get_analysis(dialog_id="d-1", timeout=5.0)
            self.assertEqual(urlopen.call_args.kwargs["timeout"], 5.0)
            self.client.send_message("a", "s", "user", "x")
            self.assertEqual(urlopen.call_args.kwargs["timeout"], 15.0)

    # --- error mapping ---------------------------------------------------------

    def test_error_maps_huma_detail(self) -> None:
        payload = json.dumps(
            {"title": "Conflict", "status": 409, "detail": "duplicate message_id"}
        ).encode()
        self.server.responses.append((409, payload))
        with self.assertRaises(PharosOneError) as ctx:
            self.client.send_message("a", "s", "user", "x")
        self.assertEqual(ctx.exception.status, 409)
        self.assertEqual(ctx.exception.detail, "duplicate message_id")
        self.assertEqual(str(ctx.exception), "HTTP 409: duplicate message_id")

    def test_error_falls_back_to_body_text(self) -> None:
        self.server.responses.append((500, b"upstream exploded"))
        with self.assertRaises(PharosOneError) as ctx:
            self.client.upsert_agent("a-1")
        self.assertEqual(ctx.exception.status, 500)
        self.assertEqual(ctx.exception.detail, "upstream exploded")

    def test_error_json_without_detail_falls_back_to_body_text(self) -> None:
        payload = b'{"error": "nope"}'
        self.server.responses.append((403, payload))
        with self.assertRaises(PharosOneError) as ctx:
            self.client.send_dialog("a", "s", [])
        self.assertEqual(ctx.exception.status, 403)
        self.assertEqual(ctx.exception.detail, payload.decode())


if __name__ == "__main__":
    unittest.main()
