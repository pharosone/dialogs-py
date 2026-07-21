"""wrap_openai tests against hand-built OpenAI-shaped fakes (no openai dep)."""

from __future__ import annotations

import asyncio
import hashlib
import unittest
from types import SimpleNamespace as NS

from pharosone_dialogs import PharosOne, PharosOneError
from pharosone_dialogs.instrument import pharos_session, wrap_openai
from pharosone_dialogs.instrument import _worker


def make_recording_pharos(fail: bool = False):
    """A real PharosOne with monkeypatched send_dialog/upsert_agent."""
    client = PharosOne(base_url="http://pharos.invalid", api_key="test-key")
    calls = {"send_dialog": [], "upsert_agent": []}

    def send_dialog(agent_id, session_id, messages, *, end_user=None):
        calls["send_dialog"].append(
            {"agent_id": agent_id, "session_id": session_id, "messages": messages}
        )
        if fail:
            raise PharosOneError(500, "boom")
        return {
            "status": "received",
            "dialog_id": "d-1",
            "flagged": False,
            "fast_scan": "ok",
        }

    def upsert_agent(agent_id, **kwargs):
        calls["upsert_agent"].append({"agent_id": agent_id, **kwargs})
        return {"id": "a-1", "name": agent_id}

    client.send_dialog = send_dialog
    client.upsert_agent = upsert_agent
    return client, calls


def text_response(text):
    return NS(
        id="chatcmpl-1",
        choices=[
            NS(
                index=0,
                message=NS(role="assistant", content=text, tool_calls=None),
                finish_reason="stop",
            )
        ],
    )


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeAsyncCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeOpenAI:
    """Duck-typed OpenAI-shaped client (sync or async completions)."""

    def __init__(self, responses, completions_cls=FakeCompletions):
        self.chat = NS(completions=completions_cls(responses))
        self.api_key = "sk-fake"
        self.models = NS(list=lambda: ["m-1"])

    def ping(self):
        return "pong"


async def async_chunks(chunks):
    for chunk in chunks:
        yield chunk


class WrapOpenAITestCase(unittest.TestCase):
    def setUp(self) -> None:
        _worker._synced_agents.clear()

    def wrap(self, fake, pharos, **kwargs):
        wrapped = wrap_openai(fake, pharos=pharos, agent_id="support-bot", **kwargs)
        self.addCleanup(wrapped.pharos_instrumentation.close, 5.0)
        return wrapped

    # -- basic snapshot -------------------------------------------------------

    def test_snapshot_and_passthrough(self) -> None:
        pharos, calls = make_recording_pharos()
        response = text_response("Hello!")
        fake = FakeOpenAI([response])
        client = self.wrap(fake, pharos, session_id="sess-1")

        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Hi"},
            ],
        )

        self.assertIs(result, response)  # exact provider response, unchanged
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(
            calls["send_dialog"],
            [
                {
                    "agent_id": "support-bot",
                    "session_id": "sess-1",
                    "messages": [
                        {"role": "user", "text": "Hi"},
                        {"role": "bot", "text": "Hello!"},
                    ],
                }
            ],
        )
        # provider saw its kwargs untouched (and no pharos_session_id)
        sent = fake.chat.completions.calls[0]
        self.assertEqual(sent["model"], "gpt-4o-mini")
        self.assertNotIn("pharos_session_id", sent)

    def test_tool_call_round_trip(self) -> None:
        pharos, calls = make_recording_pharos()
        fake = FakeOpenAI([text_response("It is sunny.")])
        client = self.wrap(fake, pharos, session_id="sess-2")

        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Paris"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 25C"},
            ],
        )
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(
            calls["send_dialog"][0]["messages"],
            [
                {"role": "user", "text": "Weather in Paris?"},
                {
                    "role": "tool",
                    "text": "",
                    "message_id": "call_1",
                    "tool_call": {
                        "name": "get_weather",
                        "label": "get_weather",
                        "status": "ok",
                        "args_preview": '{"city":"Paris"}',
                        "result_preview": "sunny, 25C",
                    },
                },
                {"role": "bot", "text": "It is sunny."},
            ],
        )

    def test_tool_result_error_heuristic_and_explicit_flag(self) -> None:
        pharos, calls = make_recording_pharos()
        fake = FakeOpenAI([text_response("a"), text_response("b")])
        client = self.wrap(fake, pharos, session_id="s")
        assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "function": {"name": "t", "arguments": "{}"}}
            ],
        }
        client.chat.completions.create(
            messages=[
                {"role": "user", "content": "q"},
                assistant,
                {"role": "tool", "tool_call_id": "c1", "content": "Error: timed out"},
            ]
        )
        client.chat.completions.create(
            messages=[
                {"role": "user", "content": "q"},
                assistant,
                {"role": "tool", "tool_call_id": "c1", "content": "fine", "is_error": True},
            ]
        )
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        first, second = calls["send_dialog"]
        self.assertEqual(first["messages"][1]["tool_call"]["status"], "error")
        self.assertEqual(second["messages"][1]["tool_call"]["status"], "error")
        self.assertEqual(second["messages"][1]["tool_call"]["result_preview"], "fine")

    def test_response_with_tool_calls_is_pending(self) -> None:
        pharos, calls = make_recording_pharos()
        response = NS(
            choices=[
                NS(
                    message=NS(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            NS(
                                id="call_9",
                                type="function",
                                function=NS(name="lookup", arguments='{"q":"x"}'),
                            )
                        ],
                    )
                )
            ]
        )
        client = self.wrap(FakeOpenAI([response]), pharos, session_id="s")
        client.chat.completions.create(messages=[{"role": "user", "content": "q"}])
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(
            calls["send_dialog"][0]["messages"][1],
            {
                "role": "tool",
                "text": "",
                "message_id": "call_9",
                "tool_call": {
                    "name": "lookup",
                    "label": "lookup",
                    "status": "pending",
                    "args_preview": '{"q":"x"}',
                },
            },
        )

    def test_vision_parts_joined_with_non_text_marker(self) -> None:
        pharos, calls = make_recording_pharos()
        client = self.wrap(FakeOpenAI([text_response("ok")]), pharos, session_id="s")
        client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look at this"},
                        {"type": "image_url", "image_url": {"url": "data:..."}},
                    ],
                }
            ]
        )
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(
            calls["send_dialog"][0]["messages"][0],
            {"role": "user", "text": "look at this\n[non-text content]"},
        )

    # -- caps and redaction ------------------------------------------------------

    def test_text_and_preview_caps(self) -> None:
        pharos, calls = make_recording_pharos()
        client = self.wrap(FakeOpenAI([text_response("ok")]), pharos, session_id="s")
        client.chat.completions.create(
            messages=[
                {"role": "user", "content": "x" * 25_000},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "t", "arguments": "a" * 2_000}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r" * 2_000},
            ]
        )
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        messages = calls["send_dialog"][0]["messages"]
        self.assertEqual(len(messages[0]["text"]), 19_999)
        self.assertTrue(messages[0]["text"].endswith("…"))
        self.assertEqual(len(messages[1]["tool_call"]["args_preview"]), 500)
        self.assertEqual(len(messages[1]["tool_call"]["result_preview"]), 500)

    def test_redact_applies_to_previews_only(self) -> None:
        pharos, calls = make_recording_pharos()
        client = self.wrap(
            FakeOpenAI([text_response("ok")]),
            pharos,
            session_id="s",
            redact=lambda text: "[REDACTED]",
        )
        client.chat.completions.create(
            messages=[
                {"role": "user", "content": "secret question"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "t", "arguments": '{"k":"v"}'}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "secret result"},
            ]
        )
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        messages = calls["send_dialog"][0]["messages"]
        self.assertEqual(messages[0]["text"], "secret question")  # text untouched
        self.assertEqual(messages[1]["tool_call"]["args_preview"], "[REDACTED]")
        self.assertEqual(messages[1]["tool_call"]["result_preview"], "[REDACTED]")

    # -- session binding -----------------------------------------------------------

    def test_session_precedence_and_kwarg_stripping(self) -> None:
        pharos, calls = make_recording_pharos()
        fake = FakeOpenAI([text_response("a"), text_response("b"), text_response("c")])
        client = self.wrap(fake, pharos)  # no wrap-time session
        messages = [{"role": "user", "content": "Hi"}]

        with pharos_session("ctx-1"):
            client.chat.completions.create(messages=messages)  # -> ctx-1
            client.chat.completions.create(
                messages=messages, pharos_session_id="explicit-1"
            )  # -> explicit-1
        client.chat.completions.create(messages=messages)  # -> sha256 fallback

        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        sessions = [call["session_id"] for call in calls["send_dialog"]]
        fallback = hashlib.sha256(b"Hi").hexdigest()
        self.assertEqual(sessions, ["ctx-1", "explicit-1", fallback])
        for sent in fake.chat.completions.calls:
            self.assertNotIn("pharos_session_id", sent)  # never reaches provider

    def test_context_scope_wins_over_wrap_time_session(self) -> None:
        pharos, calls = make_recording_pharos()
        client = self.wrap(FakeOpenAI([text_response("a")]), pharos, session_id="wrap-1")
        with pharos_session("ctx-2"):
            client.chat.completions.create(messages=[{"role": "user", "content": "Hi"}])
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(calls["send_dialog"][0]["session_id"], "ctx-2")

    # -- streaming -----------------------------------------------------------------

    def test_streaming_accumulates_and_flushes(self) -> None:
        pharos, calls = make_recording_pharos()
        chunks = [
            NS(choices=[NS(delta=NS(content="Hel", tool_calls=None), finish_reason=None)]),
            NS(choices=[NS(delta=NS(content="lo", tool_calls=None), finish_reason=None)]),
            NS(choices=[NS(delta=NS(content=None, tool_calls=None), finish_reason="stop")]),
        ]
        fake = FakeOpenAI([iter(chunks)])
        client = self.wrap(fake, pharos, session_id="s")
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}], stream=True
        )
        seen = list(stream)
        self.assertEqual(seen, chunks)  # chunks pass through unchanged
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(
            calls["send_dialog"][0]["messages"],
            [{"role": "user", "text": "Hi"}, {"role": "bot", "text": "Hello"}],
        )

    def test_streaming_tool_call_deltas(self) -> None:
        pharos, calls = make_recording_pharos()
        chunks = [
            NS(
                choices=[
                    NS(
                        delta=NS(
                            content=None,
                            tool_calls=[
                                NS(index=0, id="call_5", function=NS(name="lookup", arguments=""))
                            ],
                        ),
                        finish_reason=None,
                    )
                ]
            ),
            NS(
                choices=[
                    NS(
                        delta=NS(
                            content=None,
                            tool_calls=[
                                NS(index=0, id=None, function=NS(name=None, arguments='{"q":'))
                            ],
                        ),
                        finish_reason=None,
                    )
                ]
            ),
            NS(
                choices=[
                    NS(
                        delta=NS(
                            content=None,
                            tool_calls=[
                                NS(index=0, id=None, function=NS(name=None, arguments='"x"}'))
                            ],
                        ),
                        finish_reason=None,
                    )
                ]
            ),
        ]
        client = self.wrap(FakeOpenAI([iter(chunks)]), pharos, session_id="s")
        for _ in client.chat.completions.create(
            messages=[{"role": "user", "content": "q"}], stream=True
        ):
            pass
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(
            calls["send_dialog"][0]["messages"][1],
            {
                "role": "tool",
                "text": "",
                "message_id": "call_5",
                "tool_call": {
                    "name": "lookup",
                    "label": "lookup",
                    "status": "pending",
                    "args_preview": '{"q":"x"}',
                },
            },
        )

    def test_stream_closed_early_flushes_partial(self) -> None:
        pharos, calls = make_recording_pharos()
        chunks = [
            NS(choices=[NS(delta=NS(content="par", tool_calls=None))]),
            NS(choices=[NS(delta=NS(content="tial", tool_calls=None))]),
        ]
        client = self.wrap(FakeOpenAI([iter(chunks)]), pharos, session_id="s")
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}], stream=True
        )
        next(stream)
        stream.close()  # never fully consumed: flush what was seen
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(
            calls["send_dialog"][0]["messages"][1], {"role": "bot", "text": "par"}
        )

    # -- async client ------------------------------------------------------------

    def test_async_create_and_stream(self) -> None:
        pharos, calls = make_recording_pharos()
        chunks = [
            NS(choices=[NS(delta=NS(content="Hey", tool_calls=None))]),
            NS(choices=[NS(delta=NS(content="!", tool_calls=None))]),
        ]
        fake = FakeOpenAI(
            [text_response("Hello!"), async_chunks(chunks)],
            completions_cls=FakeAsyncCompletions,
        )
        client = self.wrap(fake, pharos, session_id="s")

        async def main():
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}]
            )
            stream = await client.chat.completions.create(
                messages=[{"role": "user", "content": "again"}], stream=True
            )
            seen = [chunk async for chunk in stream]
            return response, seen

        response, seen = asyncio.run(main())
        self.assertEqual(response.choices[0].message.content, "Hello!")
        self.assertEqual(seen, chunks)
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(len(calls["send_dialog"]), 2)
        self.assertEqual(
            calls["send_dialog"][1]["messages"],
            [{"role": "user", "text": "again"}, {"role": "bot", "text": "Hey!"}],
        )

    # -- resilience ---------------------------------------------------------------

    def test_pharos_failure_never_propagates(self) -> None:
        pharos, calls = make_recording_pharos(fail=True)
        client = self.wrap(FakeOpenAI([text_response("ok")]), pharos, session_id="s")
        with self.assertLogs("pharosone_dialogs.instrument", level="WARNING"):
            result = client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}]
            )
            self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(result.choices[0].message.content, "ok")
        self.assertEqual(len(calls["send_dialog"]), 1)  # attempted, failed, swallowed

    # -- sync_agent / on_result ------------------------------------------------------

    def test_sync_agent_upserts_once_per_process(self) -> None:
        pharos, calls = make_recording_pharos()
        fake = FakeOpenAI([text_response("a"), text_response("b")])
        client = self.wrap(fake, pharos, session_id="s", sync_agent=True)
        messages = [
            {"role": "system", "content": "  You are a support bot.  "},
            {"role": "user", "content": "Hi"},
        ]
        client.chat.completions.create(messages=messages)
        client.chat.completions.create(messages=messages)
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(
            calls["upsert_agent"],
            [{"agent_id": "support-bot", "description": "You are a support bot."}],
        )

    def test_sync_agent_off_by_default(self) -> None:
        pharos, calls = make_recording_pharos()
        client = self.wrap(FakeOpenAI([text_response("a")]), pharos, session_id="s")
        client.chat.completions.create(
            messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "Hi"}]
        )
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(calls["upsert_agent"], [])

    def test_on_result_receives_send_dialog_result(self) -> None:
        pharos, _ = make_recording_pharos()
        results = []
        client = self.wrap(
            FakeOpenAI([text_response("a")]), pharos, session_id="s", on_result=results.append
        )
        client.chat.completions.create(messages=[{"role": "user", "content": "Hi"}])
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["fast_scan"], "ok")
        self.assertIs(results[0]["flagged"], False)

    # -- proxy transparency ------------------------------------------------------------

    def test_proxy_delegates_everything_else(self) -> None:
        pharos, _ = make_recording_pharos()
        fake = FakeOpenAI([])
        client = self.wrap(fake, pharos, session_id="s")
        self.assertEqual(client.api_key, "sk-fake")
        self.assertEqual(client.ping(), "pong")
        self.assertEqual(client.models.list(), ["m-1"])
        client.api_key = "sk-new"  # writes go to the wrapped client
        self.assertEqual(fake.api_key, "sk-new")
        self.assertIn("pharos-instrumented", repr(client))


if __name__ == "__main__":
    unittest.main()
