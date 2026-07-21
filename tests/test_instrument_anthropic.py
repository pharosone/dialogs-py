"""wrap_anthropic tests against hand-built Anthropic-shaped fakes (no anthropic dep)."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace as NS

from pharosone_dialogs.instrument import wrap_anthropic
from pharosone_dialogs.instrument import _worker

try:
    from .test_instrument_openai import make_recording_pharos
except ImportError:  # unittest discover imports test modules as top-level
    from test_instrument_openai import make_recording_pharos


def text_message(text):
    return NS(
        id="msg_1",
        role="assistant",
        content=[NS(type="text", text=text)],
        stop_reason="end_turn",
    )


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeAsyncMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeAnthropic:
    def __init__(self, responses, messages_cls=FakeMessages):
        self.messages = messages_cls(responses)
        self.api_key = "sk-ant-fake"


def stream_events():
    """messages.create(stream=True) event sequence: text block + tool_use block."""
    return [
        NS(type="message_start", message=NS(role="assistant", content=[])),
        NS(type="content_block_start", index=0, content_block=NS(type="text", text="")),
        NS(type="content_block_delta", index=0, delta=NS(type="text_delta", text="Check")),
        NS(type="content_block_delta", index=0, delta=NS(type="text_delta", text="ing…")),
        NS(type="content_block_stop", index=0),
        NS(
            type="content_block_start",
            index=1,
            content_block=NS(type="tool_use", id="toolu_9", name="get_weather", text=None),
        ),
        NS(
            type="content_block_delta",
            index=1,
            delta=NS(type="input_json_delta", partial_json='{"city": '),
        ),
        NS(
            type="content_block_delta",
            index=1,
            delta=NS(type="input_json_delta", partial_json='"Paris"}'),
        ),
        NS(type="content_block_stop", index=1),
        NS(type="message_delta", delta=NS(stop_reason="tool_use")),
        NS(type="message_stop"),
    ]


async def async_events(events):
    for event in events:
        yield event


class WrapAnthropicTestCase(unittest.TestCase):
    def setUp(self) -> None:
        _worker._synced_agents.clear()

    def wrap(self, fake, pharos, **kwargs):
        wrapped = wrap_anthropic(fake, pharos=pharos, agent_id="claude-bot", **kwargs)
        self.addCleanup(wrapped.pharos_instrumentation.close, 5.0)
        return wrapped

    def test_snapshot_system_skipped_and_passthrough(self) -> None:
        pharos, calls = make_recording_pharos()
        response = text_message("Hello!")
        fake = FakeAnthropic([response])
        client = self.wrap(fake, pharos, session_id="sess-1")

        result = client.messages.create(
            model="claude-sonnet-4-5",
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
        )

        self.assertIs(result, response)
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(
            calls["send_dialog"],
            [
                {
                    "agent_id": "claude-bot",
                    "session_id": "sess-1",
                    "messages": [
                        {"role": "user", "text": "Hi"},
                        {"role": "bot", "text": "Hello!"},
                    ],
                }
            ],
        )
        self.assertEqual(fake.messages.calls[0]["system"], "You are helpful.")

    def test_tool_use_round_trip(self) -> None:
        pharos, calls = make_recording_pharos()
        client = self.wrap(FakeAnthropic([text_message("It is sunny.")]), pharos, session_id="s")
        client.messages.create(
            system=[{"type": "text", "text": "sys"}],
            messages=[
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": [
                        NS(type="text", text="Let me check."),
                        NS(type="tool_use", id="toolu_1", name="get_weather", input={"city": "Paris"}),
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "sunny, 25C",
                        }
                    ],
                },
            ],
        )
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(
            calls["send_dialog"][0]["messages"],
            [
                {"role": "user", "text": "Weather in Paris?"},
                {"role": "bot", "text": "Let me check."},
                {
                    "role": "tool",
                    "text": "",
                    "message_id": "toolu_1",
                    "tool_call": {
                        "name": "get_weather",
                        "label": "get_weather",
                        "status": "ok",
                        "args_preview": '{"city": "Paris"}',
                        "result_preview": "sunny, 25C",
                    },
                },
                {"role": "bot", "text": "It is sunny."},
            ],
        )

    def test_tool_result_is_error_and_block_list_content(self) -> None:
        pharos, calls = make_recording_pharos()
        client = self.wrap(FakeAnthropic([text_message("ok")]), pharos, session_id="s")
        client.messages.create(
            messages=[
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": [NS(type="tool_use", id="toolu_2", name="calc", input={"x": 1})],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_2",
                            "content": [{"type": "text", "text": "division by zero"}],
                            "is_error": True,
                        }
                    ],
                },
            ]
        )
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        entry = calls["send_dialog"][0]["messages"][1]
        self.assertEqual(entry["tool_call"]["status"], "error")
        self.assertEqual(entry["tool_call"]["result_preview"], "division by zero")

    def test_streaming_accumulates_text_and_tool_use(self) -> None:
        pharos, calls = make_recording_pharos()
        fake = FakeAnthropic([iter(stream_events())])
        client = self.wrap(fake, pharos, session_id="s")
        events = list(
            client.messages.create(
                messages=[{"role": "user", "content": "Weather?"}], stream=True
            )
        )
        self.assertEqual(len(events), 11)  # all events passed through
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(
            calls["send_dialog"][0]["messages"],
            [
                {"role": "user", "text": "Weather?"},
                {"role": "bot", "text": "Checking…"},
                {
                    "role": "tool",
                    "text": "",
                    "message_id": "toolu_9",
                    "tool_call": {
                        "name": "get_weather",
                        "label": "get_weather",
                        "status": "pending",
                        "args_preview": '{"city": "Paris"}',
                    },
                },
            ],
        )

    def test_async_create_and_stream(self) -> None:
        pharos, calls = make_recording_pharos()
        fake = FakeAnthropic(
            [text_message("Hello!"), async_events(stream_events())],
            messages_cls=FakeAsyncMessages,
        )
        client = self.wrap(fake, pharos, session_id="s")

        async def main():
            response = await client.messages.create(
                messages=[{"role": "user", "content": "Hi"}]
            )
            stream = await client.messages.create(
                messages=[{"role": "user", "content": "Weather?"}], stream=True
            )
            events = [event async for event in stream]
            return response, events

        response, events = asyncio.run(main())
        self.assertEqual(response.content[0].text, "Hello!")
        self.assertEqual(len(events), 11)
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(len(calls["send_dialog"]), 2)
        self.assertEqual(
            calls["send_dialog"][1]["messages"][1], {"role": "bot", "text": "Checking…"}
        )

    def test_session_kwarg_stripped_before_provider(self) -> None:
        pharos, calls = make_recording_pharos()
        fake = FakeAnthropic([text_message("ok")])
        client = self.wrap(fake, pharos)
        client.messages.create(
            messages=[{"role": "user", "content": "Hi"}], pharos_session_id="explicit-9"
        )
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(calls["send_dialog"][0]["session_id"], "explicit-9")
        self.assertNotIn("pharos_session_id", fake.messages.calls[0])

    def test_sync_agent_uses_system_param(self) -> None:
        pharos, calls = make_recording_pharos()
        client = self.wrap(
            FakeAnthropic([text_message("ok")]), pharos, session_id="s", sync_agent=True
        )
        client.messages.create(
            system=[{"type": "text", "text": "You are Claude-bot."}],
            messages=[{"role": "user", "content": "Hi"}],
        )
        self.assertTrue(client.pharos_instrumentation.drain(5.0))
        self.assertEqual(
            calls["upsert_agent"],
            [{"agent_id": "claude-bot", "description": "You are Claude-bot."}],
        )


if __name__ == "__main__":
    unittest.main()
