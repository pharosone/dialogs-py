"""LangChain + LiteLLM integration tests with hand-built payload shapes.

No real langchain/litellm dependency: the LangChain tests inject a fake
`langchain_core.callbacks.BaseCallbackHandler` into sys.modules; the LiteLLM
callback is plain duck-typing and needs nothing at all.
"""

from __future__ import annotations

import hashlib
import sys
import types
import unittest
from types import SimpleNamespace as NS

from pharosone_dialogs.instrument import pharos_session
from pharosone_dialogs.integrations.langchain import PharosCallbackHandler
from pharosone_dialogs.integrations.litellm import pharos_litellm_callback

try:
    from .test_instrument_openai import make_recording_pharos
except ImportError:  # unittest discover imports test modules as top-level
    from test_instrument_openai import make_recording_pharos


class FakeBaseCallbackHandler:
    """Stands in for langchain_core.callbacks.BaseCallbackHandler."""

    raise_error = False
    run_inline = False


def install_fake_langchain(testcase: unittest.TestCase) -> type:
    saved = {
        name: sys.modules.get(name) for name in ("langchain_core", "langchain_core.callbacks")
    }

    def restore() -> None:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    core = types.ModuleType("langchain_core")
    callbacks = types.ModuleType("langchain_core.callbacks")
    callbacks.BaseCallbackHandler = FakeBaseCallbackHandler
    core.callbacks = callbacks
    sys.modules["langchain_core"] = core
    sys.modules["langchain_core.callbacks"] = callbacks
    testcase.addCleanup(restore)
    return FakeBaseCallbackHandler


class LangChainHandlerTestCase(unittest.TestCase):
    def make_handler(self, pharos, **kwargs):
        base = install_fake_langchain(self)
        handler = PharosCallbackHandler(pharos, "lc-bot", **kwargs)
        self.assertIsInstance(handler, base)
        self.addCleanup(handler.close, 5.0)
        return handler

    def test_missing_langchain_raises_clear_import_error(self) -> None:
        saved = {
            name: sys.modules.get(name)
            for name in ("langchain_core", "langchain_core.callbacks")
        }

        def restore() -> None:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.addCleanup(restore)
        sys.modules["langchain_core"] = None  # force ImportError on import
        sys.modules.pop("langchain_core.callbacks", None)
        pharos, _ = make_recording_pharos()
        with self.assertRaisesRegex(ImportError, "langchain-core"):
            PharosCallbackHandler(pharos, "lc-bot")

    def test_chat_run_snapshot(self) -> None:
        pharos, calls = make_recording_pharos()
        handler = self.make_handler(pharos, session_id="sess-lc")
        handler.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [
                [
                    NS(type="system", content="You are terse."),
                    NS(type="human", content="Hi"),
                ]
            ],
            run_id="run-1",
        )
        handler.on_llm_end(
            NS(
                generations=[
                    [NS(message=NS(type="ai", content="Hello!", tool_calls=[]), text="Hello!")]
                ]
            ),
            run_id="run-1",
        )
        self.assertTrue(handler.drain(5.0))
        self.assertEqual(
            calls["send_dialog"],
            [
                {
                    "agent_id": "lc-bot",
                    "session_id": "sess-lc",
                    "messages": [
                        {"role": "user", "text": "Hi"},
                        {"role": "bot", "text": "Hello!"},
                    ],
                }
            ],
        )

    def test_tool_call_round_trip_across_runs(self) -> None:
        pharos, calls = make_recording_pharos()
        handler = self.make_handler(pharos, session_id="sess-lc")
        ai_with_tool = NS(
            type="ai",
            content="",
            tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}],
        )
        # Run 1: model asks for a tool.
        handler.on_chat_model_start(
            {}, [[NS(type="human", content="Weather in Paris?")]], run_id="run-1"
        )
        handler.on_llm_end(
            NS(generations=[[NS(message=ai_with_tool, text="")]]), run_id="run-1"
        )
        # Run 2: history now carries the tool result; model answers.
        handler.on_chat_model_start(
            {},
            [
                [
                    NS(type="human", content="Weather in Paris?"),
                    ai_with_tool,
                    NS(type="tool", content="sunny, 25C", tool_call_id="call_1", name=None, status="success"),
                ]
            ],
            run_id="run-2",
        )
        handler.on_llm_end(
            NS(generations=[[NS(message=NS(type="ai", content="It is sunny.", tool_calls=[]), text="It is sunny.")]]),
            run_id="run-2",
        )
        self.assertTrue(handler.drain(5.0))
        first, second = calls["send_dialog"]
        self.assertEqual(
            first["messages"][1],
            {
                "role": "tool",
                "text": "",
                "message_id": "call_1",
                "tool_call": {
                    "name": "get_weather",
                    "label": "get_weather",
                    "status": "pending",
                    "args_preview": '{"city": "Paris"}',
                },
            },
        )
        self.assertEqual(
            second["messages"],
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
                        "args_preview": '{"city": "Paris"}',
                        "result_preview": "sunny, 25C",
                    },
                },
                {"role": "bot", "text": "It is sunny."},
            ],
        )

    def test_on_tool_start_end_appends_and_resolves(self) -> None:
        pharos, calls = make_recording_pharos()
        handler = self.make_handler(pharos, session_id="sess-lc")
        handler.on_chat_model_start({}, [[NS(type="human", content="Hi")]], run_id="run-1")
        handler.on_llm_end(
            NS(generations=[[NS(message=NS(type="ai", content="Hello!", tool_calls=[]), text="Hello!")]]),
            run_id="run-1",
        )
        handler.on_tool_start({"name": "search"}, '{"q": "docs"}', run_id="tool-1")
        handler.on_tool_end("3 results", run_id="tool-1")
        self.assertTrue(handler.drain(5.0))
        self.assertEqual(len(calls["send_dialog"]), 3)
        final = calls["send_dialog"][2]["messages"]
        self.assertEqual(
            final[2],
            {
                "role": "tool",
                "text": "",
                "message_id": "lc-tool-tool-1",
                "tool_call": {
                    "name": "search",
                    "label": "search",
                    "status": "ok",
                    "args_preview": '{"q": "docs"}',
                    "result_preview": "3 results",
                },
            },
        )
        # the pending version was flushed first
        pending = calls["send_dialog"][1]["messages"][2]["tool_call"]
        self.assertEqual(pending["status"], "pending")
        self.assertNotIn("result_preview", pending)

    def test_session_falls_back_to_first_user_hash(self) -> None:
        pharos, calls = make_recording_pharos()
        handler = self.make_handler(pharos)  # no session_id
        handler.on_chat_model_start({}, [[NS(type="human", content="Hi")]], run_id="r1")
        handler.on_llm_end(
            NS(generations=[[NS(message=NS(type="ai", content="Yo", tool_calls=[]), text="Yo")]]),
            run_id="r1",
        )
        self.assertTrue(handler.drain(5.0))
        self.assertEqual(
            calls["send_dialog"][0]["session_id"], hashlib.sha256(b"Hi").hexdigest()
        )


class LiteLLMCallbackTestCase(unittest.TestCase):
    def make_callback(self, pharos, **kwargs):
        callback = pharos_litellm_callback(pharos, "llm-bot", **kwargs)
        self.addCleanup(callback.instrumentation.close, 5.0)
        return callback

    def test_success_callback_snapshot_with_metadata_session(self) -> None:
        pharos, calls = make_recording_pharos()
        callback = self.make_callback(pharos)
        callback(
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "Hi"},
                ],
                "litellm_params": {"metadata": {"pharos_session_id": "sess-9"}},
            },
            {"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]},
            0.0,
            1.0,
        )
        self.assertTrue(callback.instrumentation.drain(5.0))
        self.assertEqual(
            calls["send_dialog"],
            [
                {
                    "agent_id": "llm-bot",
                    "session_id": "sess-9",
                    "messages": [
                        {"role": "user", "text": "Hi"},
                        {"role": "bot", "text": "Hello!"},
                    ],
                }
            ],
        )

    def test_object_shaped_response_and_tool_calls(self) -> None:
        pharos, calls = make_recording_pharos()
        callback = self.make_callback(pharos, session_id="s")
        response = NS(
            choices=[
                NS(
                    message=NS(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            NS(id="call_7", function=NS(name="lookup", arguments='{"q":"x"}'))
                        ],
                    )
                )
            ]
        )
        callback({"messages": [{"role": "user", "content": "q"}]}, response, 0.0, 1.0)
        self.assertTrue(callback.instrumentation.drain(5.0))
        self.assertEqual(
            calls["send_dialog"][0]["messages"][1],
            {
                "role": "tool",
                "text": "",
                "message_id": "call_7",
                "tool_call": {
                    "name": "lookup",
                    "label": "lookup",
                    "status": "pending",
                    "args_preview": '{"q":"x"}',
                },
            },
        )

    def test_session_precedence_context_then_hash(self) -> None:
        pharos, calls = make_recording_pharos()
        callback = self.make_callback(pharos)
        with pharos_session("ctx-7"):
            callback({"messages": [{"role": "user", "content": "Hi"}]}, None, 0.0, 1.0)
        callback({"messages": [{"role": "user", "content": "Hi"}]}, None, 0.0, 1.0)
        self.assertTrue(callback.instrumentation.drain(5.0))
        sessions = [call["session_id"] for call in calls["send_dialog"]]
        self.assertEqual(sessions, ["ctx-7", hashlib.sha256(b"Hi").hexdigest()])

    def test_callback_never_raises(self) -> None:
        pharos, calls = make_recording_pharos(fail=True)
        callback = self.make_callback(pharos, session_id="s")
        with self.assertLogs("pharosone_dialogs.instrument", level="WARNING"):
            callback({"messages": [{"role": "user", "content": "Hi"}]}, None, 0.0, 1.0)
            callback(None)  # nonsense payload: swallowed, logged at worst
            self.assertTrue(callback.instrumentation.drain(5.0))
        self.assertEqual(len(calls["send_dialog"]), 2)


if __name__ == "__main__":
    unittest.main()
