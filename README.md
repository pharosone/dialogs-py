# pharosone-dialogs

Zero-dependency Python client for the PharosOne dialog ingest API: create or
update agents, stream dialog messages one by one, or send full dialog
snapshots. Python 3.10+, stdlib only (`urllib.request`).

## Install

```bash
pip install pharosone-dialogs
```

## Quickstart

```python
from pharosone_dialogs import PharosOne

client = PharosOne(
    base_url="https://pharosone.example.com",  # or env PHAROSONE_BASE_URL
    api_key="sk-...",                          # or env PHAROSONE_API_KEY
)

# Create the agent ahead of time (idempotent — safe to call on every startup).
client.upsert_agent(
    "support-bot",
    name="Support Bot",
    description="Customer support assistant for the web store.",
    goal="Resolve the customer's issue or hand off to a human.",
)

# Stream each turn as it happens.
client.send_message("support-bot", "sess-42", "user", "Where is my order?")
result = client.send_message("support-bot", "sess-42", "bot", "Let me check that for you.")
print(result)  # {"status": "received", "dialog_id": "...", "message_index": 1, "created": True}
```

Explicit constructor arguments win over the `PHAROSONE_BASE_URL` /
`PHAROSONE_API_KEY` environment variables; if neither is set, the constructor
raises `ValueError`. The request timeout defaults to 15 seconds
(`PharosOne(..., timeout=30.0)` to change it).

## Check the verdict

Every `send_message` / `send_dialog` response carries a synchronous fast
verdict: `flagged` (bool) and `fast_scan` (`"ok"` or `"failed"`).
**`fast_scan == "failed"` means the scan did not run — there is NO verdict.**
Never treat `flagged == False` as clean in that case.

For the detailed finding (flag category, severity, framework mappings,
effectiveness score), call `get_analysis`. Select the dialog either by
`dialog_id` or by `agent_id` + `session_id` — exactly one form, or the client
raises `ValueError` before making a request:

```python
result = client.send_message("support-bot", "sess-42", "bot", "Sure, here is how...")

if result["fast_scan"] == "failed":
    pass  # no verdict — retry / alert, but do NOT assume clean
elif result["flagged"]:
    analysis = client.get_analysis(dialog_id=result["dialog_id"])
    # equivalent: client.get_analysis(agent_id="support-bot", session_id="sess-42")
    print(analysis["analysis_status"])  # "pending" | "running" | "done" | "failed"
    if analysis["flag"]:
        flag = analysis["flag"]  # {"category", "title", "severity", "summary", "mappings"}
        print(flag["severity"], flag["title"])
    if analysis["effectiveness"]:
        print(analysis["effectiveness"]["score"])  # 1-100
```

`get_analysis` is synchronous on the server side: it computes the deep
analysis while the request blocks, up to ~75 seconds in the worst case. The
client therefore uses `max(timeout, 90.0)` seconds for this call instead of
the constructor timeout; pass `get_analysis(..., timeout=120.0)` to override
per call. If the analysis still is not `"done"` when the server's wait budget
runs out, the response returns the current state (`flag` / `effectiveness`
may be `None`) — calling again retries, including after a `"failed"` run.

## Tool calls

Tool activity is a first-class message (`role="tool"`) with a `tool_call`
payload, so the analysis pipeline sees what your bot actually did:

```python
client.send_message(
    "support-bot",
    "sess-42",
    "tool",
    "",
    message_id="sess-42-tool-7",
    tool_call={
        "name": "order_lookup",
        "label": "Look up order",
        "status": "ok",  # ok | denied | error | pending
        "args_preview": '{"order_id": "A-1001"}',
        "result_preview": "shipped 2026-07-18, ETA 2026-07-21",
    },
)
```

Tip: send the tool message with `status="pending"` when the call starts, then
re-send the **same** `message_id` with the final status and `result_preview` —
the row is updated in place instead of appended.

## send_message (per turn) vs send_dialog (snapshot)

**Prefer `send_message`** — call it once per turn from your bot loop:

- messages appear in the cabinet live, while the dialog is still running;
- `message_id` makes retries and late tool-result patches idempotent
  (same id = update, new id = append);
- no need to keep the whole history in memory.

**Use `send_dialog`** when you only have the finished conversation — batch
imports, post-hoc exports, or frameworks that hand you the full transcript at
the end. It replaces the entire stored dialog with the snapshot you send:

```python
from datetime import datetime, timezone

client.send_dialog(
    "support-bot",
    "sess-42",
    messages=[
        {"role": "user", "text": "Where is my order?",
         "ts": datetime(2026, 7, 20, 9, 58, tzinfo=timezone.utc), "message_id": "m-1"},
        {"role": "bot", "text": "Let me check that for you.", "message_id": "m-2"},
        {"role": "tool", "text": "",
         "tool_call": {"name": "order_lookup", "label": "Look up order", "status": "ok"},
         "message_id": "m-3"},
        {"role": "bot", "text": "It shipped on July 18.", "message_id": "m-4"},
    ],
    end_user={"external_id": "u-1", "locale": "en-US"},
)
```

Don't mix the two for the same session mid-flight: a snapshot wipes and
replaces everything streamed so far.

`ts` accepts a `datetime` (serialized as RFC 3339 UTC; naive values are taken
as UTC) or a pre-formatted RFC 3339 string. Omit it to use the server arrival
time.

## Instrument an existing client (zero-touch)

Already calling OpenAI or Anthropic directly? Wrap the client once and every
completed chat call is mirrored into PharosOne as a full-dialog snapshot — no
manual send calls. The wrapper is a transparent duck-typed proxy: it never
imports `openai`/`anthropic` (still zero dependencies), returns the provider's
exact response unchanged, and covers sync + async clients including
`stream=True` (chunks are accumulated and flushed when the stream completes;
a stream abandoned early flushes what was seen on `close()`).

```python
from openai import OpenAI
from pharosone_dialogs import PharosOne
from pharosone_dialogs.instrument import wrap_openai, pharos_session

pharos = PharosOne(base_url="https://pharosone.example.com", api_key="sk-...")
client = wrap_openai(OpenAI(), pharos=pharos, agent_id="support-bot")

with pharos_session("sess-42"):                 # bind the dialog id
    reply = client.chat.completions.create(     # your call, unchanged
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Where is my order?"}],
    )
```

`wrap_anthropic(client, pharos=..., agent_id=...)` does the same for
`anthropic.Anthropic()` / `AsyncAnthropic()` — it instruments
`messages.create` (including `stream=True`; the `messages.stream()` helper
context manager is not instrumented yet and passes through untouched).

**OpenAI-compatible endpoints work for free.** Ollama, vLLM, OpenRouter,
Azure OpenAI, LM Studio, ... — anything you reach through
`OpenAI(base_url=...)` (or any client with the same `chat.completions.create`
surface) goes through `wrap_openai` unchanged.

**Session binding.** The provider API has no dialog notion, so the wrapper
resolves the PharosOne `session_id` per call, in this order (explicit wins):

1. `pharos_session_id="sess-42"` passed to the instrumented call — stripped
   before the request, it never reaches the provider;
2. the innermost `with pharos_session("sess-42"):` scope (`contextvars`,
   async-safe);
3. `session_id=` fixed at wrap time;
4. fallback: a stable `sha256` of the first user message text — best-effort,
   good enough because each request re-sends the whole history, so every turn
   of one conversation hashes to the same dialog.

**Fire-and-forget.** Snapshots are sent from a background daemon thread with a
bounded queue (oldest snapshot dropped with a logged warning on overflow); the
LLM call never waits on PharosOne and PharosOne errors are logged, never
raised into your code. Pass `on_result=lambda r: ...` to receive each
`send_dialog` result dict (`flagged`, `fast_scan`, `dialog_id`) from the
worker — e.g. to alert or block on `flagged`. For deterministic flushing
(tests, shutdown) use `client.pharos_instrumentation.drain()` / `.close()`;
a drain also runs automatically at interpreter exit.

**Tool calls** ride along automatically: provider tool calls become
`role="tool"` messages (`pending` when issued, resolved to `ok`/`error` with a
`result_preview` when the result appears in the history). Tool args/results
are sent as previews capped at ~500 chars (message text at <20000); pass
`redact=lambda text: ...` to scrub the previews before they leave the process.
`sync_agent=True` additionally upserts the agent description from the system
prompt once per process.

**LangChain / LiteLLM one-liners** (framework packages stay optional —
`langchain-core` is only needed when the handler is instantiated):

```python
# LangChain: snapshot per chat-model run (+ tool start/end updates)
from pharosone_dialogs.integrations.langchain import PharosCallbackHandler
llm.invoke(messages, config={"callbacks": [PharosCallbackHandler(pharos, "support-bot", session_id="sess-42")]})

# LiteLLM: success_callback-compatible
import litellm
from pharosone_dialogs.integrations.litellm import pharos_litellm_callback
litellm.success_callback = [pharos_litellm_callback(pharos, "support-bot")]
# per-call session: litellm.completion(..., metadata={"pharos_session_id": "sess-42"})
```

## Errors

Non-2xx responses raise `PharosOneError` with the HTTP status and the API
error detail:

```python
from pharosone_dialogs import PharosOne, PharosOneError

try:
    client.send_dialog("support-bot", "sess-42", messages)
except PharosOneError as err:
    print(err.status, err.detail)  # e.g. 409 duplicate message_id
```

## Development

```bash
cd sdk/python
python3 -m unittest discover -s tests
```
