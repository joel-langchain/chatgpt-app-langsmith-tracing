# Tracing a ChatGPT app into LangSmith

A ChatGPT app is an MCP server. ChatGPT runs the conversation and the reasoning, your server runs the tools. On each tool call ChatGPT sends your server the tool name, the arguments, and a conversation id. It does not send the user's message, its own reasoning, or the answer it writes afterwards.

This example traces every tool call into LangSmith as one run, with the arguments and the result, and groups the calls from one conversation into a thread. The tools themselves are not changed.

## How it works

**ChatGPT sends `_meta` with every tool call.** Two fields matter, per the [Apps SDK reference](https://developers.openai.com/apps-sdk/reference).

| `_meta` key | Meaning |
|---|---|
| `openai/session` | Anonymised conversation id, the same for every call in one conversation |
| `openai/subject` | Anonymised user id |

**The MCP Python SDK opens an OpenTelemetry span for every request.** That span has the tool name and nothing else. `langsmith_tracing.py` adds a middleware that runs inside the span for each `tools/call` and sets the attributes LangSmith reads, from its [OpenTelemetry attribute table](https://docs.langchain.com/langsmith/trace-with-opentelemetry).

| Attribute | Value |
|---|---|
| `langsmith.span.kind` | `tool` |
| `langsmith.trace.name` | the tool name |
| `gen_ai.prompt` | the arguments, as JSON |
| `gen_ai.completion` | the result, as JSON |
| `langsmith.metadata.session_id` | `_meta["openai/session"]`, which forms the thread |
| `langsmith.metadata.user_id` | `_meta["openai/subject"]` |

**A standard OTLP exporter sends the spans to LangSmith.** Same route as any agent that already emits OpenTelemetry, so it fits an existing collector, and with it any masking or export to a second backend.

Two details the middleware handles.

The SDK also spans `initialize`, `tools/list`, and `server/discover`. ChatGPT sends those constantly and they carry no conversation id, so the exporter drops everything except `tools/call`.

A tool that fails returns a normal result with `isError: true`. LangSmith marks a run as failed from an `exception` event, so the middleware adds one carrying the error text ChatGPT received. Without it a failed tool call looks successful.

## Run it

```
cp .env.example .env        # add your LangSmith API key
uv sync
uv run server.py            # terminal 1, the MCP server on http://127.0.0.1:8765/mcp
uv run simulate_chatgpt.py  # terminal 2, calls the server the way ChatGPT does
uv run verify_langsmith.py  # reads the runs back, grouped by thread
```

`simulate_chatgpt.py` plays two conversations from two users, several tool calls each, with the `_meta` ChatGPT would send. The last call uses a bad id so one run shows as an error.

## What you see in LangSmith

One run per tool call. The run is named after the tool, its inputs are the arguments, its outputs are the result, and it carries latency and status. Open the Threads view and each conversation is one thread with its tool calls in order.

## On MCP Python SDK 1.x

OpenAI's [Apps SDK examples](https://github.com/openai/openai-apps-sdk-examples) use 1.x (`from mcp.server.fastmcp import FastMCP`), which 2.x removed, so an app started from them is pinned to `mcp<2`.

1.x has no middleware and no built-in OpenTelemetry. The `v1/` folder does the same job by wrapping the server's `CallToolRequest` handler. One span per tool call, the same attributes, the same threads, the same error marking.

```
cd v1 && uv sync && uv run server_v1.py    # then run ../simulate_chatgpt.py as before
```

Do not use the `Mcp-Session-Id` HTTP header as the conversation key. It was removed in the 2026-07-28 MCP specification. `openai/session` is the field ChatGPT provides for this.

## What it does not show

The user's question, ChatGPT's reasoning between calls, and the final answer stay on OpenAI's side and never reach your server. Two ways to close part of that gap.

Add a field to the tool schema, for example `customer_request`, a one line summary of what the user asked. ChatGPT fills in whatever the schema asks for, so the summary arrives with the arguments and lands in the trace. It is the model's paraphrase, not a transcript.

When the reasoning moves onto a runtime you own, trace that runtime and the tool call becomes one step inside a full trace.

## Files

| File | Purpose |
|---|---|
| `server.py` | A holidays MCP server with three tools and canned data |
| `langsmith_tracing.py` | Exporter setup, the `tools/call` filter, and the middleware |
| `simulate_chatgpt.py` | Calls the server with the `_meta` ChatGPT would send |
| `verify_langsmith.py` | Reads the runs back and prints them by thread |
| `v1/` | The same server and tracing on MCP Python SDK 1.x |
