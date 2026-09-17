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
| `gen_ai.prompt` | a `messages` list, the customer's request and the arguments sent |
| `gen_ai.completion` | a `messages` list, the tool's own return value |
| `langsmith.metadata.session_id` | `_meta["openai/session"]`, which forms the thread |
| `langsmith.metadata.user_id` | `_meta["openai/subject"]` |

**A standard OTLP exporter sends the spans to LangSmith.** Same route as any agent that already emits OpenTelemetry, so it fits an existing collector, and with it any masking or export to a second backend.

Two details the middleware handles.

The SDK also spans `initialize`, `tools/list`, and `server/discover`. ChatGPT sends those constantly and they carry no conversation id, so the exporter drops everything except `tools/call`.

A tool that fails returns a normal result with `isError: true`. LangSmith marks a run as failed from an `exception` event, so the middleware adds one carrying the error text ChatGPT received. Without it a failed tool call looks successful.

Inputs and outputs are chat messages rather than the raw argument and result JSON, which is a requirement of thread-level evaluation, covered below. Two readers need different things from those messages, so the user turn carries the customer's own words followed by the arguments that were actually sent, and the assistant turn carries the tool's own return value rather than the MCP envelope around it. A judge gets the intent, and anyone debugging still sees the parameters. The untouched argument and result JSON is on the run as `tool_arguments` and `tool_result` metadata either way.

## Run it

```
cp .env.example .env        # add your LangSmith API key
uv sync
uv run server.py            # terminal 1, the MCP server on http://127.0.0.1:8765/mcp
uv run simulate_chatgpt.py  # terminal 2, calls the server the way ChatGPT does
uv run verify_langsmith.py  # reads the runs back, grouped by thread
```

To drive the same server with a real model instead of the simulator, which is how to check what a model writes into `customer_request`, add the harness extra and give it model credentials.

```
uv sync --extra harness
uv run model_harness.py
```

`simulate_chatgpt.py` plays two conversations from two users, several tool calls each, with the `_meta` ChatGPT would send. The last call uses a bad id so one run shows as an error.

## What you see in LangSmith

One run per tool call, named after the tool, carrying latency and status. Its inputs are the customer's request followed by the arguments that were sent, and its outputs are the tool's return value, both as chat messages for the reason given above. The untouched argument and result JSON is on the run as `tool_arguments` and `tool_result` metadata. Open the Threads view and each conversation is one thread with its tool calls in order.

## On MCP Python SDK 1.x

OpenAI's [Apps SDK examples](https://github.com/openai/openai-apps-sdk-examples) use 1.x (`from mcp.server.fastmcp import FastMCP`), which 2.x removed, so an app started from them is pinned to `mcp<2`.

1.x has no middleware and no built-in OpenTelemetry. The `v1/` folder does the same job by wrapping the server's `CallToolRequest` handler. One span per tool call, the same attributes, the same threads, the same error marking.

```
cd v1 && uv sync && uv run server_v1.py
```

Run the simulator from the repo root, not from `v1/`, in a second terminal.

```
uv run simulate_chatgpt.py
```

`v1/` pins `mcp<2`, where `from mcp import Client` does not exist, so the simulator only runs against the root environment. The server it talks to can be either version.

Do not use the `Mcp-Session-Id` HTTP header as the conversation key. It was removed in the 2026-07-28 MCP specification. `openai/session` is the field ChatGPT provides for this.

## Tool responses shape what the agent can say

`search_holidays` does not return a bare empty list when nothing matches. It returns why the search missed and the closest package it does have.

```json
{
  "results": [],
  "count": 0,
  "no_match_reason": "No Malaga package matched because the cheapest is \u00a31105, above the \u00a3900 ceiling.",
  "closest": {"id": "HOL-3305", "destination": "Malaga", "nights": 7, "price_gbp": 1105, "month": "October"}
}
```

An empty list gives the model nothing, so the customer gets a dead end. With a reason and a near miss the model can explain the gap and offer the alternative. This matters more for a ChatGPT app than for an agent you run yourself, because the tool response is the only thing you control once the conversation is on OpenAI's side. It is also visible on the trace, so the quality of these responses is something you can evaluate rather than guess at.

## Evaluating a whole conversation

A run-level evaluator scores one tool call. It cannot answer whether the customer got what they asked for, because a conversation spans several calls and each one may be individually fine. That needs a thread-level evaluator, which reads the assembled conversation instead.

`create_thread_evaluator.py` creates one from code.

```
uv run create_thread_evaluator.py --project chatgpt-app-tracing
```

It posts a rule with `group_by: "thread_id"`, which is the field that makes it a thread evaluator, and an inline LLM-as-judge whose prompt receives the assembled conversation as `all_messages`. The UI creates the judge as a Prompt Hub reference instead, so a rule made here reads slightly differently from one made by hand, but it behaves the same.

Three things to know.

**Inputs and outputs must carry a top-level `messages` key**, in LangChain, OpenAI, or Anthropic format. Raw MCP arguments do not qualify, and a thread evaluator given traces it cannot assemble produces nothing at all rather than an error. That is why the middleware emits messages.

**It fires when a thread goes idle**, ten minutes by default and two minutes at the least. The setting lives on the project, so it can be changed without the UI.

```
curl -X PATCH -H "x-api-key: $LANGSMITH_API_KEY" -H "Content-Type: application/json" \
  "$LANGSMITH_ENDPOINT/api/v1/sessions/<project_id>" \
  -d '{"extra": {"thread_idle_seconds": "120"}}'
```

To see a result without waiting at all, trigger the rule by hand.

```
curl -X POST -H "x-api-key: $LANGSMITH_API_KEY" \
  "$LANGSMITH_ENDPOINT/api/v1/runs/rules/<rule_id>/trigger"
```

**Feedback attaches to one representative trace in the thread**, not to every call in it, so in the project view switch the Threads/Traces/Runs toggle to Threads (`?runview=threads`) to see the score against the conversation.

## What it does not show, and how to narrow it

The user's question, ChatGPT's reasoning between calls, and the final answer stay on OpenAI's side and never reach your server. Two ways to narrow that.

**Ask the schema for it.** A model fills in whatever a tool's schema asks for, so a field that your tool logic ignores still arrives with the arguments and lands on the trace. `search_holidays` here takes a `customer_request` string described as one short sentence covering what the customer asked for. `model_harness.py` drives the server with a real model whose instructions say nothing about that field, and the model populates it from the schema alone.

```
search_holidays  {"destination": "Lisbon", "month": "October", "max_price_gbp": 900}
  customer_request -> 'week in Lisbon under 900 pounds'

search_holidays  {"destination": "Reykjavik", "month": "November"}
  customer_request -> 'short break to see the northern lights'
```

The second one is the point. Northern lights appears in none of the structured arguments, so that intent would otherwise be invisible, and on the trace it becomes something an evaluator can judge the returned holiday against.

It is the model's paraphrase rather than a transcript, so treat it as evidence and not a record. It also means you are capturing conversation content your tool does not need, which is worth a look from whoever owns data handling before it goes near production.

**Move the reasoning onto your own runtime.** Trace that runtime and the tool call becomes one step inside a full trace rather than the whole of it.

## Files

| File | Purpose |
|---|---|
| `server.py` | A holidays MCP server with three tools and canned data |
| `langsmith_tracing.py` | Exporter setup, the `tools/call` filter, and the middleware |
| `simulate_chatgpt.py` | Calls the server with the `_meta` ChatGPT would send |
| `verify_langsmith.py` | Reads the runs back and prints them by thread |
| `model_harness.py` | Drives the server with a real model, to see what it writes into `customer_request` |
| `create_thread_evaluator.py` | Creates a thread-level LLM-as-judge evaluator on a project |
| `v1/` | The same server and tracing on MCP Python SDK 1.x |
