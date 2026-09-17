"""LangSmith tracing for an MCP server that ChatGPT calls through the Apps SDK.

Two pieces.

1. `setup_langsmith_tracing()` installs an OTLP exporter pointed at LangSmith.
   The MCP Python SDK (v2) already wraps every inbound request in an
   OpenTelemetry span, so this alone gets a span per tool call into LangSmith.
   That span carries the tool name but not the arguments or the result.

2. `langsmith_middleware` runs inside the SDK's span for each `tools/call` and
   adds what LangSmith needs to make the trace useful.

   - `langsmith.span.kind = tool` so it renders as a tool run.
   - `gen_ai.prompt` and `gen_ai.completion` carry the arguments and the result.
   - `langsmith.metadata.session_id` is taken from `_meta["openai/session"]`,
     the anonymised conversation id ChatGPT sends on every tool call. This is
     what groups the tool calls of one conversation into a LangSmith thread.
   - `langsmith.metadata.user_id` is taken from `_meta["openai/subject"]`, the
     anonymised user id.
   - `gen_ai.prompt` and `gen_ai.completion` are chat messages, which is what lets a
     thread-level evaluator assemble the thread into a conversation. The user turn
     carries the customer's words and the arguments that were sent; the assistant
     turn carries the tool's own return value. The raw argument and result JSON is
     kept on the run as metadata either way.

The `_meta` field names come from the Apps SDK reference:
https://developers.openai.com/apps-sdk/reference
LangSmith's OpenTelemetry attribute names come from:
https://docs.langchain.com/langsmith/trace-with-opentelemetry
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Sequence

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from pydantic import BaseModel

from mcp.server.context import CallNext, HandlerResult, ServerRequestContext

logger = logging.getLogger(__name__)

# ChatGPT host -> LangSmith metadata. Anything not listed is ignored.
_META_MAP = {
    "openai/session": "langsmith.metadata.session_id",
    "openai/subject": "langsmith.metadata.user_id",
    "openai/organization": "langsmith.metadata.openai_organization",
    "openai/locale": "langsmith.metadata.openai_locale",
}


class ToolCallsOnly(SpanExporter):
    """Forward only `tools/call` spans.

    The SDK also spans `server/discover`, `initialize`, and `tools/list`. ChatGPT
    issues those constantly and they carry no conversation id, so in LangSmith
    they would be untagged runs sitting outside every thread. Dropping them here
    keeps the project to one run per tool call.
    """

    def __init__(self, delegate: SpanExporter) -> None:
        self._delegate = delegate

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        keep = [s for s in spans if (s.attributes or {}).get("mcp.method.name") == "tools/call"]
        return self._delegate.export(keep) if keep else SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)


def setup_langsmith_tracing(service_name: str = "holidays-mcp") -> TracerProvider:
    """Point the process's OpenTelemetry tracer at LangSmith.

    Reads LANGSMITH_API_KEY, LANGSMITH_PROJECT and LANGSMITH_ENDPOINT. Returns the
    provider so the caller can `force_flush()` before exit.
    """
    api_key = os.environ["LANGSMITH_API_KEY"]
    project = os.environ.get("LANGSMITH_PROJECT", "default")
    endpoint = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").rstrip("/")

    exporter = OTLPSpanExporter(
        endpoint=f"{endpoint}/otel/v1/traces",
        headers={"x-api-key": api_key, "Langsmith-Project": project},
    )
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(ToolCallsOnly(exporter)))
    trace.set_tracer_provider(provider)
    logger.info("LangSmith tracing on: project=%s endpoint=%s", project, endpoint)
    return provider


def _as_messages(tool_name: str, arguments: dict, payload: Any | None = None) -> tuple[str, str | None]:
    """Render a tool call as a user turn and its result as an assistant turn.

    Thread-level evaluators assemble a conversation from the traces in a thread and
    need a top-level `messages` key on inputs and outputs. Two things matter in what
    goes into those messages. A judge needs the customer's own words, and a person
    debugging needs the arguments that were actually sent, so the user turn carries
    both. The assistant turn carries the tool's own payload rather than the MCP
    envelope around it, because `isError` and `resultType` are on the run already.
    """
    request = arguments.get("customer_request")
    rest = {k: v for k, v in arguments.items() if k != "customer_request"}
    params = ", ".join(f"{k}={v!r}" for k, v in rest.items())
    if request:
        user_text = f"{request}\n\n{tool_name}({params})" if params else f"{request}\n\n{tool_name}()"
    else:
        user_text = f"{tool_name}({params})"

    if payload is None:
        return user_text, None

    # Prefer the tool's own return value, then its text blocks, then the envelope.
    structured = payload.get("structuredContent") if isinstance(payload, dict) else None
    if isinstance(structured, dict) and "result" in structured:
        reply = json.dumps(structured["result"], ensure_ascii=False, indent=2, default=str)
    else:
        texts = [c.get("text", "") for c in (payload.get("content") or []) if isinstance(c, dict)]
        reply = "\n".join(t for t in texts if t) or json.dumps(payload, ensure_ascii=False, default=str)
    return user_text, reply


def _to_jsonable(result: HandlerResult) -> Any:
    if isinstance(result, BaseModel):
        return result.model_dump(by_alias=True, mode="json", exclude_none=True)
    return result


async def langsmith_middleware(ctx: ServerRequestContext, call_next: CallNext) -> HandlerResult:
    """Annotate the SDK's span for each tools/call with LangSmith attributes."""
    if ctx.method != "tools/call":
        return await call_next(ctx)

    span = trace.get_current_span()
    params = dict(ctx.params or {})
    tool_name = params.get("name")
    arguments = params.get("arguments") or {}
    meta = dict(ctx.meta or {})

    # Inputs and outputs are shaped as chat messages, not raw arguments. Thread-level
    # evaluators assemble a conversation from the traces in a thread and require a
    # top-level `messages` key on both sides; without it they silently never run.
    # The customer_request argument is the nearest thing to a user turn, and the tool
    # result is the reply. Raw arguments stay on the run as metadata.
    user_text, _ = _as_messages(str(tool_name), arguments)
    attrs: dict[str, Any] = {
        "langsmith.span.kind": "tool",
        "langsmith.span.tags": "chatgpt-app,mcp",
        "gen_ai.prompt": json.dumps({"messages": [{"role": "user", "content": user_text}]}, ensure_ascii=False, default=str),
        "langsmith.metadata.tool_arguments": json.dumps(arguments, ensure_ascii=False, default=str),
    }
    if isinstance(tool_name, str):
        attrs["langsmith.trace.name"] = tool_name
    for wire_key, ls_key in _META_MAP.items():
        value = meta.get(wire_key)
        if isinstance(value, str) and value:
            attrs[ls_key] = value
    span.set_attributes(attrs)

    result = await call_next(ctx)

    try:
        payload = _to_jsonable(result)
        if isinstance(payload, dict):
            payload = {k: v for k, v in payload.items() if k != "_meta"}  # serverInfo stamp, not output
        _, reply = _as_messages(str(tool_name), arguments, payload)
        span.set_attribute(
            "gen_ai.completion",
            json.dumps({"messages": [{"role": "assistant", "content": reply}]}, ensure_ascii=False, default=str),
        )
        span.set_attribute("langsmith.metadata.tool_result", json.dumps(payload, ensure_ascii=False, default=str))

        # A tool that failed comes back as a normal result with isError=true. The SDK's
        # own span sets an error status, but LangSmith's OTel ingest marks a run as
        # failed from an `exception` event, so add one carrying the tool's message.
        if isinstance(payload, dict) and payload.get("isError") is True:
            message = " ".join(
                c.get("text", "") for c in payload.get("content", []) if isinstance(c, dict)
            ).strip() or "tool error"
            span.add_event("exception", {"exception.type": "ToolError", "exception.message": message})
    except Exception:  # never let tracing break the tool
        logger.debug("could not serialise tool result for tracing", exc_info=True)
    return result
