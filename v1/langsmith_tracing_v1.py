"""LangSmith tracing for a ChatGPT app on MCP Python SDK 1.x (FastMCP).

OpenAI's Apps SDK examples are written against 1.x, so a plugin started from
them is most likely here. 1.x has no middleware list and no built-in
OpenTelemetry, so this module does both jobs itself.

    from langsmith_tracing_v1 import setup_langsmith_tracing, trace_tool_calls

    mcp = FastMCP("Holidays")
    ...define tools, or set mcp._mcp_server.request_handlers[CallToolRequest]...
    setup_langsmith_tracing()
    trace_tool_calls(mcp)          # wrap whichever call_tool handler is registered

`trace_tool_calls` replaces the server's CallToolRequest handler with a wrapper
that opens one span per tool call, records the arguments and result, reads
`_meta["openai/session"]` and `_meta["openai/subject"]` off the request, and
forwards to the original handler. The span is a root, so each tool call is one
LangSmith trace and the session id groups them into a thread.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import mcp.types as types
from mcp.server.fastmcp import FastMCP
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("langsmith-mcp-v1")

_META_MAP = {
    "openai/session": "langsmith.metadata.session_id",
    "openai/subject": "langsmith.metadata.user_id",
    "openai/organization": "langsmith.metadata.openai_organization",
    "openai/locale": "langsmith.metadata.openai_locale",
}


def setup_langsmith_tracing(service_name: str = "holidays-mcp-v1") -> TracerProvider:
    api_key = os.environ["LANGSMITH_API_KEY"]
    project = os.environ.get("LANGSMITH_PROJECT", "default")
    endpoint = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").rstrip("/")
    exporter = OTLPSpanExporter(
        endpoint=f"{endpoint}/otel/v1/traces",
        headers={"x-api-key": api_key, "Langsmith-Project": project},
    )
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info("LangSmith tracing on: project=%s endpoint=%s", project, endpoint)
    return provider


def _meta_dict(params: types.CallToolRequestParams) -> dict[str, Any]:
    """`_meta` as a plain dict. 1.x models it as RequestParams.Meta with extra='allow',
    so ChatGPT's `openai/*` keys land in model_extra."""
    meta = params.meta
    if meta is None:
        return {}
    out: dict[str, Any] = dict(meta.model_extra or {})
    if meta.progressToken is not None:
        out["progressToken"] = meta.progressToken
    return out


def trace_tool_calls(mcp: FastMCP) -> None:
    server = mcp._mcp_server  # the low-level server FastMCP wraps
    original = server.request_handlers[types.CallToolRequest]

    async def traced_call_tool(req: types.CallToolRequest) -> types.ServerResult:
        name = req.params.name
        arguments = req.params.arguments or {}
        meta = _meta_dict(req.params)

        attrs: dict[str, Any] = {
            "mcp.method.name": "tools/call",
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": name,
            "langsmith.span.kind": "tool",
            "langsmith.trace.name": name,
            "langsmith.span.tags": "chatgpt-app,mcp,sdk-v1",
            "gen_ai.prompt": json.dumps(arguments, ensure_ascii=False, default=str),
        }
        for wire_key, ls_key in _META_MAP.items():
            value = meta.get(wire_key)
            if isinstance(value, str) and value:
                attrs[ls_key] = value

        with _tracer.start_as_current_span(f"tools/call {name}", kind=SpanKind.SERVER, attributes=attrs) as span:
            result = await original(req)
            try:
                payload = result.root.model_dump(by_alias=True, mode="json", exclude_none=True)
                payload.pop("_meta", None)
                span.set_attribute("gen_ai.completion", json.dumps(payload, ensure_ascii=False, default=str))
                if payload.get("isError") is True:
                    message = " ".join(c.get("text", "") for c in payload.get("content", []) if isinstance(c, dict)).strip()
                    span.add_event("exception", {"exception.type": "ToolError", "exception.message": message or "tool error"})
            except Exception:
                logger.debug("could not serialise tool result for tracing", exc_info=True)
            return result

    server.request_handlers[types.CallToolRequest] = traced_call_tool
