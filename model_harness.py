"""Check that a real model populates customer_request, and that it reaches the trace.

ChatGPT fills in whatever a tool's schema asks for. This drives the same server with
a real model through the OpenAI Agents SDK, so you can see what the model writes into
customer_request before you add the field to a live plugin.

Not ChatGPT itself. Same idea, your own system prompt, so treat it as a check on the
mechanism rather than a reproduction of the product.

Run (server.py already running, model credentials in the env):
    uv sync --extra harness
    uv run model_harness.py
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled  # noqa: E402
from agents.mcp import MCPServerStreamableHttp  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

URL = f"http://{os.environ.get('MCP_HOST', '127.0.0.1')}:{os.environ.get('MCP_PORT', '8765')}/mcp"
# `or` rather than a get() default, because .env.example ships these keys empty and an
# empty value is present, not absent.
MODEL = os.environ.get("HARNESS_MODEL") or "gpt-4o-mini"

# Phrased the way a customer would, with intent the tool parameters cannot hold.
QUESTIONS = [
    "My wife and I want a week somewhere warm in October, ideally Lisbon, and we are trying to stay under 900 pounds all in.",
    "Looking for a short break in Reykjavik, we want to see the northern lights in November.",
]


async def main() -> None:
    set_tracing_disabled(True)  # the server's own tracing is what we are looking at
    set_default_openai_client(
        AsyncOpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None, api_key=os.environ["OPENAI_API_KEY"])
    )
    async with MCPServerStreamableHttp(name="holidays", params={"url": URL}, cache_tools_list=True) as srv:
        agent = Agent(
            name="holidays assistant",
            # Deliberately says nothing about customer_request. The model fills it from the schema.
            instructions="Use the tools for every fact about holidays, prices, and availability. Be brief.",
            model=MODEL,
            mcp_servers=[srv],
        )
        for q in QUESTIONS:
            print(f"\nUSER {q}")
            result = await Runner.run(agent, q)
            print(f"ASSISTANT {result.final_output[:200]}")

    print("\nDone. The tool calls are in your LangSmith project. They carry no session_id, so")
    print("they are not grouped into a thread, because the Agents SDK does not send _meta the")
    print("way ChatGPT does. Look at the arguments on each tool run and you will see")
    print("customer_request alongside destination, month, and max_price_gbp.")


if __name__ == "__main__":
    asyncio.run(main())
