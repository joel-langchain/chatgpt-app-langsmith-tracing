"""Pretend to be the ChatGPT host.

ChatGPT sends every tool call with `_meta` carrying an anonymised conversation
id (`openai/session`), an anonymised user id (`openai/subject`), and a locale.
This script makes the same calls the same way, so the server can be checked
without publishing anything to ChatGPT.

Two conversations, two users, several tool calls each. In LangSmith each tool
call is one trace, and the traces of one conversation share a thread.

The second conversation asks for an id that does not exist, which comes back as a
normal result, and then sends a malformed date, which does not. Only the second of
those is an error run.

Run (with server.py already running):
    uv run simulate_chatgpt.py
"""

from __future__ import annotations

import asyncio
import os
import uuid

from dotenv import load_dotenv

from mcp import Client

load_dotenv()

URL = f"http://{os.environ.get('MCP_HOST', '127.0.0.1')}:{os.environ.get('MCP_PORT', '8765')}/mcp"


def chatgpt_meta(session: str, subject: str) -> dict:
    # Field names and meanings per https://developers.openai.com/apps-sdk/reference
    return {
        "openai/session": session,
        "openai/subject": subject,
        "openai/locale": "en-GB",
        "openai/userAgent": "ChatGPT/1.2026.09 (simulated)",
    }


CONVERSATIONS = [
    # (session id, user id, [(tool, args), ...])  one entry per turn ChatGPT would make
    (
        f"conv_{uuid.uuid4().hex[:12]}",
        f"user_{uuid.uuid4().hex[:8]}",
        [
            ("search_holidays", {"destination": "Lisbon", "month": "October",
             "customer_request": "a week in Lisbon in October, somewhere warm"}),
            ("get_holiday", {"holiday_id": "HOL-2210"}),
            ("check_availability", {"holiday_id": "HOL-2210", "departure_date": "2026-10-14"}),
            ("check_availability", {"holiday_id": "HOL-2210", "departure_date": "2026-10-18"}),
        ],
    ),
    (
        f"conv_{uuid.uuid4().hex[:12]}",
        f"user_{uuid.uuid4().hex[:8]}",
        [
            ("search_holidays", {"destination": "Malaga", "max_price_gbp": 900,
             "customer_request": "Malaga on a budget, nothing over 900"}),
            ("search_holidays", {"destination": "Lisbon", "max_price_gbp": 900,
             "customer_request": "cheapest Lisbon option they have"}),
            ("get_holiday", {"holiday_id": "HOL-9999"}),  # unknown id -> a normal result naming the ids that exist
            # malformed argument rather than a miss -> a real failure, so one run shows as an error
            ("check_availability", {"holiday_id": "HOL-2210", "departure_date": "next Tuesday"}),
        ],
    ),
]


async def main() -> None:
    async with Client(URL) as client:
        tools = await client.list_tools()
        print("server tools:", [t.name for t in tools.tools])
        for session, subject, turns in CONVERSATIONS:
            print(f"\nconversation {session}  user {subject}")
            for tool, args in turns:
                res = await client.call_tool(tool, args, meta=chatgpt_meta(session, subject))
                flag = "ERROR" if res.is_error else "ok"
                body = res.structured_content if res.structured_content is not None else [c.model_dump() for c in res.content]
                print(f"  {tool}({args}) -> {flag} {str(body)[:110]}")
    print("\nDone. Check LangSmith project", os.environ.get("LANGSMITH_PROJECT") or "default", "in a few seconds.")


if __name__ == "__main__":
    asyncio.run(main())
