"""The same stand-in holidays server, written the way OpenAI's Apps SDK Python
examples are written: FastMCP from MCP Python SDK 1.x, served as an ASGI app.

Run:
    uv run server_v1.py
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from mcp.server.fastmcp import FastMCP  # noqa: E402

from langsmith_tracing_v1 import setup_langsmith_tracing, trace_tool_calls  # noqa: E402

mcp = FastMCP(
    "Holidays (demo, sdk v1)",
    instructions="Search and inspect package holidays. Data is canned for the demo.",
    host=os.environ.get("MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("MCP_PORT", "8765")),
)

_HOLIDAYS = [
    {"id": "HOL-1041", "destination": "Lisbon", "nights": 4, "board": "Room only", "price_gbp": 489, "month": "October"},
    {"id": "HOL-2210", "destination": "Lisbon", "nights": 7, "board": "Bed and breakfast", "price_gbp": 812, "month": "October"},
    {"id": "HOL-3305", "destination": "Malaga", "nights": 7, "board": "Half board", "price_gbp": 1105, "month": "October"},
    {"id": "HOL-4120", "destination": "Reykjavik", "nights": 3, "board": "Room only", "price_gbp": 640, "month": "November"},
]


@mcp.tool()
def search_holidays(
    destination: str,
    month: str | None = None,
    max_price_gbp: int | None = None,
    customer_request: str | None = None,
) -> dict:
    """Search package holidays by destination, optionally filtered by month and a price ceiling in GBP.

    customer_request: one short sentence describing what the customer asked for, in their own terms.
    Not used to filter. It is recorded on the trace so the request can be compared with what was returned.

    A search that matches nothing returns why it missed and the closest package, so the
    assistant can explain the gap instead of saying nothing is available. An empty list
    on its own gives the model nothing to work with.
    """
    at_destination = [h for h in _HOLIDAYS if h["destination"].lower() == destination.lower()]
    out = list(at_destination)
    if month:
        out = [h for h in out if h["month"].lower() == month.lower()]
    if max_price_gbp is not None:
        out = [h for h in out if h["price_gbp"] <= max_price_gbp]
    if out:
        return {"results": out, "count": len(out)}

    if not at_destination:
        return {"results": [], "count": 0, "no_match_reason": f"We do not currently sell packages to {destination}."}

    cheapest = min(at_destination, key=lambda h: h["price_gbp"])
    reasons = []
    if month and not any(h["month"].lower() == month.lower() for h in at_destination):
        reasons.append(f"nothing departs in {month}")
    if max_price_gbp is not None and cheapest["price_gbp"] > max_price_gbp:
        reasons.append(f"the cheapest is £{cheapest['price_gbp']}, above the £{max_price_gbp} ceiling")
    return {
        "results": [],
        "count": 0,
        "no_match_reason": f"No {destination} package matched because " + " and ".join(reasons or ["of the filters applied"]) + ".",
        "closest": cheapest,
    }


@mcp.tool()
def get_holiday(holiday_id: str) -> dict:
    """Return the full record for one holiday by id."""
    for h in _HOLIDAYS:
        if h["id"] == holiday_id:
            return {**h, "includes": ["Return flights from London", "Hotel", "23kg baggage"]}
    raise ValueError(f"No holiday with id {holiday_id!r}")


@mcp.tool()
def check_availability(holiday_id: str, departure_date: str) -> dict:
    """Check whether a holiday can depart on a given ISO date. Returns seats left and the live price."""
    base = next((h for h in _HOLIDAYS if h["id"] == holiday_id), None)
    if base is None:
        raise ValueError(f"No holiday with id {holiday_id!r}")
    day = int(departure_date[-2:])
    seats = (day * 7) % 9
    return {"holiday_id": holiday_id, "departure_date": departure_date, "available": seats > 0, "seats_left": seats, "price_gbp": base["price_gbp"] + (day % 5) * 15}


# Tracing last, after the tools exist, so the wrapper sees the real call_tool handler.
provider = setup_langsmith_tracing()
trace_tool_calls(mcp)

if __name__ == "__main__":
    try:
        mcp.run(transport="streamable-http")
    finally:
        provider.force_flush()
        provider.shutdown()
