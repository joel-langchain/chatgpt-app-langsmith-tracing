"""Create a thread-level LLM-as-judge evaluator on a tracing project, from code.

A run-level evaluator scores one tool call. This one scores a whole conversation,
which is what you want when a single customer request spans several tool calls.

It needs the project's traces to carry a top-level `messages` key on inputs and
outputs, which is what `langsmith_tracing.py` emits. Without that the thread
cannot be assembled and the evaluator will not run.

Run:
    uv run create_thread_evaluator.py                      # live only
    uv run create_thread_evaluator.py --backfill-hours 24   # also score existing threads
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone

import httpx
from dotenv import load_dotenv

load_dotenv()

JUDGE_SYSTEM = """You are reviewing one customer conversation with a holiday booking assistant.

The conversation is a sequence of tool calls. A user turn is either what the customer asked
for in their own words, or a description of a tool the assistant called. An assistant turn is
what that tool returned.

Decide whether the conversation served what the customer actually asked for. Consider whether
the results match the stated constraints, whether stated intent that is not a search parameter
was respected, and whether any error left the customer without an answer.

Score 1 if the conversation served the request, 0 if it did not."""

JUDGE_SCHEMA = {
    "type": "object",
    "title": "served_request",
    "properties": {
        "reasoning": {"type": "string", "description": "One or two sentences of justification."},
        "served_request": {"type": "integer", "enum": [0, 1], "description": "1 if served, 0 if not."},
    },
    "required": ["reasoning", "served_request"],
}


def judge_model(model_name: str) -> dict:
    """The serialized LangChain chat model the platform expects.

    The API key is a reference resolved from your workspace secrets, not a value.
    """
    return {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
        "kwargs": {
            "model_name": model_name,
            "temperature": 0,
            "max_tokens": 500,
            "max_retries": 0,
            "openai_api_key": {"lc": 1, "type": "secret", "id": ["OPENAI_API_KEY"]},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("LANGSMITH_PROJECT", "chatgpt-app-tracing"))
    ap.add_argument("--name", default="thread-served-customer-request")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--sampling-rate", type=float, default=1.0)
    ap.add_argument("--backfill-hours", type=int, default=0, help="also score threads idle within this many hours")
    ap.add_argument("--spend-limit-usd", type=float, default=1.0)
    args = ap.parse_args()

    endpoint = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").rstrip("/")
    headers = {"x-api-key": os.environ["LANGSMITH_API_KEY"]}

    with httpx.Client(base_url=endpoint, headers=headers, timeout=60) as http:
        sessions = http.get("/api/v1/sessions", params={"name": args.project}).raise_for_status().json()
        session_id = (sessions if isinstance(sessions, list) else [sessions])[0]["id"]

        body = {
            "display_name": args.name,
            "session_id": session_id,
            "sampling_rate": args.sampling_rate,
            "group_by": "thread_id",  # this is what makes it a thread evaluator
            "extend_evaluator_trace_retention": True,
            "spend_limit": {"limit_usd": args.spend_limit_usd, "window": "weekly"},
            "evaluators": [
                {
                    "structured": {
                        "prompt": [["system", JUDGE_SYSTEM], ["human", "{conversation}"]],
                        "template_format": "f-string",
                        "schema": JUDGE_SCHEMA,
                        "model": judge_model(args.model),
                        # all_messages is the assembled conversation for the thread
                        "variable_mapping": {"conversation": "all_messages"},
                    }
                }
            ],
        }
        if args.backfill_hours:
            body["backfill_from"] = (
                datetime.now(timezone.utc) - timedelta(hours=args.backfill_hours)
            ).isoformat()

        r = http.post("/api/v1/runs/rules", json=body)
        if r.status_code >= 400:
            print(f"failed {r.status_code}\n{r.text[:1500]}")
            raise SystemExit(1)
        rule = r.json()

    print(f"created thread evaluator {rule['display_name']!r}")
    print(f"  rule id      {rule['id']}")
    print(f"  project      {rule.get('session_name')}")
    print(f"  group_by     {rule.get('group_by')}")
    print(f"  feedback key {rule.get('evaluator_name') or args.name}")
    if rule.get("backfill_id"):
        print(f"  backfill     {rule['backfill_id']} status {rule.get('backfill_status')}")
    print("\nThreads are scored once idle, 10 minutes by default. Feedback lands on one")
    print("representative trace per thread, not on every tool call in it.")


if __name__ == "__main__":
    main()
