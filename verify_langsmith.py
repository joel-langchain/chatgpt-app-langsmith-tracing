"""Read back what landed in LangSmith and check the threading worked.

Run after simulate_chatgpt.py:
    uv run verify_langsmith.py
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

project = os.environ["LANGSMITH_PROJECT"]
client = Client(api_key=os.environ["LANGSMITH_API_KEY"], api_url=os.environ.get("LANGSMITH_ENDPOINT"))

since = datetime.now(timezone.utc) - timedelta(minutes=int(os.environ.get("VERIFY_MINUTES", "30")))
runs = list(client.list_runs(project_name=project, is_root=True, start_time=since))  # list_runs is deprecated in langsmith 0.12, fine for a read-back
print(f"{len(runs)} root runs in '{project}' since {since:%H:%M} UTC\n")

by_session: dict[str, list] = defaultdict(list)
for r in runs:
    md = (r.extra or {}).get("metadata", {})
    by_session[md.get("session_id", "<no session_id>")].append(r)

for session, rs in by_session.items():
    rs.sort(key=lambda r: r.start_time)
    user = {(r.extra or {}).get("metadata", {}).get("user_id") for r in rs}
    print(f"thread {session}  user {','.join(str(u) for u in user)}  ({len(rs)} tool calls)")
    for r in rs:
        status = "error" if r.error or r.status == "error" else "ok"
        ms = (r.end_time - r.start_time).total_seconds() * 1000 if r.end_time else float("nan")
        print(f"   {r.start_time:%H:%M:%S}  {r.run_type:<6} {r.name:<20} {status:<5} {ms:6.1f} ms  in={str(r.inputs)[:60]}")
    print()

ok = all(s != "<no session_id>" for s in by_session)
print("PASS: every root run carries a session_id" if ok and runs else "CHECK: see above")
