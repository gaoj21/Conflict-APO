"""Smoke test of the configured provider: real, billed calls; prints what came back, including usage.
  set -a; . ./.env; set +a
  python llm/examples_terminal.py
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import UsageTracker, batch_result, chat_result, default_provider, list_providers  # noqa: E402

for p in list_providers():
    print(("* " if p["default"] else "  ") + f"{p['name']:<12} {str(p['type']):<18} {str(p['model']):<32} "
          f"available={p['available']} missing_env={p['missing_env']}")
print("default provider:", default_provider())
t = UsageTracker()
r = chat_result(None, [{"role": "user", "content": "Reply with exactly: hello"}], temperature=0)
t.add(r)
print("chat_result:", repr(r.content[:200]), "| provider:", r.provider, "| model:", r.model, "| usage:", r.usage)
rs = batch_result(None, [[{"role": "user", "content": f"What is {i}+{i}? Reply with the number only."}] for i in range(3)],
                  temperature=0)
for r in rs:
    t.add(r)
print("batch_result:", [r.content.strip()[:20] for r in rs])
print("totals:", t.summary())
