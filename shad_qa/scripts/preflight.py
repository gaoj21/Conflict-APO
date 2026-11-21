"""Preflight for run_pipeline_llm.sh: import the harness and the llm package, make one real call, check that the
datasets load. Prints the results folder (GEM_OUT) on stdout; every message and error goes to stderr."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../baselines")
err = lambda *a: print(*a, file=sys.stderr, flush=True)
try:
    import gem
except Exception as e:  # noqa: BLE001
    err(f"[preflight] cannot import the harness / llm package: {e}"); sys.exit(1)
err(f"[preflight] provider: GEM_PROVIDER={gem.PROVIDER}  model label: {gem.JUDGE}  results: {gem.OUT}")
for n in os.environ.get("PREFLIGHT_DS", "hs2 ell merlin").split():
    try:
        d = gem.DS(n); err(f"[preflight] dataset {n}: fit {len(d.rows['fit'])} / sel {len(d.rows['sel'])} / test {len(d.rows['test'])}")
    except Exception as e:  # noqa: BLE001
        err(f"[preflight] dataset {n} does not load: {e}"); sys.exit(1)
t = time.time()
try:
    txt, it, ot, _ = gem.chat([dict(role="user", content="Reply with the single word OK.")], temperature=0.0, max_tokens=20, cache=False)
except Exception as e:  # noqa: BLE001
    err(f"[preflight] the test call failed: {e}"); sys.exit(1)
err(f"[preflight] test call OK in {time.time() - t:.1f}s: reply {txt.strip()[:40]!r}, tokens in/out {it}/{ot}")
os.makedirs(f"{gem.OUT}/logs", exist_ok=True)
print(gem.OUT)
