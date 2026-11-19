"""Compact progress table of run_pipeline_llm.sh (all details stay in $GEM_OUT/logs/*.log).
  python scripts/progress.py [ds ...]            print once
  python scripts/progress.py --watch 30 [ds ...] refresh every 30 s until the pipeline writes PIPELINE_DONE
Columns: v0 judged / items dropped; TextGrad x ShAD rounds and plain TextGrad (step x/12); then test Spearman x100
(no decoder) of TextGrad + format, + length rules, + text rules, + combined as each finishes.
"""
import json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../baselines")
import gem  # noqa: E402

O = gem.OUT; L = f"{O}/logs"


def tg(log):
    if not os.path.exists(log):
        return "waiting"
    t = open(log).read()
    if "Traceback" in t or "Abort" in t:
        return "FAILED (see log)"
    if "done, spent" in t:
        return "done"
    s = re.findall(r"step (\d+) val MAE", t)
    return f"step {s[-1] if s else 0}/12"


def num(f, key):
    try:
        return f"{json.load(open(f))[key]:.1f}"
    except Exception:  # noqa: BLE001
        return "running" if os.path.exists(os.path.dirname(f)) else "-"


def v0(d):
    if not os.path.exists(f"{O}/{d}/v0/seed0/scores_test.csv"):
        return "judging" if os.path.exists(f"{L}/{d}_v0.log") else "waiting"
    f = f"{O}/{d}/excluded_ids.txt"
    n = len(open(f).read().split()) if os.path.exists(f) else 0
    return f"done, {n} dropped"


def table(ds):
    rows = [("dataset", "v0 / screen", "TG x ShAD rounds", "plain TG", "+format", "+length", "+text", "+combined")]
    for d in ds:
        p = f"{O}/{d}/postrules_textgrad_ccc/seed0/result.json"
        rows.append((d, v0(d), tg(f"{L}/{d}_tg_shadrules.log"), tg(f"{L}/{d}_textgrad_ccc.log"),
                     num(p, "base"), num(p, "rules"),
                     num(f"{O}/{d}/llmrules_textgrad_ccc/seed0/result.json", "text_rules"),
                     num(f"{O}/{d}/llmrules_comb_textgrad_ccc/seed0/result.json", "text_rules")))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows)


args = sys.argv[1:]
every = None
if args[:1] == ["--watch"]:
    every, args = float(args[1]), args[2:]
ds = args or ["hs2", "ell", "merlin"]
while True:
    out = f"{time.strftime('%H:%M:%S')}  results: {O}\n{table(ds)}\n(test Spearman x100, raw judge, no decoder; logs in {L})"
    if every is None:
        print(out); break
    print("\033[2J\033[H" + out, flush=True)  # clear the screen and redraw
    if os.path.exists(f"{L}/PIPELINE_DONE"):
        break
    time.sleep(every)
