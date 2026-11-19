"""Judge rows with one independent headless Claude call per row (claude -p, no tools, no settings).

  python scripts/claude_judge.py VER --split sel [--root results/claude_loop] [--n 20] [--tag r1] [--workers 8]
writes <root>/out/<VER><tag>_<split>.csv: id, one column per key of the judge's final JSON (PPE: verdict), reason,
in_tok, out_tok, cost. The judge sees only <root>/prompts/<VER>.txt + <root>/format.txt and the row (rows carry a
rendered "user" field, or are PPE pairs); no labels.
"""
import argparse, glob, json, re, subprocess
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="results/claude_loop")
ap.add_argument("ver"); ap.add_argument("--split", default="sel"); ap.add_argument("--n", type=int, default=0)
ap.add_argument("--tag", default=""); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--model", default="haiku")
a = ap.parse_args()
D = a.root
system = open(f"{D}/prompts/{a.ver}.txt").read().strip() + "\n\n" + open(f"{D}/format.txt").read().strip()
rows = [json.loads(l) for f in sorted(glob.glob(f"{D}/chunks/{a.split}_*.jsonl")) for l in open(f)]
rows = rows[:a.n] if a.n else rows


def judge(r):
    user = r.get("user") or (f"## User prompt:\n{r['prompt']}\n\n## Response 1:\n{r['response_1']}\n\n## Response 2:\n{r['response_2']}"
            "\n\n## Your evaluation:")
    for _ in range(3):
        try:
            p = subprocess.run(["claude", "-p", "--model", a.model, "--output-format", "json", "--no-session-persistence",
                                "--tools", "", "--setting-sources", "", "--system-prompt", system],
                               input=user, capture_output=True, text=True, timeout=300, cwd="/tmp")
            o = json.loads(p.stdout)
            txt, u = o.get("result", ""), o.get("usage", {})
            js = re.findall(r"\{[^{}]*\}", txt)
            try:
                sc = {k: float(v) for k, v in json.loads(js[-1]).items()}
            except Exception:  # noqa: BLE001
                sc = {}
            if "preference" in sc:
                sc = {"verdict": int(sc.pop("preference"))}
            return dict(id=r["id"], **sc, reason=" ".join(txt.split())[-300:],
                        in_tok=u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0),
                        out_tok=u.get("output_tokens", 0), cost=o.get("total_cost_usd", 0.0))
        except Exception as e:  # noqa: BLE001
            err = str(e)[:200]
    return dict(id=r["id"], verdict=None, reason="ERROR " + err, in_tok=0, out_tok=0, cost=0.0)


with ThreadPoolExecutor(a.workers) as ex:
    out = pd.DataFrame(list(ex.map(judge, rows)))
path = f"{D}/out/{a.ver}{a.tag}_{a.split}.csv"
out.to_csv(path, index=False)
key = "verdict" if "verdict" in out else [c for c in out if c not in ("id", "reason", "in_tok", "out_tok", "cost")][0]
print(f"{path}: {out[key].notna().sum()}/{len(out)} parsed, in {out.in_tok.mean():.0f} out {out.out_tok.mean():.0f} "
      f"tok/call, ${out.cost.sum():.3f}")
