"""Attribute assignment + merge on sel. Run after scripts/claude_opt.py ROOT text. The optimizer saw the fit items'
errors but only aggregate scores on sel, so sel serves both to accept edits and to assign attributes; its only bias
favours the versions it accepted (the final prompt), which makes the method conservative.

  1. every archive version (v0 + all candidates of the text run, kept or rejected) is already judged on sel;
  2. per attribute k and version v: d_vk = advantage of v's readout over the final prompt's readout (5-fold within-fold
     CV Spearman on sel, paired), se_vk = paired-bootstrap SE; empirical-Bayes shrinkage toward 0 with
     tau2 = max(var(d) - mean(se^2), 0): d*_vk = d_vk tau2 / (tau2 + se_vk^2); route k to argmax_v d*_vk (final: 0);
     a non-final source is kept only if the shrunk gain it brings, summed over its attributes, exceeds COST (the price
     of carrying one more source, as in scripts/claude_portfolio.py); otherwise its attributes go back to the final;
  3. if any attribute left the final prompt: Sonnet merges the sources into MERGES single prompts (judged on fit, sel,
     test); the deployed prompt is the best of {final, merged...} by sel readout Spearman (readout fitted on fit), so
     keeping the final needs no threshold;
  4. test: readout fitted on fit + sel, for v0, final, each merged prompt, the deployed one, and the routed
     sources themselves (one call per distinct source).

  python scripts/claude_routemerge.py results/claude_hs2r [--workers 10]
"""
import glob, json, os, re, subprocess, sys

import numpy as np, pandas as pd
from scipy.stats import rankdata

ARGV = sys.argv[1:]
sys.argv = sys.argv[:1] + [ARGV[0]] + ARGV[1:]
__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_portfolio.py")
exec(open(__file__).read().split('if __name__ == "__main__":')[0]
     .replace('sys.argv[:1] + [ARGV[0], "portfolio"] + ARGV[1:]', 'sys.argv[:1] + [ARGV[0], "routemerge"] + ARGV[1:]'))
exec(open(os.path.join(os.path.dirname(__file__), "claude_merge.py")).read().split("MERGE_SYS = ")[1].split("def main")[0]
     .join(["MERGE_SYS = ", ""]))  # MERGE_SYS / MERGE_TASK prompts
TEXT = f"{R}/opt/text"
FOLDS, B, MERGES = 5, 200, 2
A_J = [x + "_j" for x in A]


def base(n):
    return f"{R}/out" if n == "v0" else f"{TEXT}/out"


def prompt_of(n):
    return open(f"{R}/prompts/v0.txt" if n == "v0" else f"{TEXT}/prompts/{n}.txt").read()


def load(n, split, b=None):
    return frame(pd.read_csv(f"{b or base(n)}/{n}_{split}.csv"), split)


def readout_eval(n, train_splits, ev_split, b=None):
    mt = {"fit": pd.concat([load(n, s, b) for s in train_splits], ignore_index=True), "sel": load(n, ev_split, b)}
    T_ = table([mt], "sel")
    return per_task(T_[A].to_numpy(float), decoder(table([mt], "fit"), 1)(T_))


def rs(x, y):
    return np.corrcoef(rankdata(x), rankdata(y))[0, 1]


def main():
    global O
    names = ["v0"] + sorted({os.path.basename(f)[:-8] for f in glob.glob(f"{TEXT}/out/s*c*_sel.csv")})
    fin = open(f"{TEXT}/final.txt").read().strip()
    O_me, O = O, TEXT  # cache archive judgments with the text run
    O = TEXT
    judge_rows(fin, prompt_of(fin), "test")
    O = O_me
    # ---- route
    ids = sorted(set.intersection(*[set(load(n, "sel").id) for n in names]))
    OOF = {}
    for n in names:
        d = load(n, "sel").set_index("id").loc[ids].reset_index()
        fold = np.arange(len(d)) % FOLDS
        P = np.zeros((len(d), K))
        for f in range(FOLDS):
            mt = {"fit": d[fold != f].reset_index(drop=True), "sel": d[fold == f].reset_index(drop=True)}
            P[fold == f] = decoder(table([mt], "fit"), 1)(table([mt], "sel"))
        OOF[n] = P
        H = d[A].to_numpy(float)
    rng = np.random.default_rng(0)
    boots = [np.concatenate([rng.choice(np.flatnonzero(fold == f), (fold == f).sum()) for f in range(FOLDS)]) for _ in range(B)]
    cv = lambda P, k, ix: np.nanmean([rs(H[ix][fold[ix] == f, k], P[ix][fold[ix] == f, k]) for f in range(FOLDS)])
    allix = np.arange(len(H))
    vs = [n for n in names if n != fin]
    d = np.array([[cv(OOF[n], k, allix) - cv(OOF[fin], k, allix) for k in range(K)] for n in vs])
    se = np.array([[np.std([cv(OOF[n], k, b) - cv(OOF[fin], k, b) for b in boots]) for k in range(K)] for n in vs])
    tau2 = max(np.var(d) - np.mean(se ** 2), 0.0)
    shr = d * tau2 / (tau2 + se ** 2)
    route = [vs[int(np.argmax(shr[:, k]))] if shr[:, k].max() > 0 else fin for k in range(K)]
    for s_ in set(route) - {fin}:
        ks = [k for k in range(K) if route[k] == s_]
        if sum(shr[vs.index(s_), k] for k in ks) <= COST:
            for k in ks:
                route[k] = fin
    log(f"[routemerge] final {fin}; tau {np.sqrt(tau2):.4f}, typical se {np.sqrt(np.mean(se ** 2)):.4f}; routing: " +
        ", ".join(f"{A[k]} <- {route[k]} (d {d[vs.index(route[k]), k] if route[k] != fin else 0:+.3f}, "
                  f"d* {shr[vs.index(route[k]), k] if route[k] != fin else 0:+.4f})" for k in range(K)))
    srcs = list(dict.fromkeys(route))
    if os.environ.get("ASSIGN_ONLY"):  # stop after the assignment (offline check)
        return
    # ---- merge
    merged = []
    if srcs != [fin]:
        path = f"{O}/prompts/merge_opt.txt"
        if not os.path.exists(path):
            versions = "\n\n".join(f"<version name=\"{n}\">\n{prompt_of(n).strip()}\n</version>" for n in srcs)
            routing = "\n".join(f"- {a_}: use version {r}" for a_, r in zip(A, route))
            system = MERGE_SYS.format(attrs=", ".join(A), lo=LO, hi=HI)
            user = MERGE_TASK.format(versions=versions, routing=routing, n=MERGES)
            txt = ""
            for _ in range(3):
                p = subprocess.run(["claude", "-p", "--model", a.optimizer, "--output-format", "json", "--no-session-persistence",
                                    "--tools", "", "--setting-sources", "", "--system-prompt", system],
                                   input=user, capture_output=True, text=True, timeout=900, cwd="/tmp")
                try:
                    o = json.loads(p.stdout)
                except Exception:  # noqa: BLE001
                    continue
                if o.get("is_error"):
                    raise Abort("merger: " + str(o.get("result", ""))[:300])
                txt = o.get("result", "")
                if "<prompt1>" in txt:
                    break
            open(path, "w").write(txt)
            open(f"{O}/prompts/merge_input.txt", "w").write(system + "\n\n=====\n\n" + user)
        merged = [c.strip() for c in re.findall(r"<prompt\d>(.*?)</prompt\d>", open(path).read(), re.S)][:MERGES]
        for i, c in enumerate(merged):
            open(f"{O}/prompts/m{i}.txt", "w").write(c)
            for s_ in ("fit", "sel", "test"):
                judge_rows(f"m{i}", c, s_)
        O_me = O
        for n in srcs:
            O = R if n == "v0" else TEXT
            judge_rows(n, prompt_of(n), "test")
        O = O_me
    # ---- deploy: best of final / merged by sel readout (fitted on fit)
    cands = {fin: TEXT + "/out"} | {f"m{i}": f"{O}/out" for i in range(len(merged))}
    rmean = {n: float(np.nanmean(readout_eval(n, ["fit"], "sel", b))) for n, b in cands.items()}
    dep = max(rmean, key=rmean.get)
    log(f"[routemerge] sel readout means {rmean} -> deploy {dep}")
    # ---- test
    rows = {"v0 + readout (1 call)": readout_eval("v0", ["fit", "sel"], "test"),
            f"text final {fin} + readout (1 call)": readout_eval(fin, ["fit", "sel"], "test")}
    for i in range(len(merged)):
        rows[f"merged m{i} + readout (1 call)"] = readout_eval(f"m{i}", ["fit", "sel"], "test", f"{O}/out")
    rows[f"DEPLOYED {dep} (1 call)"] = rows[[k for k in rows if f" {dep} " in k][0]]
    if srcs != [fin]:
        rr = {n: readout_eval(n, ["fit", "sel"], "test") for n in srcs}
        rows[f"routed sources ({len(srcs)} calls)"] = np.array([rr[route[k]][k] for k in range(K)])
    out = [f"=== {os.path.basename(R)} assign-and-merge (on sel), test: {' '.join(x[:6] for x in A)} | mean"]
    for lab, v in rows.items():
        out.append(f"  {lab:38s} " + " ".join(f"{x:6.3f}" for x in v) + f" | {np.nanmean(v):.3f}")
    print("\n".join(out)); open(f"{O}/result.txt", "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Abort as e:
        log(f"ABORT {e}")
        sys.exit(2)
