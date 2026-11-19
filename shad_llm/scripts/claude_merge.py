"""Route-and-merge: turn the archive of a shared-prompt optimization run (v0 + every candidate, kept or rejected, all
already judged on fit and sel) into ONE judge prompt.

  1. route: per attribute, the archive version whose own score (through the single-prompt readout: anchored ridge
     with length, scripts/claude_portfolio.py decoder) agrees best with humans, by 5-fold within-fold CV Spearman on
     fit + sel pooled;
  2. merge: Sonnet writes one prompt that takes each attribute's criteria from its source version (MERGES variants);
  3. evaluate on test: each merged prompt + its readout (1 call), the text run's final prompt + readout (1 call), and
     the routed sources themselves (attribute k read from its source's readout; one call per distinct source).

  python scripts/claude_merge.py results/claude_br [--merges 2] [--workers 10]
"""
import glob, json, os, re, subprocess, sys

import numpy as np, pandas as pd

ARGV = sys.argv[1:]
sys.argv = sys.argv[:1] + [ARGV[0]] + ARGV[1:]
__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_portfolio.py")
_src = open(__file__).read().split('if __name__ == "__main__":')[0]
_src = _src.replace('sys.argv[:1] + [ARGV[0], "portfolio"] + ARGV[1:]', 'sys.argv[:1] + [ARGV[0], "merge"] + [x for x in ARGV[1:] if not x.startswith("--merges")]')
exec(_src)
TEXT = f"{R}/opt/text"
MERGES = int(next((x.split("=")[1] for x in ARGV if x.startswith("--merges=")), 2))


def load(name, split):
    f = f"{R}/out/v0_{split}.csv" if name == "v0" else f"{TEXT}/out/{name}_{split}.csv"
    return frame(pd.read_csv(f), split)


def prompt_of(name):
    return open(f"{R}/prompts/v0.txt" if name == "v0" else f"{TEXT}/prompts/{name}.txt").read()


def cv_scores(name, folds=5):
    """Per-attribute within-fold CV Spearman of the single-prompt readout on fit + sel pooled."""
    d = pd.concat([load(name, "fit"), load(name, "sel")], ignore_index=True)
    fold = np.arange(len(d)) % folds
    H = d[A].to_numpy(float)
    out = []
    for f in range(folds):
        tr, te = d[fold != f].reset_index(drop=True), d[fold == f].reset_index(drop=True)
        mt = {"fit": tr, "sel": te}
        out.append(per_task(H[fold == f], decoder(table([mt], "fit"), 1)(table([mt], "sel"))))
    return np.nanmean(out, 0)


MERGE_SYS = """You merge several versions of the instructions (system prompt) of an LLM judge into ONE set of instructions. The judge reads one item and rates it on several attributes at once ({attrs}; integers {lo}-{hi}). Each version was found to judge some attributes better than the others. For every attribute you are told which version judges it best: the merged instructions must carry over THAT version's criteria, definitions, examples and procedure for that attribute as faithfully as possible (reword only to fit them together), while the shared framing stays short and neutral. Do not add new criteria of your own. The output-format section is appended automatically; do not write one. Keep the attribute names and the {lo}-{hi} scale."""

MERGE_TASK = """{versions}

<routing>
{routing}
</routing>

Write {n} different merged versions (e.g. one that keeps the source wording almost verbatim per attribute, one that integrates them more tightly). Reply with a one-line rationale for each, then the versions as <prompt1>...</prompt1>, <prompt2>...</prompt2>, ..."""


def main():
    global O
    names = ["v0"] + sorted({os.path.basename(f)[:-8] for f in glob.glob(f"{TEXT}/out/s*c*_sel.csv")})
    S = {n: cv_scores(n) for n in names}
    route = [max(names, key=lambda n: S[n][k]) for k in range(K)]
    log(f"[merge] routing (fit+sel CV): " + ", ".join(f"{a_} <- {r} ({S[r][k]:.3f}; v0 {S['v0'][k]:.3f})" for k, (a_, r) in enumerate(zip(A, route))))
    srcs = list(dict.fromkeys(route))
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
            log(f"  merger ${o.get('total_cost_usd', 0):.3f}")
            if "<prompt1>" in txt:
                break
        open(path, "w").write(txt)
        open(f"{O}/prompts/merge_input.txt", "w").write(system + "\n\n=====\n\n" + user)
    txt = open(path).read()
    merged = [c.strip() for c in re.findall(r"<prompt\d>(.*?)</prompt\d>", txt, re.S)][:MERGES]
    if not merged:
        raise Abort("merger reply without prompts")
    # judge merged prompts on fit / sel / test; routed sources on test (into the text run's cache)
    for i, c in enumerate(merged):
        open(f"{O}/prompts/m{i}.txt", "w").write(c)
        for s_ in ("fit", "sel", "test"):
            judge_rows(f"m{i}", c, s_)
    O_merge, O = O, TEXT
    fin = open(f"{TEXT}/final.txt").read().strip()
    for n in dict.fromkeys(srcs + [fin]):
        judge_rows(n, prompt_of(n), "test")
    O = O_merge
    # ---- test evaluation
    G = GOLD[GOLD.split == "test"]
    def readout(name, base):
        tr = pd.concat([frame(pd.read_csv(f"{base}/{name}_fit.csv"), "fit"), frame(pd.read_csv(f"{base}/{name}_sel.csv"), "sel")], ignore_index=True)
        te = frame(pd.read_csv(f"{base}/{name}_test.csv"), "test")
        mt = {"fit": tr, "sel": te}
        T_ = table([mt], "sel")
        return T_.id.to_numpy(), decoder(table([mt], "fit"), 1)(T_), T_[A].to_numpy(float)
    base = lambda n: f"{R}/out" if n == "v0" else f"{TEXT}/out"
    rows = {}
    for n, lab in (("v0", "v0 + readout (1 call)"), (fin, f"text final {fin} + readout (1 call)")):
        ids, P, H = readout(n, base(n)); rows[lab] = per_task(H, P)
    sel_means = {}
    for i in range(len(merged)):
        ids, P, H = readout(f"m{i}", f"{O}/out"); rows[f"merged m{i} + readout (1 call)"] = per_task(H, P)
        mt = {"fit": frame(pd.read_csv(f"{O}/out/m{i}_fit.csv"), "fit"), "sel": frame(pd.read_csv(f"{O}/out/m{i}_sel.csv"), "sel")}
        T_ = table([mt], "sel")
        sel_means[i] = float(np.nanmean(per_task(T_[A].to_numpy(float), decoder(table([mt], "fit"), 1)(T_))))
    # routed sources: attribute k from its source's readout (on the test rows all sources judged)
    R_ = {n: readout(n, base(n)) for n in srcs}
    common = set.intersection(*[set(R_[n][0]) for n in srcs])
    r = []
    for k, n in enumerate(route):
        ids, P, H = R_[n]; m = np.isin(ids, list(common))
        r.append(per_task(H[m], P[m])[k])
    rows[f"routed sources ({len(srcs)} calls)"] = np.array(r)
    print(f"=== {os.path.basename(R)} route-and-merge, test: {' '.join(x[:6] for x in A)} | mean")
    for lab, v in rows.items():
        print(f"  {lab:38s} " + " ".join(f"{x:6.3f}" for x in v) + f" | {np.nanmean(v):.3f}")
    best = max(sel_means, key=sel_means.get)
    print(f"  merged prompt chosen on sel: m{best} (sel readout means {sel_means})")


if __name__ == "__main__":
    try:
        main()
    except Abort as e:
        log(f"ABORT {e}")
        sys.exit(2)
