"""Multi-view decoder: the judge = M prompts from the optimization archive (v0 + every candidate of the text run, kept
or rejected) + one per-attribute decoder over all their scores and length (claude_portfolio.decoder). Claim: at M calls
it beats text final + readout, and beats M calls of the final prompt itself (self-ensemble, same decoder).

  1. greedy on sel (decoder fitted on fit): start from the final prompt, add the archive version that raises the
     mean decoded Spearman on sel most, up to MMAX views (no threshold: every M is reported);
  2. judge the chosen versions on test, and the final prompt MMAX-1 more times on fit/sel/test (self-ensemble);
  3. test: decoder fitted on fit + sel, for M = 1..MMAX, archive views vs self-ensemble views.

  python scripts/claude_multiview.py results/claude_hs2r [--workers 10] [--select-only]
"""
import os, sys
import numpy as np, pandas as pd

ARGV = sys.argv[1:]
SELECT_ONLY = "--select-only" in ARGV
ARGV = [x for x in ARGV if x != "--select-only"]
sys.argv = sys.argv[:1] + [ARGV[0]] + ARGV[1:]
__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_portfolio.py")
exec(open(__file__).read().split('if __name__ == "__main__":')[0]
     .replace('sys.argv[:1] + [ARGV[0], "portfolio"] + ARGV[1:]', 'sys.argv[:1] + [ARGV[0], "text"] + ARGV[1:]'))
MMAX = 3
import glob


def prompt_of(n):
    base = n.split("_r")[0]
    return open(f"{R}/prompts/v0.txt" if base == "v0" else f"{O}/prompts/{base}.txt").read()


def view(n, splits=("fit", "sel")):
    m = {}
    for s in splits:
        f = f"{R}/out/v0_{s}.csv" if n == "v0" else f"{O}/out/{n}_{s}.csv"
        m[s] = frame(pd.read_csv(f) if os.path.exists(f) else judge_rows(n, prompt_of(n), s), s)
    return m


def score(views, tr_split="fit", ev_split="sel"):
    tr, te = table(views, tr_split), table(views, ev_split)
    return per_task(te[A].to_numpy(float), decoder(tr, len(views))(te))


def main():
    fin = open(f"{O}/final.txt").read().strip()
    names = ["v0"] + sorted({os.path.basename(f)[:-8] for f in glob.glob(f"{O}/out/s*c*_sel.csv") if "_r" not in os.path.basename(f)})
    V = {n: view(n) for n in names}
    chosen, curve = [fin], [float(np.nanmean(score([V[fin]])))]
    for _ in range(MMAX - 1):
        rest = [n for n in names if n not in chosen]
        sc = {n: float(np.nanmean(score([V[c] for c in chosen] + [V[n]]))) for n in rest}
        best = max(sc, key=sc.get)
        chosen.append(best); curve.append(sc[best])
    log(f"[multiview] greedy on sel: {chosen}, sel decoded mean {np.round(curve, 4).tolist()}")
    if SELECT_ONLY:
        return
    reruns = [f"{fin}_r{i}" for i in range(1, MMAX)]
    allv = {}
    for n in dict.fromkeys(chosen + reruns):
        allv[n] = view(n, ("fit", "sel", "test"))
    def test(views):
        tr = pd.concat([table(views, "fit"), table(views, "sel")], ignore_index=True)
        te = table(views, "test")
        return per_task(te[A].to_numpy(float), decoder(tr, len(views))(te))
    out = [f"=== {os.path.basename(R)} multi-view decoder, test: {' '.join(x[:6] for x in A)} | mean"]
    for M in range(1, MMAX + 1):
        r = test([allv[n] for n in chosen[:M]])
        out.append(f"  archive views {chosen[:M]!s:34s} ({M} calls) " + " ".join(f"{x:6.3f}" for x in r) + f" | {np.nanmean(r):.3f}")
    for M in range(2, MMAX + 1):
        vs = [allv[fin]] + [allv[n] for n in reruns[:M - 1]]
        r = test(vs)
        out.append(f"  self-ensemble {fin} x{M}{'':22s} ({M} calls) " + " ".join(f"{x:6.3f}" for x in r) + f" | {np.nanmean(r):.3f}")
        # same calls, averaged scores as one view
        avg = {s: vs[0][s].copy() for s in ("fit", "sel", "test")}
        for s in avg:
            T_ = table(vs, s)
            avg[s] = T_[["id", "z"] + A].copy()
            for c in J:
                avg[s][c] = T_[[f"{i}:{c}" for i in range(len(vs))]].mean(1).to_numpy()
        r = test([avg])
        out.append(f"  self-ensemble {fin} x{M} averaged{'':13s} ({M} calls) " + " ".join(f"{x:6.3f}" for x in r) + f" | {np.nanmean(r):.3f}")
    print("\n".join(out)); open(f"{R}/multiview_result.txt", "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Abort as e:
        log(f"ABORT {e}")
        sys.exit(2)
