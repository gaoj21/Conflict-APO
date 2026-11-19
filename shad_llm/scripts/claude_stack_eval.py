"""Conflict repair by stacking prompt versions: the scores of several prompt versions of the same judge (v0 + versions
from an optimization run) go into one anchored readout per attribute (fitted on fit + sel; ridge shrunk toward an
anchor, anchor and shrinkage chosen together by within-fold CV Spearman). Anchors: each version's own score alone,
or the average of all versions' own scores. An attribute that one version hurt can lean on the others.

  python scripts/claude_stack_eval.py results/claude_br text s1c2 s1c1 ...   (versions besides v0, first = final)
"""
import itertools, os, sys
import numpy as np, pandas as pd

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1] + [ARGS[0], "x"]
exec(open(os.path.join(os.path.dirname(__file__), "claude_opt.py")).read().split("# ---------------------------------------------------------------- main loop")[0])
R, ARM, VERS = ARGS[0], ARGS[1], ["v0"] + ARGS[2:]
J = [x + "_j" for x in A]
K = len(A)


def load(split):
    out = None
    for v in VERS:
        f = f"{R}/out/v0_{split}.csv" if v == "v0" else f"{R}/opt/{ARM}/out/{v}_{split}.csv"
        d = frame(pd.read_csv(f), split)[["id", "z"] + A + J].rename(columns={c: f"{v}:{c}" for c in J})
        out = d if out is None else out.merge(d[["id"] + [f"{v}:{c}" for c in J]], on="id")
    return out


tr = pd.concat([load("fit"), load("sel")], ignore_index=True); te = load("test")
H, Y = te[A].to_numpy(float), std(tr[A].to_numpy(float))


def X(d, vers):
    cols = [f"{v}:{c}" for v in vers for c in J]
    m, s = tr[cols].mean().to_numpy(), tr[cols].std().replace(0, 1).to_numpy()
    return np.c_[(d[cols].to_numpy(float) - m) / s, (d.z - tr.z.mean()) / tr.z.std()]


def stack(vers, folds=5, anchors="all"):
    Xtr, Xte = X(tr, vers), X(te, vers)
    fold = np.arange(len(Xtr)) % folds
    out = np.zeros((len(Xte), K))
    for k in range(K):
        own = [i * K + k for i in range(len(vers))]
        E = []
        if anchors in ("all", "single"):
            for j in own:
                e = np.zeros(Xtr.shape[1]); e[j] = 1; E.append(e)
        if anchors in ("all", "avg") and len(vers) > 1:
            e = np.zeros(Xtr.shape[1]); e[own] = 1 / len(own); E.append(e)
        y = Y[:, k]

        def fit(Xa, ya, e, lam):
            xe = Xa @ e
            w0 = e * (xe @ ya) / max(xe @ xe, 1e-9)
            return w0 if lam is None else np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ ya + lam * w0)

        opts = list(itertools.product(range(len(E)), LAMS + [None]))
        cv = [np.nanmean([sp(y[fold == f], Xtr[fold == f] @ fit(Xtr[fold != f], y[fold != f], E[e], lam)) for f in range(folds)])
              for e, lam in opts]
        e, lam = opts[int(np.nanargmax(cv))]
        out[:, k] = Xte @ fit(Xtr, y, E[e], lam)
    return out


def row(lab, P, ref=None):
    r = per_task(H, P)
    print(f"  {lab:34s} " + " ".join(f"{x:6.3f}" for x in r) + f" | {np.nanmean(r):.3f}")


print(f"=== {os.path.basename(R)} {ARM} test n={len(te)}: {' '.join(x[:6] for x in A)} | mean   (versions {VERS})")
row("v0 raw", te[[f"v0:{c}" for c in J]].to_numpy(float))
for v in VERS:
    row(f"{v} readout (1 call)", stack([v]))
fin = VERS[1]
row(f"v0+{fin} stack, anchor {fin} only", stack(["v0", fin], anchors="single"))
row(f"v0+{fin} stack, anchor avg", stack(["v0", fin], anchors="avg"))
row(f"v0+{fin} stack, anchor CV (2 calls)", stack(["v0", fin]))
for n in range(3, len(VERS) + 1):
    row(f"{'+'.join(VERS[:n])} stack ({n} calls)", stack(VERS[:n]))
