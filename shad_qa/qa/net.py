"""Small translatable network for the QA pilot.

Inputs are named (the solver's 1-5 ratings, question length, category), each also in its "low" form, standardized.
Layer 1 is non-negative and L1-sparse, so every hidden unit is a short positive combination of named inputs (a
nameable shared concept); softplus keeps it smooth. One head per behavior gives P(correct | behavior). Hidden size and
L1 are chosen by 5-fold CV log loss on train.
  .venv/bin/python qa/net.py
Prints: CV choice, test accuracy of the network used as a router (pick the behavior with the highest predicted
success) next to the best fixed behavior and the oracle, and the weights of both layers.
"""
import os

import numpy as np, pandas as pd
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f"{ROOT}/results/gemma_qa/mmlupro"
FEATS = ["calculation", "steps", "knowledge", "ambiguity", "confidence"]


def load(split, cats=None, mu=None, sd=None):
    F = pd.read_csv(f"{D}/{split}/features.csv").set_index("qid")
    R = pd.read_csv(f"{D}/{split}/runs.csv")
    Y = R.pivot_table(index="qid", columns="action", values="correct").loc[F.index]
    X = F[FEATS].fillna(F[FEATS].median()).copy()
    X["question length"] = np.log(F.q_words)
    cats = cats or sorted(F.category.unique())
    for c in cats:
        X[f"category {c}"] = (F.category == c).astype(float)
    mu = X.mean() if mu is None else mu; sd = X.std().replace(0, 1) if sd is None else sd
    Z = (X - mu) / sd
    Z = pd.concat([Z, (-Z).add_prefix("low ")], axis=1)  # "more X" and "less X" are both nameable, both non-negative
    return Z, Y, F.category, cats, mu, sd


sig = lambda t: 1 / (1 + np.exp(-t))
spl = lambda t: np.logaddexp(0, t)


def fit(X, Y, H, l1, l2=1e-3, seed=0):
    n, d = X.shape; A = Y.shape[1]; M = ~np.isnan(Y); Y0 = np.nan_to_num(Y); N = M.sum()
    shapes = [(d, H), (H,), (H, A), (A,)]; sizes = [int(np.prod(s)) for s in shapes]

    def unpack(p):
        out, i = [], 0
        for s, k in zip(shapes, sizes):
            out.append(p[i:i + k].reshape(s)); i += k
        return out

    def f(p):
        W1, b1, V, c = unpack(p)
        pre = X @ W1 + b1; h = spl(pre); lo = h @ V + c
        ll = np.where(M, np.logaddexp(0, lo) - Y0 * lo, 0).sum() / N
        g = np.where(M, sig(lo) - Y0, 0) / N
        dV = h.T @ g + 2 * l2 * V; dc = g.sum(0); dpre = (g @ V.T) * sig(pre)
        return (ll + l1 * W1.sum() + l2 * (V ** 2).sum(),
                np.concatenate([(X.T @ dpre + l1).ravel(), dpre.sum(0), dV.ravel(), dc]))

    rng = np.random.default_rng(seed)
    p0 = np.concatenate([rng.uniform(0, .3, sizes[0]), np.zeros(H), rng.normal(0, .3, sizes[2]), np.zeros(A)])
    bounds = [(0, None)] * sizes[0] + [(None, None)] * (sum(sizes) - sizes[0])
    p = minimize(f, p0, jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=3000)).x
    W1, b1, V, c = unpack(p)
    return dict(W1=W1, b1=b1, V=V, c=c, prob=lambda Xn: sig(spl(Xn @ W1 + b1) @ V + c))


def logloss(P, Y):
    M = ~np.isnan(Y); P = np.clip(P, 1e-6, 1 - 1e-6)
    return -(np.nan_to_num(Y) * np.log(P) + (1 - np.nan_to_num(Y)) * np.log(1 - P))[M].mean()


if __name__ == "__main__":
    Xtr, Ytr, Ctr, cats, mu, sd = load("train")
    Xte, Yte, Cte, *_ = load("test", cats, mu, sd)
    acts = list(Ytr.columns); xtr, ytr = Xtr.to_numpy(), Ytr.to_numpy(float)
    folds = np.random.default_rng(0).integers(0, 5, len(xtr))
    grid = [(H, l1) for H in (1, 2, 3, 4) for l1 in (1e-3, 3e-3, 1e-2)]
    cv = {}
    for H, l1 in grid:
        cv[(H, l1)] = np.mean([logloss(fit(xtr[folds != k], ytr[folds != k], H, l1)["prob"](xtr[folds == k]), ytr[folds == k])
                               for k in range(5)])
    H, l1 = min(cv, key=cv.get)
    print("CV log loss:", {f"H={h},l1={l}": round(v, 4) for (h, l), v in cv.items()})
    print(f"chosen H={H}, l1={l1}")
    net = fit(xtr, ytr, H, l1)

    yte = Yte.to_numpy(float); P = net["prob"](Xte.to_numpy())
    route = yte[np.arange(len(yte)), P.argmax(1)]
    fixed = {a: np.nanmean(yte[:, i]) for i, a in enumerate(acts)}
    best = max(fixed, key=fixed.get)
    print(f"\ntest accuracy x100: router {100 * np.nanmean(route):.1f} | best fixed ({best}) {100 * fixed[best]:.1f} | "
          f"oracle {100 * np.nanmean(np.nanmax(yte, 1)):.1f} | test log loss {logloss(P, yte):.4f}")
    print("router picks:", pd.Series(np.array(acts)[P.argmax(1)]).value_counts().to_dict())
    byc = pd.DataFrame(dict(cat=Cte.to_numpy(), router=route, fixed=yte[:, acts.index(best)])).groupby("cat").mean() * 100
    print("per category (router vs best fixed):", {c: f"{r.router:.0f}/{r.fixed:.0f}" for c, r in byc.iterrows()})

    print("\nlayer 1 (input -> concept, non-negative; only weights >= 10% of the unit's largest shown)")
    for j in range(H):
        w = pd.Series(net["W1"][:, j], index=Xtr.columns).sort_values(ascending=False)
        print(f"  concept {j + 1}: " + ", ".join(f"{k} {v:.2f}" for k, v in w[w >= .1 * w.iloc[0]].items()))
    print("layer 2 (concept -> success logit per behavior), and intercepts")
    print(pd.DataFrame(np.vstack([net["V"], net["c"]]), index=[f"concept {j + 1}" for j in range(H)] + ["intercept"],
                       columns=acts).round(2).to_string())
    np.savez(f"{D}/net.npz", W1=net["W1"], b1=net["b1"], V=net["V"], c=net["c"], inputs=np.array(Xtr.columns),
             acts=np.array(acts), mu=mu.to_numpy(), sd=sd.to_numpy())
