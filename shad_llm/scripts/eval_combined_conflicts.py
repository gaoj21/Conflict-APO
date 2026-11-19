"""Combined rules (text cross-attribute + numeric length) on final APO prompts: mean test Spearman x100 and per-attribute
changes vs the same prompt with the format change only; an attribute counts as up / down when the change exceeds two
paired-bootstrap SE (test items). A conflict = at least one attribute significantly up and one significantly down."""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../baselines")
import gem  # noqa: E402
from evaluate import sp  # noqa: E402


def mat(ds, f):
    df = ds.worst_case(pd.read_csv(f)).set_index("id")
    return df.loc[[r["id"] for r in ds.rows["test"]], ds.A].to_numpy(float)


rows = []
for n in ["hs2", "hanna", "ell", "merlin"]:
    ds = gem.DS(n); H = ds.human([r["id"] for r in ds.rows["test"]]); K = len(ds.A)
    for b in ("mipro", "textgrad_ccc"):
        d = f"{gem.OUT}/{n}/llmrules_comb_{b}/seed0"
        if not os.path.exists(f"{d}/result.json"):
            continue
        sel = [f for f in glob.glob(f"{d}/c*_test.csv")]
        if not sel:
            continue
        P1 = mat(ds, sel[0]); P0 = mat(ds, f"{gem.OUT}/{n}/postrules_{b}/seed0/scores_test_base.csv")
        orig = mat(ds, f"{gem.OUT}/{n}/{'dspy_mipro' if b == 'mipro' else b}/seed0/scores_test.csv")
        d_att = np.array([sp(H[:, k], P1[:, k]) - sp(H[:, k], P0[:, k]) for k in range(K)])
        rng = np.random.default_rng(0); bs = []
        for _ in range(300):
            i = rng.integers(0, len(H), len(H)); bs.append([sp(H[i, k], P1[i, k]) - sp(H[i, k], P0[i, k]) for k in range(K)])
        se = np.nanstd(np.array(bs), 0)
        up, dn = int((d_att > 2 * se).sum()), int((d_att < -2 * se).sum())
        m = lambda P: 100 * np.nanmean([sp(H[:, k], P[:, k]) for k in range(K)])
        rows.append(dict(ds=n, start=b, original=round(m(orig), 1), format_only=round(m(P0), 1), combined=round(m(P1), 1),
                         sig_up=up, sig_down=dn, conflict=int(up > 0 and dn > 0),
                         per_attr=" ".join(f"{a[:5]}{100 * v:+.1f}" for a, v in zip(ds.A, d_att))))
with pd.option_context("display.width", 250, "display.max_colwidth", 120):
    print(pd.DataFrame(rows).to_string(index=False))
