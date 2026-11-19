"""What does the readout use? v0, per attribute: (a) own score + length, (b) all K scores, no length, (c) all K scores +
length (the full readout). Fitted on TRAIN splits, evaluated on EVAL split.
  python scripts/readout_ablation.py results/claude_hs2 [fit,sel test]"""
import os, sys
import numpy as np, pandas as pd
sys.argv, ARGS = sys.argv[:1] + [sys.argv[1], "x"], sys.argv[1:]
exec(open(os.path.join(os.path.dirname(__file__), "claude_opt.py")).read().split("# ---------------------------------------------------------------- main loop")[0])
TRS, EV = (ARGS[1].split(","), ARGS[2]) if len(ARGS) > 2 else (["fit", "sel"], "test")
J = [x + "_j" for x in A]
ld = lambda s: frame(pd.read_csv(os.environ["V0DIR"] + f"/v0_{s}.csv") if os.environ.get("V0DIR") else (pd.read_csv(f"{R}/out/v0_{s}.csv") if os.path.exists(f"{R}/out/v0_{s}.csv") else pd.read_csv(f"{R}/opt/text/out/v0_{s}.csv")), s)
tr = pd.concat([ld(s) for s in TRS], ignore_index=True); te = ld(EV)
def Z(d): return ((d[J].to_numpy(float) - tr[J].to_numpy(float).mean(0)) / tr[J].to_numpy(float).std(0),
                  ((d.z - tr.z.mean()) / tr.z.std()).to_numpy()[:, None])
(Str, ztr), (Ste, zte) = Z(tr), Z(te)
Ytr, H = std(tr[A].to_numpy(float)), te[A].to_numpy(float)
res = {"raw": per_task(H, te[J].to_numpy(float))}
P = np.zeros((len(te), len(A)))
for k in range(len(A)):
    f = anchored_fit(np.c_[Str[:, [k]], ztr], Ytr[:, [k]]); P[:, k] = f(np.c_[Ste[:, [k]], zte])[:, 0]
res["own + length"] = per_task(H, P)
res["all scores"] = per_task(H, anchored_fit(Str, Ytr)(Ste))
res["all scores + length"] = per_task(H, anchored_fit(np.c_[Str, ztr], Ytr)(np.c_[Ste, zte]))
print(f"=== {os.path.basename(R)} v0, fit on {TRS}, eval {EV}: {' '.join(x[:6] for x in A)} | mean")
for k, v in res.items():
    print(f"  {k:20s} " + " ".join(f"{x:6.3f}" for x in v) + f" | {np.nanmean(v):.3f}")
