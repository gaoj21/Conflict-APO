"""Content labels for APO: the human scores with their per-attribute length component removed,
y~_k = y_k - b_k (z - mean z), b_k the least-squares slope of y_k on the standardized log length z over fit + sel.
APO optimizes the prompt toward y~ (via GEM_TEACHER), so moves along the length axis earn nothing; ShAD, which sees
z, puts each attribute's length dependence back. Test keeps the human scores. Rounded to 0.1 for the optimizers.
  .venv/bin/python scripts/make_content_labels.py <ds> <out.csv>"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../baselines")
from gem import DS  # noqa: E402

ds, out = DS(sys.argv[1]), sys.argv[2]
ids = [r["id"] for s in ("fit", "sel") for r in ds.rows[s]]
z = ds.gold.loc[ids, "z"].to_numpy(float); zc = z - z.mean()
Y = ds.human(ids)
b = (zc @ (Y - Y.mean(0))) / (zc @ zc)
C = np.round(Y - np.outer(zc, b), 1)
pd.DataFrame(C, index=pd.Index(ids, name="id"), columns=ds.A).reset_index().to_csv(out, index=False)
r = [np.corrcoef(z, Y[:, k])[0, 1] for k in range(len(ds.A))]
print(ds.name, "length corr of human labels:", dict(zip(ds.A, np.round(r, 2))))
print("  after removal:", dict(zip(ds.A, np.round([np.corrcoef(z, C[:, k])[0, 1] for k in range(len(ds.A))], 2))))
