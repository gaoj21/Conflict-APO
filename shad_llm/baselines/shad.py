"""ShAD inside the optimization loop: a decoder D0 fitted once on the initial prompt's fit scores, used to turn the
judge's raw scores into decoded ratings on the human scale for the optimizer's metric, feedback and selection.

The decoder is the one of evaluate.py (anchored_fit). After optimization the final prompt is scored and evaluated
exactly like every other method (evaluate.py refits the decoder on the final prompt's fit + sel scores).
"""
import numpy as np, pandas as pd

from evaluate import anchored_fit


class Decoder:
    def __init__(self, ds, df):
        """df: id + one column per attribute, the initial prompt's judge scores on labelled items."""
        self.ds, A = ds, ds.A
        X = df[A].to_numpy(float)
        self.fill = np.nanmean(X, 0)
        X = np.where(np.isnan(X), self.fill, X)
        self.m, self.s = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1)
        z = ds.gold.loc[df.id, "z"].to_numpy(float)
        self.zm, self.zs = z.mean(), z.std()
        Y = ds.human(df.id)
        self.ym, self.ys = Y.mean(0), Y.std(0)
        self.f = anchored_fit(self._feat(X, z), (Y - self.ym) / self.ys)

    def _feat(self, X, z):
        return np.c_[(X - self.m) / self.s, (np.asarray(z, float) - self.zm) / self.zs]

    def __call__(self, X, ids):
        """X: (n, K) raw scores (NaN allowed); ids: item ids. Returns decoded ratings on the human scale, clipped."""
        X = np.asarray(X, float).reshape(len(ids), -1)
        X = np.where(np.isnan(X), self.fill, X)
        z = self.ds.gold.loc[list(ids), "z"].to_numpy(float)
        P = self.f(self._feat(X, z)) * self.ys + self.ym
        return np.clip(P, self.ds.LO, self.ds.HI)

    def df(self, df):
        out = df.copy()
        out[self.ds.A] = self(df[self.ds.A].to_numpy(float), df.id.tolist())
        return out
