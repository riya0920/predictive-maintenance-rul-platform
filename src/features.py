"""Degradation features for the tree baseline, and windowing for the sequence model.

The tree model does not see time, so time has to be put into the columns: level,
short and long rolling means, dispersion, and the least-squares slope over a
window (the degradation *trend*, which is what a reliability engineer reads off a
trend plot). The sequence model gets the raw window and is expected to learn the
same thing -- the comparison is only fair if the tree is given a real chance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHORT_W = 5
LONG_W = 20


def _rolling_slope(y: "pd.Series", w: int) -> "pd.Series":
    """Least-squares slope of y over a trailing window of w points, vectorised.

    For a window of w consecutive integers the denominator of the OLS slope is a
    constant, w*(w^2-1)/12, so the whole thing reduces to two rolling sums. The
    naive rolling().apply() version of this ran ~40x slower on FD004 and produced
    the same numbers; the check is in tests/test_features.py.
    """
    idx = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    sy = y.rolling(w).sum()
    si = idx.rolling(w).sum()
    siy = (idx * y).rolling(w).sum()
    num = siy - (si / w) * sy
    denom = w * (w * w - 1) / 12.0
    return num / denom


def build_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-row features using only that row's past (causal; no leakage across units)."""
    g = df.groupby("unit", sort=False)
    out = {"cycle": df["cycle"].to_numpy(dtype=float)}
    for c in cols:
        s = df[c]
        out[f"{c}"] = s.to_numpy(dtype=float)
        rm_s = g[c].transform(lambda x: x.rolling(SHORT_W, min_periods=1).mean())
        rm_l = g[c].transform(lambda x: x.rolling(LONG_W, min_periods=1).mean())
        rs_l = g[c].transform(lambda x: x.rolling(LONG_W, min_periods=2).std())
        first = g[c].transform("first")
        out[f"{c}_rm{SHORT_W}"] = rm_s.to_numpy(dtype=float)
        out[f"{c}_rm{LONG_W}"] = rm_l.to_numpy(dtype=float)
        out[f"{c}_sd{LONG_W}"] = rs_l.fillna(0.0).to_numpy(dtype=float)
        out[f"{c}_delta"] = (s - first).to_numpy(dtype=float)
        out[f"{c}_slope{LONG_W}"] = (
            g[c].transform(lambda x: _rolling_slope(x, LONG_W)).fillna(0.0).to_numpy(dtype=float)
        )
    return pd.DataFrame(out, index=df.index)


def sequence_windows(
    df: pd.DataFrame, cols: list[str], window: int, targets: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Left-padded windows, one per row, so every row is scorable.

    Padding with the first observed cycle (edge padding) rather than zeros: a zero
    row is not a plausible engine state and teaches the model that early life
    looks like a sensor fault.
    """
    xs, ys, idx = [], [], []
    vals = df[cols].to_numpy(dtype=np.float32)
    units = df["unit"].to_numpy()
    positions = np.arange(len(df))
    for u in np.unique(units):
        m = units == u
        v = vals[m]
        p = positions[m]
        pad = np.repeat(v[:1], window - 1, axis=0)
        vv = np.concatenate([pad, v], axis=0)
        for i in range(len(v)):
            xs.append(vv[i : i + window])
            idx.append(p[i])
            if targets is not None:
                ys.append(targets[p[i]])
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32) if targets is not None else None
    return x, y, np.asarray(idx)
