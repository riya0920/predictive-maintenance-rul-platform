"""C-MAPSS loading, piecewise-linear RUL targets, and operating-condition normalisation.

The dataset ships four sub-datasets that differ in exactly two ways:

    FD001   1 operating condition   1 fault mode   (HPC degradation)
    FD002   6 operating conditions  1 fault mode
    FD003   1 operating condition   2 fault modes  (HPC + fan)
    FD004   6 operating conditions  2 fault modes

That table is the whole difficulty gradient, and it is why any result reported on
FD001 alone is uninformative about the method.

Censoring: TRAIN units run to failure (last cycle == failure). TEST units are
truncated at a random point and the true remaining life at that point is given in
RUL_FDxxx.txt. So the test set is right-censored -- there is no failure event in
it, and no trajectory-level metric (lead time, alpha-lambda) can be computed on
it. Those live in alarm.py and run on held-out TRAIN units instead.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "CMAPSS"

OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit", "cycle"] + OP_COLS + SENSOR_COLS

# The standard early-life plateau. Justified in README; sensitivity swept in train.py.
DEFAULT_RUL_CAP = 125


def _read(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, : len(COLS)]
    df.columns = COLS
    return df


def load(fd: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return (train, test, rul_at_last_test_cycle) for e.g. fd='FD001'."""
    train = _read(DATA / f"train_{fd}.txt")
    test = _read(DATA / f"test_{fd}.txt")
    rul = _read_rul(DATA / f"RUL_{fd}.txt")
    return train, test, rul


def _read_rul(path: pathlib.Path) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].to_numpy(dtype=float)


def piecewise_rul(df: pd.DataFrame, cap: int | None = DEFAULT_RUL_CAP) -> pd.Series:
    """Piecewise-linear RUL: min(cycles_to_failure, cap).

    Why cap at all: for the first several hundred cycles the engine is healthy and
    the sensors say so -- the signal is flat. An uncapped linear target asks the
    model to distinguish RUL=300 from RUL=250 from evidence that does not exist,
    and the only way to reduce that loss is to regress on cycle count, i.e. to
    learn the *censoring pattern of the training set* rather than degradation.
    The cap makes early life a single class ("healthy, far from failure") and
    spends the model's capacity on the region where the sensors actually move.
    """
    life = df.groupby("unit")["cycle"].transform("max")
    rul = life - df["cycle"]
    return rul.clip(upper=cap) if cap is not None else rul


def operating_regimes(df: pd.DataFrame, n_regimes: int = 6) -> np.ndarray:
    """Discretise the 3 operating settings into regimes.

    FD002/FD004 fly six conditions. Raw sensor values are dominated by *which
    condition the engine is in*, not by how degraded it is -- the between-regime
    variance swamps the degradation signal. Rounding the settings recovers the six
    clusters exactly (they are discrete by construction in C-MAPSS), which is
    cheaper and more faithful than k-means.
    """
    key = np.round(df[OP_COLS].to_numpy(dtype=float), decimals=0)
    _, labels = np.unique(key, axis=0, return_inverse=True)
    return labels


class ConditionNormaliser:
    """Per-regime z-scoring of sensors, fit on train and applied to test.

    This is the single highest-leverage step for FD002/FD004. Without it a model
    spends its capacity learning the six condition offsets.
    """

    def __init__(self) -> None:
        self.stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.cols: list[str] = []
        self.global_: tuple[np.ndarray, np.ndarray] | None = None

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "ConditionNormaliser":
        self.cols = cols
        reg = operating_regimes(df)
        x = df[cols].to_numpy(dtype=float)
        self.global_ = (x.mean(0), np.where(x.std(0) < 1e-9, 1.0, x.std(0)))
        for r in np.unique(reg):
            xi = x[reg == r]
            sd = xi.std(0)
            self.stats[int(r)] = (xi.mean(0), np.where(sd < 1e-9, 1.0, sd))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        reg = operating_regimes(df)
        x = df[self.cols].to_numpy(dtype=float).copy()
        for r in np.unique(reg):
            mu, sd = self.stats.get(int(r), self.global_)
            m = reg == r
            x[m] = (x[m] - mu) / sd
        out = df.copy()
        out[self.cols] = x
        return out


def informative_sensors(train: pd.DataFrame, normaliser: ConditionNormaliser | None) -> list[str]:
    """Drop sensors that carry no information after condition normalisation.

    Several C-MAPSS sensors are constant (s1, s5, s10, s16, s18, s19 on FD001) and
    a few are integer-valued counters. Selecting by post-normalisation variance
    rather than by a hard-coded list means FD002/FD004 keep sensors that only
    become informative once the regime offset is removed.
    """
    df = normaliser.transform(train) if normaliser is not None else train
    keep = []
    for c in SENSOR_COLS:
        v = df[c].to_numpy(dtype=float)
        if np.nanstd(v) > 1e-6 and len(np.unique(np.round(v, 6))) > 8:
            keep.append(c)
    return keep
