"""Survival analysis for RUL: learning from engines that have NOT failed yet.

THE GAP. DEPLOYMENT_REALITY.md says the normal fleet has a handful of failures
and many units still running, and that the right tool is survival analysis,
because "this engine has run 180 cycles and not failed" is evidence. The
regression models in this repo throw that evidence away: a censored engine has
no RUL label, so it is dropped. complete.py stage 8 measured the cost of that
(+9 RMSE at 8 failures). Nothing here used the censored engines. This module does.

LANDMARKING. Every observed cycle of every engine becomes one sample: the
covariates are the features at that cycle, and the outcome is the time from
that cycle to the end of observation. If the engine failed, that time is its
true remaining life (event = 1). If it is still running, the time is a lower
bound on remaining life (event = 0), which is exactly what a survival likelihood
knows how to use.

TWO MODELS, and the difference between them is the experiment:

  weibull     Weibull accelerated-failure-time model (lifelines). Log-life is
              linear in the features. Weibull is the standard distribution in
              reliability engineering, and it predicts a full survival curve, so
              the median remaining life comes out directly.

  imputed     Keep the GBM, but give it labels for the censored engines. For a
              censored row the capped RUL is unknown but the survival model gives
              its conditional expectation: E[min(T, 125) | T > d, x]. The GBM
              trains on real labels for failed engines and these expectations for
              censored ones. The survival model supplies the censoring logic; the
              GBM keeps its nonlinearity.

A CONTROL THAT MATTERS. Fitting the Weibull on failed engines only separates
"survival models are better" from "using censored engines is better". Only the
second is the claim worth making, and without the control it cannot be told
apart from the first.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

CAP = 125


def landmark_targets(df: pd.DataFrame, failed_units: set[int]) -> tuple[np.ndarray, np.ndarray]:
    """Duration (cycles to end of observation, +1 so it is strictly positive)
    and event flag for every row of a possibly-censored fleet."""
    last = df.groupby("unit")["cycle"].transform("max").to_numpy()
    duration = last - df["cycle"].to_numpy() + 1.0
    event = df["unit"].isin(failed_units).to_numpy().astype(int)
    return duration.astype(float), event


def survival_columns(cols: list[str]) -> list[str]:
    """A compact covariate set for a linear-in-log-life model.

    The full 85-column feature table is built for trees. A linear model given all
    of it gets collinear copies of each sensor (raw, 5-mean, 20-mean) and spends
    its penalty sorting them out. One smoothed level, one change-since-new and one
    trend per sensor, plus age, carries the same information with less of that.
    """
    keep = ["cycle"] + [c for c in cols if c.endswith(("_rm20", "_delta", "_slope20"))]
    return [c for c in keep if c in cols]


class WeibullRUL:
    def __init__(self, penalizer: float = 0.05):
        self.penalizer = penalizer

    def fit(self, x: pd.DataFrame, duration: np.ndarray, event: np.ndarray,
            max_rows: int = 6000, seed: int = 0) -> "WeibullRUL":
        from lifelines import WeibullAFTFitter

        self.cols_ = list(x.columns)
        self.mu_ = x.mean()
        self.sd_ = x.std().replace(0, 1.0)
        z = (x - self.mu_) / self.sd_
        # Rows of one engine are strongly correlated, so thinning costs little
        # information and keeps the fit to seconds.
        if len(z) > max_rows:
            idx = np.random.default_rng(seed).choice(len(z), max_rows, replace=False)
            z, duration, event = z.iloc[idx], duration[idx], event[idx]
        d = z.reset_index(drop=True).copy()
        d["_T"] = duration
        d["_E"] = event
        self.model_ = WeibullAFTFitter(penalizer=self.penalizer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(d, "_T", "_E")
        return self

    def _z(self, x: pd.DataFrame) -> pd.DataFrame:
        return ((x[self.cols_] - self.mu_) / self.sd_).reset_index(drop=True)

    def _params(self, x: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Per-row Weibull scale lambda and the shared shape rho."""
        p = self.model_.params_
        lam = p.loc["lambda_"]
        z = self._z(x)
        lin = np.full(len(z), float(lam.get("Intercept", 0.0)))
        for c in self.cols_:
            if c in lam.index:
                lin += float(lam[c]) * z[c].to_numpy()
        rho = float(np.exp(p.loc["rho_"]["Intercept"]))
        return np.exp(lin), rho

    def predict_rul(self, x: pd.DataFrame, cap: float | None = CAP) -> np.ndarray:
        """Median remaining life, in the same units as the regression target."""
        lam, rho = self._params(x)
        med = lam * np.log(2.0) ** (1.0 / rho) - 1.0
        med = np.maximum(med, 0.0)
        return np.minimum(med, cap) if cap is not None else med

    def imputed_capped_rul(self, x: pd.DataFrame, duration: np.ndarray,
                           cap: float = CAP) -> np.ndarray:
        """E[min(RUL, cap) | engine outlived its observation window, x].

        `duration` is the landmark duration (cycles seen from this row to the end
        of observation, +1), so the engine is known to have T > duration on the
        model's time scale, and RUL = T - 1. Uses
            E[min(T, c) | T > d] = d + integral_d^c S(t) dt / S(d)
        on an integer grid, which is exact at integer d. A row that already
        survived past the cap gets the cap: its capped label is known exactly.
        """
        lam, rho = self._params(x)
        c = cap + 1.0
        t = np.arange(0.0, c + 1.0)                       # 0 .. cap+1
        s = np.exp(-(t[None, :] / lam[:, None]) ** rho)   # rows x grid
        cum = np.concatenate([np.zeros((len(lam), 1)),
                              np.cumsum(0.5 * (s[:, 1:] + s[:, :-1]), axis=1)], axis=1)
        d = np.clip(np.asarray(duration, float), 0, c).astype(int)
        rows = np.arange(len(lam))
        s_d = s[rows, d]
        tail = cum[:, -1] - cum[rows, d]
        with np.errstate(divide="ignore", invalid="ignore"):
            et = np.where(s_d > 1e-12, d + tail / s_d, d)
        rul = np.minimum(et, c) - 1.0
        return np.where(d >= c, cap, np.clip(rul, 0.0, cap))


def concordance(true_rul: np.ndarray, pred_rul: np.ndarray) -> float:
    """Harrell's C on uncensored pairs: how often the engine predicted to fail
    first actually fails first. 0.5 is chance, 1.0 is a perfect ordering."""
    t = np.asarray(true_rul, float)
    p = np.asarray(pred_rul, float)
    num = den = 0.0
    for i in range(len(t)):
        m = t[i] < t
        den += m.sum()
        num += (p[i] < p[m]).sum() + 0.5 * (p[i] == p[m]).sum()
    return float(num / den) if den else float("nan")
