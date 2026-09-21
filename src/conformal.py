"""Calibrated uncertainty on RUL, and an alarm that fires on the safe end of it.

THE QUESTION. The point model says "23 cycles left". A planner acting on it needs
to know how wrong that can be, and a safety case needs a number with a stated
guarantee attached: "at least 18 cycles left, and that statement is true 90% of
the time". Conformal prediction produces exactly that, with no assumption about
the model or the error distribution. The only assumption is exchangeability
between the calibration engines and the engines being scored.

ONE-SIDED, on purpose. The dangerous error in maintenance is over-predicting
remaining life, so only a LOWER bound is built. A two-sided interval would spend
half its miscoverage budget on the harmless side.

THREE VARIANTS, because they answer different questions:

  split       one correction q for every prediction: L = pred - q. Guarantees
              90% coverage on average, but the average is dominated by the long
              flat healthy region where the model is easy, so coverage near end
              of life (the only place anyone acts) can be worse.

  mondrian    a separate q per band of predicted RUL. Coverage then holds inside
              each band, including the one below 50 cycles.

  cqr         conformalised quantile regression (Romano et al., 2019). A GBM is
              trained to predict the 10th percentile directly, then corrected on
              the calibration engines, per band. This is the only variant whose
              width can differ between two engines with the SAME point prediction,
              which is the whole reason it can change an alarm decision.

WHY ONLY CQR CAN BEAT A THRESHOLD SHIFT. `split` and `mondrian` compute the lower
bound as a function of the point prediction alone. Alarming on "lower bound < T"
is then the same as alarming on "prediction < T'" for some other T'. It changes
where the threshold sits, nothing else. CQR's bound depends on the features, so
an engine whose sensors are unusual gets a wider band and alarms earlier than a
typical engine with the same point prediction. Whether that buys anything is an
empirical question, and uq_survival.py answers it at matched average warning.

EXCHANGEABILITY is at the level of ENGINES, not rows. Calibration engines are
whole units the models never trained on, split exactly like the holdout.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ALPHA = 0.10
# Bands of predicted RUL. The low band is where alarm decisions are made.
BANDS = (0.0, 25.0, 50.0, 100.0, np.inf)


def conformal_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    """Finite-sample-corrected (1 - alpha) quantile of nonconformity scores.

    The ceil((n + 1)(1 - alpha)) / n level is what turns "about 90%" into a
    guarantee of at least 90%. With too few scores to reach that level the only
    honest bound is infinitely wide.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(s[k - 1])


def band_of(pred: np.ndarray, bands=BANDS) -> np.ndarray:
    return np.clip(np.searchsorted(bands, pred, side="right") - 1, 0, len(bands) - 2)


class LowerBound:
    """Fit on calibration rows, then produce a one-sided lower bound on RUL."""

    def __init__(self, kind: str = "split", alpha: float = ALPHA, bands=BANDS):
        if kind not in ("split", "mondrian", "cqr"):
            raise ValueError(kind)
        self.kind, self.alpha, self.bands = kind, alpha, bands
        self.q_: dict[int, float] = {}

    def _base(self, point: np.ndarray, q_lo: np.ndarray | None) -> np.ndarray:
        return q_lo if self.kind == "cqr" else point

    def calibrate(self, y: np.ndarray, point: np.ndarray,
                  q_lo: np.ndarray | None = None) -> "LowerBound":
        base = self._base(point, q_lo)
        # Nonconformity = how far the base OVER-states remaining life.
        scores = base - y
        if self.kind == "split":
            self.q_ = {-1: conformal_quantile(scores, self.alpha)}
        else:
            b = band_of(point, self.bands)
            self.q_ = {i: conformal_quantile(scores[b == i], self.alpha)
                       for i in range(len(self.bands) - 1)}
        return self

    def predict(self, point: np.ndarray, q_lo: np.ndarray | None = None) -> np.ndarray:
        base = self._base(point, q_lo)
        if self.kind == "split":
            q = np.full(len(point), self.q_[-1])
        else:
            q = np.array([self.q_[i] for i in band_of(point, self.bands)])
        return np.maximum(base - q, 0.0)


def fit_quantile_gbm(x: np.ndarray, y: np.ndarray, quantile: float = ALPHA,
                     seed: int = 20260818) -> HistGradientBoostingRegressor:
    """Same capacity as models.fit_gbm, pinball loss at `quantile`."""
    m = HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile, max_iter=400, learning_rate=0.06,
        max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=seed)
    return m.fit(x, y)


def coverage_report(y: np.ndarray, lower: np.ndarray, point: np.ndarray) -> dict:
    """Coverage overall and by TRUE remaining life, plus how much margin it costs."""
    ok = y >= lower
    out = {"coverage": float(ok.mean()), "mean_margin": float(np.mean(point - lower)),
           "n": int(len(y))}
    for lo, hi, name in ((0, 25, "true_rul_0_25"), (25, 50, "true_rul_25_50"),
                         (50, 125, "true_rul_50_125"), (125, np.inf, "healthy_cap")):
        m = (y >= lo) & (y < hi) if np.isfinite(hi) else y >= lo
        out[name] = float(ok[m].mean()) if m.any() else None
        out[name + "_margin"] = float(np.mean((point - lower)[m])) if m.any() else None
    return out
