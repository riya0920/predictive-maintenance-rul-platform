"""RUL scoring: RMSE, the PHM08 asymmetric score, horizon-stratified error,
and a simplified alpha-lambda prognostic-horizon analysis.

PHM08 score (Saxena, Goebel, Simon & Eklund, "Damage propagation modeling for
aircraft engine run-to-failure simulation", PHM 2008 -- the C-MAPSS paper, and the
scoring function used for the PHM08 data challenge):

    d = RUL_pred - RUL_true
    s = exp(-d/13) - 1   for d <  0   (EARLY: predicted failure sooner than truth)
    s = exp( d/10) - 1   for d >= 0   (LATE:  predicted failure later than truth)

The asymmetry is the point. 10 < 13 means a late prediction of a given magnitude
costs more than an early one of the same magnitude, because an early prediction
buys an unnecessary maintenance event and a late one buys an in-service failure.
It is exponential rather than quadratic because maintenance cost is not
proportional to error -- being 60 cycles late is not twice as bad as 30, it is
categorically worse (you did not get the part on site in time either way, but now
you also lost the asset).

Where the asymmetry FLIPS: when the consequence of failure is cheap relative to
the maintenance action. A $30 bearing on a redundant pump with condition-based
spares in the crib -- running it to failure is the *correct* policy, and early
prediction is the expensive error. The asymmetry encodes economics, not physics,
so it is a per-asset parameter, not a constant.
"""
from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))


def horizon_strata(
    y_true: np.ndarray, y_pred: np.ndarray, edges=(0, 20, 50, 90, 126)
) -> list[dict]:
    """Error as a function of how close to failure we are.

    A model that is wild at RUL=120 and tight at RUL=15 is a GOOD maintenance
    model: the alarm decision is taken near the end of life, so that is where
    accuracy is spent. Aggregate RMSE hides this entirely.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_true >= lo) & (y_true < hi)
        if m.sum() == 0:
            continue
        rows.append(
            {
                "band": f"[{lo},{hi})",
                "n": int(m.sum()),
                "rmse": rmse(y_true[m], y_pred[m]),
                "bias": float(np.mean(y_pred[m] - y_true[m])),
                "phm_per_sample": phm_score(y_true[m], y_pred[m]) / int(m.sum()),
            }
        )
    return rows


def _ph(inside: np.ndarray) -> int:
    """First index after which `inside` is True for the rest of the trajectory."""
    n = len(inside)
    for i in range(n):
        if inside[i:].all():
            return i
    return n


def alpha_lambda(
    true_rul: np.ndarray, pred_rul: np.ndarray, alpha: float = 0.2,
    min_cone_cycles: float = 5.0,
) -> dict:
    """alpha-lambda / prognostic-horizon analysis for ONE trajectory.

    Saxena, Celaya, Saha, Saha & Goebel, "Metrics for evaluating performance of
    prognostic techniques" (PHM 2008). The accuracy cone is +/- alpha * true_RUL,
    shrinking as the unit approaches failure -- 20% of 100 cycles is loose, 20% of
    10 cycles is tight.

    Prognostic horizon = the earliest index after which the prediction stays inside
    the cone for the rest of life. "Enters AND STAYS" is the whole idea: a model
    that dips into the cone, leaves, and comes back has not converged, and a
    snapshot metric would credit it anyway.

    TWO cones are computed, and the difference is the point:

      strict     +/- alpha*RUL, exactly as defined. At RUL = 3 this demands the
                 prediction be within 0.6 of a cycle, forever after. Essentially
                 nothing passes, which makes the strict PH a near-constant zero
                 and therefore useless as a discriminator -- it is measuring the
                 definition, not the model.
      floored    +/- max(alpha*RUL, min_cone_cycles). Below a few cycles of RUL the
                 distinction between "fails Tuesday" and "fails Wednesday" is not
                 actionable anyway; the maintenance window has already closed. The
                 floor encodes that, and it is what makes the metric comparative.

    Reporting only the floored version would be flattering; reporting only the
    strict version would be a statistic about arithmetic. Both are returned.
    """
    true_rul = np.asarray(true_rul, dtype=float)
    pred_rul = np.asarray(pred_rul, dtype=float)
    n = len(true_rul)
    err = np.abs(pred_rul - true_rul)

    inside_strict = err <= alpha * np.maximum(true_rul, 1.0)
    inside_floor = err <= np.maximum(alpha * true_rul, min_cone_cycles)

    ph_s, ph_f = _ph(inside_strict), _ph(inside_floor)
    out = {
        "converged_strict": bool(ph_s < n),
        "converged": bool(ph_f < n),
        "prognostic_horizon_cycles": float(true_rul[ph_f]) if ph_f < n else 0.0,
        "lambda_fraction": float(ph_f / n) if ph_f < n else 1.0,
        "frac_inside_cone": float(inside_floor.mean()),
        "frac_inside_cone_strict": float(inside_strict.mean()),
    }
    # alpha-lambda accuracy at fixed life fractions: is the prediction inside the
    # cone when the unit has consumed lambda of its life? This is the form the
    # metric usually appears in, and it does not depend on "and stays".
    for lam in (0.25, 0.5, 0.75, 0.9):
        idx = min(n - 1, int(lam * n))
        out[f"inside_at_lambda_{lam}"] = bool(inside_floor[idx])
    return out
