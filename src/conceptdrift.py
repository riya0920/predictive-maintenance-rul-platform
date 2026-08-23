"""Concept drift: when the inputs look identical and the model is wrong anyway.

THE GAP. EXTENSIONS.md §2 built a PSI-based monitor over the model's INPUTS, and
closed by naming what it cannot do:

    "The drift monitor is input-side only. No labelled backtest against realised
     failures, so the class of drift where only the sensor-to-life relationship
     changes remains uncovered."

That sentence describes the drift that actually ends monitoring programmes.
Input drift is the easy case -- it is visible without labels, it is usually caused
by something a human already knows about (a new supplier, a recalibration, a new
site), and PSI finds it. Concept drift is the hard case:

    P(x) unchanged.  P(y | x) changed.

The sensors look exactly as they always did. Every input-side monitor in the
world is silent, by construction, because nothing about the inputs is different.
What changed is the *meaning* of those inputs -- the same vibration signature now
precedes failure in 40 cycles instead of 60. A fleet gets this from a lubricant
change, a duty-cycle change, a repair-standard change, or a fuel change: things
that alter degradation physics without altering what the sensors read.

WHY THIS IS A STRUCTURAL LIMIT AND NOT A TUNING PROBLEM. No amount of
sensitivity on an input monitor detects a change in a conditional distribution
when the marginal is fixed. Detecting it requires an outcome, and outcomes arrive
one failure at a time, months late. That is the real cost: the detection latency
of a concept-drift monitor is bounded below by the failure rate of the fleet.

WHAT THIS MODULE DOES:
  1. injects concept drift with the input distribution held fixed
  2. confirms the PSI monitor is silent on it (the negative control)
  3. builds the label-aware monitors that do work, and measures how many
     realised failures each needs before it fires
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# injection
# ---------------------------------------------------------------------------

def accelerate_degradation(rul: np.ndarray, units: np.ndarray, *,
                           factor: float = 0.65, affected: set | None = None,
                           ) -> tuple[np.ndarray, dict]:
    """Compress remaining life by `factor` without touching a single sensor value.

    This is the cleanest possible concept drift: identical inputs, different
    outcome. The engine's sensors trace exactly the path they always did; it just
    fails sooner along that path.

    Implemented on the TARGET rather than the features precisely so that the
    input distribution is provably unchanged -- there is no argument to be had
    about whether the PSI monitor "should" have caught it. It cannot have, because
    the bytes it reads are bit-identical.
    """
    out = rul.astype(float).copy()
    if affected is None:
        m = np.ones(len(rul), dtype=bool)
    else:
        m = np.isin(units, list(affected))
    out[m] = out[m] * factor
    return out, {"factor": factor, "rows_affected": int(m.sum()),
                 "units_affected": int(len(np.unique(units[m])))}


# ---------------------------------------------------------------------------
# label-aware monitors
# ---------------------------------------------------------------------------

def residual_monitor(y_true: np.ndarray, y_pred: np.ndarray, *,
                     baseline_mean: float, baseline_sd: float,
                     k: float = 3.0) -> dict:
    """Is the model's error distribution where it was at qualification time?

    The statistic is the MEAN residual, not the RMSE, and the sign is the whole
    point. Concept drift of the "fails sooner" kind makes the model
    systematically OPTIMISTIC: it predicts more life than the engine has, so
    residuals go positive together. RMSE rises too, but it also rises when the
    model merely gets noisier, and those two want different responses -- a biased
    model needs retraining, a noisy one may just need more data.
    """
    resid = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    n = max(len(resid), 1)
    m = float(np.mean(resid))
    # The statistic is a MEAN, so it is compared against the standard error of a
    # mean -- baseline_sd / sqrt(n) -- not against the per-observation SD. Using
    # the raw SD here is a real and easy mistake: it makes the monitor roughly
    # sqrt(n) times too insensitive, which for n=20 is a factor of 4.5, and it
    # fails silently because a monitor that never fires looks like a calm process.
    se = max(baseline_sd, 1e-9) / np.sqrt(n)
    z = (m - baseline_mean) / se
    return {"mean_residual": m, "z": float(z), "fired": bool(abs(z) > k),
            "n": int(n), "standard_error": float(se),
            "direction": "optimistic" if m > baseline_mean else "pessimistic",
            "rmse": float(np.sqrt(np.mean(resid ** 2)))}


def page_hinkley(values: np.ndarray, *, delta: float = 0.5,
                 threshold: float = 25.0) -> dict:
    """Sequential change detector on a stream of residuals.

    Page-Hinkley rather than a fixed window because the question a maintenance
    team asks is "has it changed *yet*", answered as early as possible, not "was
    the last 90 days different from the 90 before". `delta` is the slack that
    stops ordinary noise accumulating; `threshold` trades detection delay against
    false alarms.

    Returns the index of the first alarm, which is what makes this measurable:
    with one realised failure per index, the alarm index IS the number of
    failures the fleet had to suffer before the monitor spoke.
    """
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return {"fired": False, "at": None, "n": 0}
    mt, m_min, running = 0.0, 0.0, 0.0
    for i, v in enumerate(x):
        running += v
        mean_so_far = running / (i + 1)
        mt += v - mean_so_far - delta
        m_min = min(m_min, mt)
        if mt - m_min > threshold:
            return {"fired": True, "at": int(i), "n": len(x),
                    "statistic": float(mt - m_min)}
    return {"fired": False, "at": None, "n": len(x),
            "statistic": float(mt - m_min)}


def alarm_rate_monitor(preds_by_unit: list[np.ndarray], threshold: float, k: int,
                       *, baseline_rate: float, n_sigma: float = 3.0) -> dict:
    """The label-free fallback: how often the alarm policy fires.

    Included because it is the monitor a plant can actually run daily. It needs no
    labels, so it has no failure-rate floor -- but it is a much blunter instrument
    than the residual monitor, and the experiment reports both so the trade is
    visible rather than asserted.
    """
    fired = []
    for p in preds_by_unit:
        below = p < threshold
        run = 0
        hit = False
        for b in below:
            run = run + 1 if b else 0
            if run >= k:
                hit = True
                break
        fired.append(hit)
    rate = float(np.mean(fired)) if fired else 0.0
    n = max(len(fired), 1)
    se = np.sqrt(max(baseline_rate * (1 - baseline_rate), 1e-9) / n)
    return {"rate": rate, "baseline": baseline_rate,
            "z": float((rate - baseline_rate) / max(se, 1e-9)),
            "fired": bool(abs(rate - baseline_rate) > n_sigma * se),
            "n_units": len(fired)}


# ---------------------------------------------------------------------------
# detection latency
# ---------------------------------------------------------------------------

def detection_latency(resid_stream: np.ndarray, *, baseline_mean: float,
                      baseline_sd: float, delta: float = 0.5,
                      threshold: float = 25.0) -> dict:
    """How many realised failures before each monitor notices.

    This is the number that decides whether a concept-drift monitor is worth
    building for a given fleet. A monitor needing 30 failures is useless to an
    operator with four failures a year and essential to one with four a week --
    and that is a property of the FLEET, not of the monitor.
    """
    z = (resid_stream - baseline_mean) / max(baseline_sd, 1e-9)
    ph = page_hinkley(z, delta=delta, threshold=threshold)

    # A simple 3-sigma rule on a growing mean, for comparison.
    naive = None
    for i in range(1, len(z) + 1):
        se = 1.0 / np.sqrt(i)
        if abs(np.mean(z[:i])) > 3 * se:
            naive = i
            break
    return {"page_hinkley_failures": ph["at"] + 1 if ph["fired"] else None,
            "page_hinkley_fired": ph["fired"],
            "cumulative_3sigma_failures": naive,
            "n_failures_available": len(z)}


# ---------------------------------------------------------------------------
# bootstrap intervals -- the "20 units is thin" gap
# ---------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, stat=np.median, n_boot: int = 4000,
                 alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap for a statistic over units.

    Closes the README's item 7: "the P05 lead time is the 5th percentile of at
    most 20 numbers". A percentile of 20 numbers has an interval wide enough to
    change decisions, and quoting it without one invites a reader to treat it as
    a specification.

    Percentile bootstrap rather than BCa: with n=20 the bias-correction term is
    itself estimated from 20 points, and the extra machinery implies a precision
    the sample does not have.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"point": float(stat(x)) if len(x) else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "width": float("nan"),
                "n": len(x), "n_boot": 0}
    rng = np.random.default_rng(seed)
    draws = np.array([stat(rng.choice(x, size=len(x), replace=True))
                      for _ in range(n_boot)])
    return {"point": float(stat(x)),
            "lo": float(np.quantile(draws, alpha / 2)),
            "hi": float(np.quantile(draws, 1 - alpha / 2)),
            "n": int(len(x)), "n_boot": n_boot,
            "width": float(np.quantile(draws, 1 - alpha / 2)
                           - np.quantile(draws, alpha / 2))}
