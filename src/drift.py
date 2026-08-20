"""Drift monitoring and the retrain trigger.

The question this answers is not "has the data changed" -- data always changes --
but "has it changed in a way that should cost somebody a weekend". That means a
trigger with a threshold, a persistence rule, and an owner, not a dashboard with a
line going up.

FOUR SIGNALS, RANKED BY HOW FAST THEY CAN PAGE SOMEBODY
-------------------------------------------------------
    1. input PSI          label-free, available immediately
    2. prediction PSI     label-free, available immediately
    3. alarm-rate drift   label-free, and it is the one that maps to a business
                          consequence: the maintenance team's workload
    4. error / PHM score  needs failures, so it arrives WEEKS late and cannot
                          page anyone in time

The ranking is the point. A monitoring plan built on (4) is a post-mortem
generator. In this domain a run-to-failure event is the label, and it arrives
after the thing you were trying to prevent -- so the only signals that can act as
an alarm are 1-3, and the only one of those tied to a cost is (3).

PSI (Population Stability Index), the standard credit-risk statistic, reused here
because sensor distributions are exactly the thing it was built for:

    PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)

Conventional bands, stated because a PSI without bands is a number without a
decision:
    < 0.10   no significant shift
    0.10-0.25  moderate shift -- monitor, investigate if it persists
    > 0.25   significant shift -- investigate now

Those bands are conventions from credit scoring, not physics. They are used here
because they are what a reviewer will expect, and `psi_bands()` names their origin
so nobody mistakes them for something derived.
"""
from __future__ import annotations

import numpy as np

PSI_BANDS = ((0.10, "stable"), (0.25, "moderate"), (float("inf"), "significant"))


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10,
        eps: float = 1e-6) -> float:
    """PSI with quantile bins taken from the EXPECTED (reference) distribution.

    Quantile bins from the reference, not from the combined data: the reference is
    the thing being compared against, so its bin edges have to be frozen with it.
    Re-deriving edges from the pooled data lets a shift hide inside the binning,
    which is the most common way a PSI implementation quietly under-reports.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < n_bins or len(actual) == 0:
        return float("nan")

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e_frac = np.histogram(expected, bins=edges)[0] / len(expected)
    a_frac = np.histogram(actual, bins=edges)[0] / len(actual)
    e_frac = np.clip(e_frac, eps, None)
    a_frac = np.clip(a_frac, eps, None)
    return float(np.sum((a_frac - e_frac) * np.log(a_frac / e_frac)))


def band(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    for threshold, name in PSI_BANDS:
        if value < threshold:
            return name
    return "significant"


def feature_drift(reference: np.ndarray, current: np.ndarray,
                  names: list[str], n_bins: int = 10) -> list[dict]:
    """Per-feature PSI between a reference window and a current window."""
    rows = []
    for j, nm in enumerate(names):
        v = psi(reference[:, j], current[:, j], n_bins)
        rows.append({"feature": nm, "psi": v, "band": band(v)})
    return sorted(rows, key=lambda r: -(r["psi"] if np.isfinite(r["psi"]) else -1))


def alarm_rate(pred: np.ndarray, threshold: float) -> float:
    """Fraction of predictions below the alarm threshold.

    The label-free signal with a business consequence attached. If this doubles,
    the maintenance team's workload doubles, and that is true whether the cause is
    a genuine fleet-wide degradation, a sensor recalibration, or a model bug --
    all three of which are worth a phone call.
    """
    p = np.asarray(pred, dtype=float)
    return float(np.mean(p < threshold)) if len(p) else float("nan")


class RetrainTrigger:
    """A documented, testable retrain condition.

    Three parts, because any one alone is a false-alarm generator:

      MAGNITUDE    the drift statistic must exceed a band threshold
      PERSISTENCE  it must do so for `k` consecutive evaluation windows -- one bad
                   window is a bad window, and a fleet-wide sensor wash on a
                   Tuesday should not trigger a retrain
      SCOPE        it must affect at least `min_features` features, OR the
                   prediction distribution itself. A single drifting sensor is a
                   SENSOR ticket, not a model ticket, and conflating the two sends
                   the wrong team.

    The scope rule is the one most implementations miss, and it is the difference
    between "the model is stale" and "the thermocouple on unit 12 is dying".
    """

    def __init__(self, psi_threshold: float = 0.25, k_windows: int = 2,
                 min_features: int = 3, alarm_rate_ratio: float = 1.5):
        self.psi_threshold = psi_threshold
        self.k_windows = k_windows
        self.min_features = min_features
        self.alarm_rate_ratio = alarm_rate_ratio
        self.history: list[dict] = []

    def evaluate(self, window_name: str, feature_psi: list[dict],
                 score_psi: float, alarm_rate_now: float,
                 alarm_rate_ref: float) -> dict:
        n_drifted = sum(1 for r in feature_psi
                        if np.isfinite(r["psi"]) and r["psi"] >= self.psi_threshold)
        ratio = (alarm_rate_now / alarm_rate_ref) if alarm_rate_ref > 0 else float("nan")

        reasons = []
        if n_drifted >= self.min_features:
            reasons.append(f"{n_drifted} features at PSI >= {self.psi_threshold}")
        if np.isfinite(score_psi) and score_psi >= self.psi_threshold:
            reasons.append(f"prediction PSI {score_psi:.3f}")
        if np.isfinite(ratio) and (ratio >= self.alarm_rate_ratio
                                   or ratio <= 1 / self.alarm_rate_ratio):
            reasons.append(f"alarm rate x{ratio:.2f}")

        rec = {
            "window": window_name,
            "n_features_drifted": n_drifted,
            "max_feature_psi": max((r["psi"] for r in feature_psi
                                    if np.isfinite(r["psi"])), default=float("nan")),
            "score_psi": score_psi,
            "alarm_rate_now": alarm_rate_now,
            "alarm_rate_ref": alarm_rate_ref,
            "alarm_rate_ratio": ratio,
            "breached": bool(reasons),
            "reasons": reasons,
        }
        self.history.append(rec)

        recent = self.history[-self.k_windows:]
        rec["consecutive_breaches"] = sum(1 for h in recent if h["breached"])
        rec["RETRAIN"] = bool(len(recent) >= self.k_windows
                              and all(h["breached"] for h in recent))
        rec["action"] = (
            "RETRAIN — persistent, multi-feature drift" if rec["RETRAIN"]
            else "investigate — single window breached" if rec["breached"]
            else "no action"
        )
        return rec
