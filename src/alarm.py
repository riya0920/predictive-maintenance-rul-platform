"""The maintenance-decision layer: the part that turns a regression into a policy.

Nothing in here is about accuracy. It is about WHEN THE WORK ORDER GETS CUT, which
is the only thing a maintenance planner is buying.

Policy: raise an alarm when predicted RUL < T for k CONSECUTIVE cycles.
The persistence rule k exists because a single-cycle dip below T is noise, and an
alarm a planner has learned to ignore is worse than no alarm at all.

Definitions used throughout (stated because everyone means something different):
  * sustained alarm  -- first cycle of the first run of >= k consecutive cycles
                        with pred < T that is never de-asserted before failure.
  * lead time        -- failure_cycle - sustained_alarm_cycle. The number the
                        planner cares about: how much warning do I get.
  * nuisance alarm   -- a run of >= k cycles with pred < T that DOES de-assert
                        (prediction climbs back over T). Counted per unit lifetime.
                        This is the alarm-fatigue number.
  * missed failure   -- lead time < L_MIN, i.e. the alarm came too late to act on,
                        or never came. Warning you cannot act on is not warning.
"""
from __future__ import annotations

import numpy as np

L_MIN = 10  # cycles of warning below which the alarm is unactionable


def _runs_below(mask: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Return [start, end) index pairs of runs of True with length >= k."""
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= k:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def evaluate_unit(pred: np.ndarray, threshold: float, k: int) -> dict:
    """pred is the predicted RUL at every cycle of one run-to-failure trajectory."""
    n = len(pred)
    mask = pred < threshold
    runs = _runs_below(mask, k)
    sustained = None
    nuisance = 0
    for start, end in runs:
        if end >= n:  # runs to the end of life -> this is the real alarm
            sustained = start + k - 1  # alarm asserts once persistence is satisfied
        else:
            nuisance += 1
    if sustained is None:
        lead = 0.0
        missed = True
    else:
        lead = float(n - 1 - sustained)
        missed = lead < L_MIN
    return {
        "lead_time": lead,
        "missed": missed,
        "nuisance_alarms": nuisance,
        "life": n,
    }


def cost_of_policy(
    units: list[np.ndarray],
    threshold: float,
    k: int,
    c_fail: float = 20.0,
    c_pm: float = 1.0,
    c_life: float = 0.02,
    c_nuisance: float = 0.5,
) -> dict:
    """Expected cost per unit under a cost model whose numbers are ASSUMED.

    c_fail       cost of an in-service failure (unplanned removal, secondary
                 damage, AOG) in units of one planned maintenance event
    c_pm         cost of the planned maintenance event itself
    c_life       cost per cycle of remaining useful life thrown away by pulling the
                 asset early -- the term that stops the policy from just alarming
                 on cycle 1 to guarantee zero failures
    c_nuisance   cost of an inspection triggered by an alarm that then cleared

    These are assumptions, not measurements. The sensitivity sweep in train.py is
    what makes them honest: the chosen operating point has to survive a range of
    c_fail/c_pm, or the policy is a fit to my guess about the economics.
    """
    per = [evaluate_unit(p, threshold, k) for p in units]
    total = 0.0
    for r in per:
        if r["missed"]:
            total += c_fail
        else:
            total += c_pm + c_life * max(0.0, r["lead_time"] - L_MIN)
        total += c_nuisance * r["nuisance_alarms"]
    leads = np.array([r["lead_time"] for r in per if not r["missed"]], dtype=float)
    lifetimes = np.array([r["life"] for r in per], dtype=float)
    nuis = np.array([r["nuisance_alarms"] for r in per], dtype=float)
    return {
        "threshold": float(threshold),
        "k": int(k),
        "cost_per_unit": total / len(per),
        "missed_rate": float(np.mean([r["missed"] for r in per])),
        "lead_mean": float(leads.mean()) if len(leads) else 0.0,
        "lead_p05": float(np.percentile(leads, 5)) if len(leads) else 0.0,
        "lead_p50": float(np.percentile(leads, 50)) if len(leads) else 0.0,
        "lead_p95": float(np.percentile(leads, 95)) if len(leads) else 0.0,
        "frac_ge_10": float(np.mean(leads >= 10)) if len(leads) else 0.0,
        "nuisance_per_unit": float(nuis.mean()),
        "nuisance_per_1000_cycles": float(1000.0 * nuis.sum() / lifetimes.sum()),
    }


def tune(units: list[np.ndarray], thresholds, ks, **cost_kwargs) -> tuple[dict, list[dict]]:
    grid = [cost_of_policy(units, t, k, **cost_kwargs) for t in thresholds for k in ks]
    best = min(grid, key=lambda r: r["cost_per_unit"])
    return best, grid
