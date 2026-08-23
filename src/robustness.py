"""Making C-MAPSS behave like a real fleet, and measuring what that costs.

docs/DEPLOYMENT_REALITY.md is a list of ways C-MAPSS is a luxury dataset. It was
written as prose and never measured, which made it the largest unquantified claim
in this project: a document asserting "this would be much harder in production"
with no number attached to "much".

Each function here degrades the dataset along ONE axis and re-scores. The point
is not that the model gets worse -- of course it does -- it is the ORDERING. If
label noise costs 3 RMSE and censoring costs 25, then a programme that spends its
first year cleaning maintenance records has optimised the wrong thing, and no
amount of prose establishes that ranking.

THE FOUR AXES, and why each is the version of the problem that actually occurs:

  CENSORING       C-MAPSS runs every engine to failure. A real fleet has a handful
                  of failures and thousands of units still running. The failures
                  are also not a random sample -- an engine that failed is an
                  engine whose maintenance did not catch it, so the labelled set
                  is biased toward exactly the degradation the current programme
                  misses.

  SENSOR DRIFT    C-MAPSS sensors are perfect. Real ones drift, get recalibrated
                  in steps, and get replaced -- and a replacement is a step
                  change in an input that means nothing about the engine.

  LABEL NOISE     "Failure" in C-MAPSS is an exact cycle. In a fleet it is a
                  maintenance record, written by a person, after the fact, often
                  dated to the shift rather than the event, and sometimes
                  recording the removal rather than the fault.

  HETEROGENEITY   C-MAPSS units within a sub-dataset share a design. A real fleet
                  mixes build standards, service bulletins and operators. The
                  model is fitted on the fleet you have and run on the fleet you
                  get.

WHAT THIS IS NOT. These are injected degradations of a clean dataset, so they
measure sensitivity to a *model* of each problem, not the problems themselves.
The censoring model in particular is optimistic: it drops units, but the units it
keeps are still perfectly labelled, whereas real censoring and real label noise
arrive together and interact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. censoring
# ---------------------------------------------------------------------------

def censor_fleet(train: pd.DataFrame, rul: np.ndarray, *, n_failures: int,
                 informative: bool = True, seed: int = 0,
                 ) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Keep `n_failures` run-to-failure units; truncate the rest mid-life.

    `informative=True` implements the bias that makes censoring hard rather than
    merely small. Real failures are not a random draw: an engine reaches failure
    because nobody intervened, so the observed failures over-represent fast,
    unusual degradation and under-represent the slow drift a maintenance
    programme already catches. Here that is modelled by preferring SHORT-lived
    units as the observed failures.

    Set `informative=False` for a random draw. The difference between the two at
    the same `n_failures` is the cost of the bias alone, separated from the cost
    of having less data -- which is the comparison worth making, because only one
    of the two is fixable by waiting.
    """
    rng = np.random.default_rng(seed)
    units = train["unit"].to_numpy()
    lives = train.groupby("unit")["cycle"].max()
    uids = lives.index.to_numpy()

    if informative:
        # Sample without replacement, weighted toward short lives.
        w = 1.0 / lives.to_numpy(dtype=float)
        w = w / w.sum()
        failed = rng.choice(uids, size=min(n_failures, len(uids)), replace=False, p=w)
    else:
        failed = rng.choice(uids, size=min(n_failures, len(uids)), replace=False)
    failed = set(int(u) for u in failed)

    keep_mask = np.zeros(len(train), dtype=bool)
    censor_at = {}
    for u in uids:
        m = units == u
        if int(u) in failed:
            keep_mask |= m
        else:
            # A surviving unit is observed up to "now", uniformly through its life.
            life = int(lives.loc[u])
            cut = int(rng.integers(max(2, life // 5), max(3, life)))
            censor_at[int(u)] = cut
            idx = np.flatnonzero(m)[:cut]
            keep_mask[idx] = True

    sub = train.loc[keep_mask].copy()
    sub_rul = rul[keep_mask]
    # Censored rows carry no usable RUL target: their true remaining life is
    # unknown and strictly greater than zero. Marked, not silently kept.
    is_failed = np.array([int(u) in failed for u in sub["unit"].to_numpy()])
    return sub, sub_rul, {
        "n_units_total": len(uids), "n_failures_observed": len(failed),
        "n_censored": len(uids) - len(failed),
        "rows_kept": int(keep_mask.sum()), "rows_total": len(train),
        "labelled_rows": int(is_failed.sum()),
        "informative": informative,
        "failed_units": sorted(failed),
        "is_failed_row": is_failed,
    }


# ---------------------------------------------------------------------------
# 2. sensor drift and replacement
# ---------------------------------------------------------------------------

def inject_sensor_drift(df: pd.DataFrame, sensors: list[str], *,
                        drift_over_life: float = 1.0, n_drifting: int = 3,
                        n_replacements: int = 2, seed: int = 0,
                        ) -> tuple[pd.DataFrame, dict]:
    """Linear drift on a few sensors plus step changes from replacements.

    Drift is expressed in units of the sensor's own standard deviation per 1000
    cycles, so it is comparable across sensors of wildly different scale. A step
    change from a replacement is drawn at ~1 sd, which is the case that matters:
    large enough to move the model, small enough that nobody notices it in a trend
    plot.

    Both are applied per UNIT along its own cycle axis, because a sensor drifts
    with the hours on the sensor, not with wall-clock time across the fleet.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    chosen = list(rng.choice(sensors, size=min(n_drifting, len(sensors)),
                             replace=False))
    sd = {c: float(df[c].std()) or 1.0 for c in chosen}
    events = []

    # Drift is specified over a TYPICAL UNIT LIFE, not per 1000 cycles. The first
    # version used sd/1000 cycles, and FD001 units live ~200 cycles, so the largest
    # drift actually injected was 0.1 sd -- far below the noise floor. The
    # experiment then reported that sensor drift costs nothing, which was a
    # statement about the injection, not about the model.
    typical_life = float(df.groupby("unit")["cycle"].max().median())
    for c in chosen:
        cyc = out["cycle"].to_numpy(dtype=float)
        out[c] = out[c].to_numpy(dtype=float) + drift_over_life * sd[c] * cyc / typical_life

    units = out["unit"].unique()
    for _ in range(n_replacements):
        c = str(rng.choice(chosen))
        u = int(rng.choice(units))
        m = (out["unit"] == u).to_numpy()
        life = int(m.sum())
        at = int(rng.integers(life // 4, max(life // 4 + 1, 3 * life // 4)))
        step = float(rng.normal(0, sd[c]))
        idx = np.flatnonzero(m)[at:]
        out.iloc[idx, out.columns.get_loc(c)] = \
            out.iloc[idx, out.columns.get_loc(c)].to_numpy(dtype=float) + step
        events.append({"unit": u, "sensor": c, "at_cycle": at,
                       "step_in_sd": step / sd[c]})

    return out, {"drifting_sensors": chosen, "drift_over_life_sd": drift_over_life,
                 "typical_life_cycles": typical_life, "replacements": events}


# ---------------------------------------------------------------------------
# 3. label noise
# ---------------------------------------------------------------------------

def noisy_labels(rul: np.ndarray, units: np.ndarray, *, sd_cycles: float = 5.0,
                 late_bias_cycles: float = 3.0, seed: int = 0,
                 ) -> tuple[np.ndarray, dict]:
    """Per-unit error in the recorded failure cycle: noisy AND biased late.

    The bias is the part that matters and the part a symmetric-noise model
    misses. A maintenance record is written when the engine is removed, which is
    at best the shift the fault was found and usually later, so the recorded
    failure is systematically LATER than the true one. A model trained on those
    labels learns that engines survive longer than they do -- it is
    optimistically biased in the one direction that costs money.

    Applied per unit, not per row: a mis-dated failure shifts that unit's whole
    RUL curve, which is a very different (and more damaging) error than
    independent per-row jitter that averages out.
    """
    rng = np.random.default_rng(seed)
    out = rul.astype(float).copy()
    shifts = {}
    for u in np.unique(units):
        s = float(rng.normal(late_bias_cycles, sd_cycles))
        shifts[int(u)] = s
        out[units == u] += s
    return np.maximum(out, 0.0), {
        "sd_cycles": sd_cycles, "late_bias_cycles": late_bias_cycles,
        "mean_shift": float(np.mean(list(shifts.values()))),
        "n_units": len(shifts),
    }


# ---------------------------------------------------------------------------
# 4. fleet heterogeneity
# ---------------------------------------------------------------------------

def split_by_build(train: pd.DataFrame, rul: np.ndarray, sensors: list[str], *,
                   effect_sd: float = 0.4, seed: int = 0,
                   ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, dict]:
    """Two build standards; train on one, test on the other.

    The offset is a per-sensor constant applied to build B, which is the mild
    version of heterogeneity -- a real service bulletin changes the degradation
    *rate*, not just the level. Using the mild version on purpose: if even a
    constant offset costs real accuracy, the rate-change case does not need to be
    simulated to make the point, and simulating it would be inventing a
    degradation physics I cannot validate.
    """
    rng = np.random.default_rng(seed)
    uids = train["unit"].unique()
    rng.shuffle(uids)
    half = len(uids) // 2
    a, b = set(int(u) for u in uids[:half]), set(int(u) for u in uids[half:])

    offs = {c: float(rng.normal(0, effect_sd) * (train[c].std() or 1.0))
            for c in sensors}
    mask_a = train["unit"].isin(a).to_numpy()
    ta, tb = train.loc[mask_a].copy(), train.loc[~mask_a].copy()
    for c, o in offs.items():
        tb[c] = tb[c].to_numpy(dtype=float) + o
    return ta, rul[mask_a], tb, rul[~mask_a], {
        "build_a_units": len(a), "build_b_units": len(b),
        "offset_sd_fraction": effect_sd,
        "max_offset_in_sd": float(max(abs(v) / (train[c].std() or 1.0)
                                      for c, v in offs.items())),
    }
