"""ML-1 tests: the metrics and target engineering everything else is scored with.

A wrong PHM score or a leaky feature would not crash anything — it would just make
every table in RESULTS.md wrong in a way that reads fine. These are the tests that
would notice.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import alarm  # noqa: E402
import cmapss  # noqa: E402
import features  # noqa: E402
import metrics  # noqa: E402


# ---------------------------------------------------------------- metrics

def test_phm_score_penalises_late_more_than_early():
    """The asymmetry is the whole reason the metric exists."""
    late = metrics.phm_score([100.0], [130.0])     # predicted failure later than truth
    early = metrics.phm_score([100.0], [70.0])
    assert late > early
    # exp(30/10)-1 vs exp(30/13)-1
    assert late == pytest.approx(np.exp(30 / 10) - 1, rel=1e-9)
    assert early == pytest.approx(np.exp(30 / 13) - 1, rel=1e-9)


def test_phm_score_is_zero_on_perfect_predictions():
    assert metrics.phm_score([10.0, 50.0], [10.0, 50.0]) == pytest.approx(0.0)


def test_phm_score_is_a_sum_not_a_mean():
    one = metrics.phm_score([100.0], [110.0])
    two = metrics.phm_score([100.0, 100.0], [110.0, 110.0])
    assert two == pytest.approx(2 * one)


def test_rmse_matches_numpy():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.5, 1.0, 4.0])
    assert metrics.rmse(a, b) == pytest.approx(np.sqrt(np.mean((a - b) ** 2)))


def test_alpha_lambda_floored_cone_is_looser_than_strict():
    true = np.arange(100, 0, -1, dtype=float)
    pred = true + 2.0                      # a constant 2-cycle error
    r = metrics.alpha_lambda(true, pred, alpha=0.2, min_cone_cycles=5.0)
    # Strict fails near the end (0.2*3 = 0.6 < 2.0); floored passes throughout.
    assert r["frac_inside_cone"] > r["frac_inside_cone_strict"]
    assert r["converged"]
    assert not r["converged_strict"]


def test_alpha_lambda_perfect_prediction_converges_under_both():
    true = np.arange(80, 0, -1, dtype=float)
    r = metrics.alpha_lambda(true, true.copy())
    assert r["converged"] and r["converged_strict"]
    assert r["lambda_fraction"] == pytest.approx(0.0)


def test_horizon_strata_partition_the_data():
    true = np.array([5.0, 25.0, 60.0, 100.0, 120.0])
    pred = true + 1
    rows = metrics.horizon_strata(true, pred)
    assert sum(r["n"] for r in rows) == len(true)


# ---------------------------------------------------------------- targets

def test_piecewise_rul_caps_and_counts_down_to_zero():
    df = pd.DataFrame({"unit": [1] * 200, "cycle": np.arange(1, 201)})
    r = cmapss.piecewise_rul(df, cap=125)
    assert r.max() == 125
    assert r.iloc[-1] == 0
    assert r.iloc[-11] == 10


def test_uncapped_rul_reaches_full_life():
    df = pd.DataFrame({"unit": [1] * 200, "cycle": np.arange(1, 201)})
    assert cmapss.piecewise_rul(df, cap=None).max() == 199


def test_rul_is_computed_per_unit_not_across_units():
    df = pd.DataFrame({"unit": [1] * 50 + [2] * 30,
                       "cycle": list(range(1, 51)) + list(range(1, 31))})
    r = cmapss.piecewise_rul(df, cap=None)
    assert r.iloc[49] == 0        # last cycle of unit 1
    assert r.iloc[79] == 0        # last cycle of unit 2
    assert r.iloc[50] == 29       # first cycle of unit 2


# ---------------------------------------------------------------- features

def test_rolling_slope_recovers_a_known_gradient():
    y = pd.Series(np.arange(100, dtype=float) * 3.0)
    s = features._rolling_slope(y, 20)
    assert s.iloc[-1] == pytest.approx(3.0, rel=1e-9)
    assert s.iloc[19] == pytest.approx(3.0, rel=1e-9)
    assert np.isnan(s.iloc[18])   # not enough history yet


def test_features_never_look_across_unit_boundaries():
    """The leak that would make every result meaningless."""
    df = pd.DataFrame({
        "unit": [1] * 30 + [2] * 30,
        "cycle": list(range(1, 31)) * 2,
        "s1": [0.0] * 30 + [100.0] * 30,
    })
    f = features.build_features(df, ["s1"])
    # Unit 2's first row must see only unit 2 data: its delta-from-first is 0 and
    # its rolling mean is its own value, not a blend with unit 1's zeros.
    assert f["s1_delta"].iloc[30] == pytest.approx(0.0)
    assert f["s1_rm5"].iloc[30] == pytest.approx(100.0)


def test_sequence_windows_pad_with_the_first_observation():
    df = pd.DataFrame({"unit": [1] * 5, "cycle": range(1, 6),
                       "s1": [7.0, 1.0, 2.0, 3.0, 4.0]})
    x, _, idx = features.sequence_windows(df, ["s1"], window=3)
    assert x.shape == (5, 3, 1)
    # First row is edge-padded with its own first value, never with zeros.
    assert np.allclose(x[0, :, 0], [7.0, 7.0, 7.0])
    assert np.allclose(x[2, :, 0], [7.0, 1.0, 2.0])


# ---------------------------------------------------------------- alarm policy

def test_sustained_alarm_and_lead_time():
    # RUL prediction falls below 30 at index 70 and stays there to failure at 99.
    pred = np.concatenate([np.full(70, 60.0), np.full(30, 20.0)])
    r = alarm.evaluate_unit(pred, threshold=30, k=3)
    assert not r["missed"]
    assert r["lead_time"] == pytest.approx(99 - 72)   # asserts once k satisfied
    assert r["nuisance_alarms"] == 0


def test_nuisance_alarm_is_counted_when_it_clears():
    pred = np.concatenate([np.full(20, 60.0), np.full(5, 10.0),
                           np.full(45, 60.0), np.full(30, 20.0)])
    r = alarm.evaluate_unit(pred, threshold=30, k=3)
    assert r["nuisance_alarms"] == 1
    assert not r["missed"]


def test_alarm_that_never_fires_is_a_miss():
    r = alarm.evaluate_unit(np.full(100, 90.0), threshold=30, k=3)
    assert r["missed"]
    assert r["lead_time"] == 0.0


def test_persistence_trades_lead_time_for_nuisance_alarms():
    rng = np.random.default_rng(0)
    units = []
    for _ in range(30):
        base = np.concatenate([np.full(70, 60.0), np.linspace(55, 5, 30)])
        units.append(base + rng.normal(0, 12, 100))
    k1 = alarm.cost_of_policy(units, 30, 1)
    k5 = alarm.cost_of_policy(units, 30, 5)
    assert k1["nuisance_per_unit"] >= k5["nuisance_per_unit"]
    assert k1["lead_mean"] >= k5["lead_mean"]
