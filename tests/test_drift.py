"""Tests for the drift monitor and the retrain trigger."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import drift  # noqa: E402


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    b = rng.normal(size=5000)
    assert drift.psi(a, b) < 0.05
    assert drift.band(drift.psi(a, b)) == "stable"


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=8000)
    vals = [drift.psi(ref, rng.normal(loc=s, size=8000)) for s in (0.2, 0.5, 1.0, 2.0)]
    assert vals == sorted(vals), vals
    assert drift.band(vals[-1]) == "significant"


def test_psi_bins_come_from_the_reference_not_the_pooled_data():
    """Bin edges must be frozen with the reference.

    If edges are re-derived from the combined data, a shift can hide inside the
    binning -- the bins move with the data and the statistic under-reports. This
    checks the edges are reference-derived by confirming a pure location shift is
    still detected when the two samples have equal size.
    """
    rng = np.random.default_rng(2)
    ref = rng.normal(size=4000)
    cur = rng.normal(loc=1.5, size=4000)
    assert drift.psi(ref, cur) > 0.25


def test_trigger_needs_persistence_not_one_bad_window():
    t = drift.RetrainTrigger(psi_threshold=0.25, k_windows=2, min_features=3)
    big = [{"feature": f"f{i}", "psi": 0.9, "band": "significant"} for i in range(5)]
    r1 = t.evaluate("w1", big, 0.9, 0.2, 0.1)
    assert r1["breached"] and not r1["RETRAIN"], "one window must not retrain"
    r2 = t.evaluate("w2", big, 0.9, 0.2, 0.1)
    assert r2["RETRAIN"], "two consecutive breaches must retrain"


def test_trigger_scope_rule_distinguishes_a_sensor_from_a_model():
    """One drifting feature is a SENSOR ticket, not a model ticket."""
    t = drift.RetrainTrigger(psi_threshold=0.25, k_windows=1, min_features=3)
    one = ([{"feature": "f0", "psi": 3.0, "band": "significant"}]
           + [{"feature": f"f{i}", "psi": 0.01, "band": "stable"} for i in range(1, 6)])
    r = t.evaluate("w", one, 0.01, 0.10, 0.10)
    assert not r["breached"], "a single drifting feature must not trigger a retrain"


def test_alarm_rate_ratio_alone_can_trigger():
    """The label-free signal tied to a cost must be able to fire on its own."""
    t = drift.RetrainTrigger(k_windows=1)
    quiet = [{"feature": f"f{i}", "psi": 0.01, "band": "stable"} for i in range(6)]
    r = t.evaluate("w", quiet, 0.01, 0.30, 0.10)   # alarm rate tripled
    assert r["breached"]
    assert any("alarm rate" in s for s in r["reasons"]), r["reasons"]


def test_psi_handles_degenerate_input():
    assert np.isnan(drift.psi(np.array([1.0, 2.0]), np.array([1.0])))
    assert drift.band(float("nan")) == "unknown"
