"""Conformal lower bounds and survival analysis: the guarantees, checked on data
where the right answer is known."""
import numpy as np
import pandas as pd
import pytest

import conformal as CF
import survival as SV


# ---------------------------------------------------------------- conformal

def test_conformal_quantile_is_finite_sample_corrected():
    s = np.arange(1, 10, dtype=float)             # n = 9
    # ceil(10 * 0.9) = 9 -> the largest score, not the 90th percentile (8.2)
    assert CF.conformal_quantile(s, 0.10) == 9.0


def test_conformal_quantile_too_few_scores_is_infinite():
    assert CF.conformal_quantile(np.array([1.0, 2.0]), 0.10) == np.inf
    assert CF.conformal_quantile(np.array([]), 0.10) == np.inf


@pytest.mark.parametrize("kind", ["split", "mondrian", "cqr"])
def test_lower_bound_covers_at_least_nominal_on_exchangeable_data(kind):
    rng = np.random.default_rng(0)
    n = 20000
    y = rng.uniform(0, 125, n)
    # heteroscedastic, over-predicting model: the dangerous direction
    point = y + rng.normal(3, 2 + 0.1 * y)
    q_lo = point - 1.28 * (2 + 0.1 * y) * 0.5      # a mis-scaled quantile model
    cal, new = slice(0, n // 2), slice(n // 2, n)
    lb = CF.LowerBound(kind).calibrate(y[cal], point[cal], q_lo[cal])
    lo = lb.predict(point[new], q_lo[new])
    cov = np.mean(y[new] >= lo)
    assert 0.89 <= cov <= 0.92


def test_mondrian_holds_coverage_inside_each_band():
    rng = np.random.default_rng(1)
    n = 40000
    y = rng.uniform(0, 125, n)
    point = y + rng.normal(0, np.where(y < 25, 1.0, 15.0))   # tight near failure
    cal, new = slice(0, n // 2), slice(n // 2, n)
    split = CF.LowerBound("split").calibrate(y[cal], point[cal])
    mond = CF.LowerBound("mondrian").calibrate(y[cal], point[cal])
    low = point[new] < 25
    m_split = np.mean((point[new] - split.predict(point[new]))[low])
    m_mond = np.mean((point[new] - mond.predict(point[new]))[low])
    # the global correction is sized for the noisy band and wastes margin near failure
    assert m_mond < 0.5 * m_split
    b = CF.band_of(point[new])
    for i in range(len(CF.BANDS) - 1):
        m = b == i
        if m.sum() > 500:
            assert np.mean(y[new][m] >= mond.predict(point[new][m])) >= 0.88


def test_lower_bound_never_negative():
    lb = CF.LowerBound("split").calibrate(np.zeros(100), np.full(100, 50.0))
    assert lb.predict(np.array([1.0, 10.0])).min() >= 0.0


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        CF.LowerBound("bayes")


# ---------------------------------------------------------------- survival

def test_landmark_targets_mark_censoring():
    df = pd.DataFrame({"unit": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2]})
    dur, ev = SV.landmark_targets(df, failed_units={1})
    assert dur.tolist() == [3.0, 2.0, 1.0, 2.0, 1.0]
    assert ev.tolist() == [1, 1, 1, 0, 0]


def test_imputation_matches_monte_carlo():
    w = SV.WeibullRUL.__new__(SV.WeibullRUL)
    lam, rho = np.array([80.0, 150.0, 60.0]), 2.5
    w._params = lambda x: (lam, rho)
    d = np.array([30.0, 60.0, 200.0])
    got = w.imputed_capped_rul(pd.DataFrame(index=range(3)), d)
    rng = np.random.default_rng(0)
    for i in range(2):
        t = lam[i] * rng.weibull(rho, 1_000_000)
        t = t[t > d[i]]
        assert got[i] == pytest.approx(np.mean(np.minimum(t - 1, 125)), abs=0.3)
    assert got[2] == 125.0     # survived past the cap: the capped label is exact


def test_imputed_label_is_at_least_what_was_observed():
    w = SV.WeibullRUL.__new__(SV.WeibullRUL)
    w._params = lambda x: (np.full(50, 40.0), 3.0)
    d = np.linspace(1, 120, 50)
    got = w.imputed_capped_rul(pd.DataFrame(index=range(50)), d)
    assert np.all(got >= np.minimum(d - 1, 125) - 1e-9)


def test_weibull_recovers_a_known_median_under_censoring():
    rng = np.random.default_rng(3)
    n = 4000
    x = rng.normal(size=n)
    lam = np.exp(4.0 + 0.4 * x)
    t = lam * rng.weibull(3.0, n)
    c = rng.uniform(10, 150, n)
    dur, ev = np.minimum(t, c), (t <= c).astype(int)
    m = SV.WeibullRUL(penalizer=0.0).fit(pd.DataFrame({"x": x}), dur, ev)
    xs = pd.DataFrame({"x": [0.0]})
    true_median = np.exp(4.0) * np.log(2) ** (1 / 3.0) - 1
    assert m.predict_rul(xs, cap=None)[0] == pytest.approx(true_median, rel=0.05)


def test_concordance_extremes():
    t = np.array([5.0, 10.0, 20.0, 40.0])
    assert SV.concordance(t, t) == 1.0
    assert SV.concordance(t, -t) == 0.0
    assert SV.concordance(t, np.ones(4)) == 0.5


def test_survival_columns_are_compact():
    cols = ["cycle", "s2", "s2_rm5", "s2_rm20", "s2_sd20", "s2_delta", "s2_slope20"]
    assert SV.survival_columns(cols) == ["cycle", "s2_rm20", "s2_delta", "s2_slope20"]
