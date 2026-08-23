"""Tests for the third-pass modules.

The registry tests are the ones that matter. Everything else here checks that a
function does what it says; those check that the system REFUSES the thing it
exists to refuse, which is the only kind of guarantee a lineage mechanism can
offer.
"""
from __future__ import annotations

import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cmapss  # noqa: E402
import conceptdrift as CD  # noqa: E402
import mixture as MIX  # noqa: E402
import registry as REG  # noqa: E402
import robustness as ROB  # noqa: E402
import sequence_models as SEQ  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _frame(n_units=6, life=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, life + 1):
            rows.append({"unit": u, "cycle": c,
                         "op1": 0.0, "op2": 0.0, "op3": 100.0,
                         **{f"s{i}": rng.normal(i, 1.0) + 0.02 * c
                            for i in range(1, 22)}})
    return pd.DataFrame(rows)


class _Dummy:
    def predict(self, x):
        return np.zeros(len(x))


def _bundle(df, sensors=None):
    sensors = sensors or ["s2", "s3", "s4"]
    norm = cmapss.ConditionNormaliser().fit(df, sensors)
    return REG.ModelBundle(_Dummy(), norm, sensors, ["f1", "f2"], 1, 125, "gbm")


# ---------------------------------------------------------------------------
# registry: the refusals
# ---------------------------------------------------------------------------

def test_bundle_without_normaliser_is_refused():
    df = _frame()
    b = REG.ModelBundle(_Dummy(), None, ["s2"], ["f1"], 1, 125, "gbm")
    with pytest.raises(ValueError, match="no normaliser"):
        b.validate()


def test_normaliser_missing_a_sensor_is_refused():
    df = _frame()
    norm = cmapss.ConditionNormaliser().fit(df, ["s2"])
    b = REG.ModelBundle(_Dummy(), norm, ["s2", "s3"], ["f1"], 1, 125, "gbm")
    with pytest.raises(ValueError, match="cannot transform"):
        b.validate()


def test_swapped_normaliser_is_caught_by_the_fingerprint(tmp_path):
    """The failure the whole design exists to stop, and the only silent one."""
    df = _frame()
    reg = REG.Registry(tmp_path)
    meta = reg.save("m", _bundle(df))
    reg.load_bundle("m", meta["version"])          # sanity: loads clean

    other = cmapss.ConditionNormaliser().fit(_frame(seed=99), ["s2", "s3", "s4"])
    (tmp_path / "m" / "v1" / "normaliser.pkl").write_bytes(pickle.dumps(other))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        reg.load_bundle("m", meta["version"])


def test_missing_normaliser_file_is_refused_not_defaulted(tmp_path):
    df = _frame()
    reg = REG.Registry(tmp_path)
    reg.save("m", _bundle(df))
    (tmp_path / "m" / "v1" / "normaliser.pkl").unlink()
    with pytest.raises(FileNotFoundError, match="refusing to load"):
        reg.load_bundle("m", 1)


def test_versions_increment_and_production_stage_resolves(tmp_path):
    df = _frame()
    reg = REG.Registry(tmp_path)
    reg.save("m", _bundle(df))
    v2 = reg.save("m", _bundle(df), stage="Production")
    assert reg.versions("m") == [1, 2]
    assert reg.production("m") == v2["version"]


# ---------------------------------------------------------------------------
# concept drift
# ---------------------------------------------------------------------------

def test_acceleration_leaves_inputs_untouched_by_construction():
    rul = np.array([50.0, 40.0, 30.0])
    out, _ = CD.accelerate_degradation(rul, np.array([1, 1, 2]), factor=0.5)
    assert np.allclose(out, [25.0, 20.0, 15.0])


def test_residual_monitor_uses_standard_error_not_raw_sd():
    """A mean is compared against the SE of a mean. Regression for a real bug.

    With the per-observation SD the z is sqrt(n) too small and the monitor never
    fires -- which looks exactly like a calm process.
    """
    y = np.zeros(25)
    pred = np.full(25, 3.0)
    r = CD.residual_monitor(y, pred, baseline_mean=0.0, baseline_sd=5.0)
    assert r["standard_error"] == pytest.approx(5.0 / 5.0)
    assert r["z"] == pytest.approx(3.0)
    assert r["direction"] == "optimistic"


def test_page_hinkley_fires_on_a_shifted_stream_and_not_on_a_flat_one():
    flat = np.zeros(200)
    assert not CD.page_hinkley(flat)["fired"]
    shifted = np.concatenate([np.zeros(50), np.full(150, 4.0)])
    out = CD.page_hinkley(shifted)
    assert out["fired"] and out["at"] > 40


def test_bootstrap_ci_brackets_the_point_and_always_reports_width():
    v = np.random.default_rng(0).normal(10, 2, 40)
    out = CD.bootstrap_ci(v, np.median)
    assert out["lo"] <= out["point"] <= out["hi"]
    assert "width" in CD.bootstrap_ci(np.array([1.0]), np.median)


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------

def test_censoring_keeps_only_the_requested_number_of_failures():
    df = _frame(n_units=10)
    rul = np.linspace(100, 0, len(df))
    sub, sub_rul, info = ROB.censor_fleet(df, rul, n_failures=3)
    assert info["n_failures_observed"] == 3
    assert info["n_censored"] == 7
    assert info["rows_kept"] < info["rows_total"]
    assert len(sub) == len(sub_rul) == info["rows_kept"]


def test_informative_censoring_prefers_shorter_lived_units():
    """The bias is the point: real failures over-represent fast degradation."""
    rng = np.random.default_rng(0)
    rows = []
    for u in range(1, 21):
        for c in range(1, (10 if u <= 10 else 100) + 1):
            rows.append({"unit": u, "cycle": c, "op1": 0.0, "op2": 0.0,
                         "op3": 100.0, **{f"s{i}": 1.0 for i in range(1, 22)}})
    df = pd.DataFrame(rows)
    _, _, info = ROB.censor_fleet(df, np.zeros(len(df)), n_failures=8,
                                  informative=True, seed=1)
    short = sum(1 for u in info["failed_units"] if u <= 10)
    assert short >= 6, "informative censoring should favour short-lived units"


def test_drift_scales_with_life_not_with_a_fixed_cycle_count():
    """Regression: specifying drift per 1000 cycles made it invisible at n=200."""
    df = _frame(life=50)
    out, info = ROB.inject_sensor_drift(df, ["s2", "s3", "s4"],
                                        drift_over_life=1.0, n_replacements=0)
    sd = df["s2"].std()
    last = df["cycle"].max()
    delta = (out["s2"] - df["s2"]).to_numpy()
    assert delta[df["cycle"].to_numpy() == last].mean() == pytest.approx(sd, rel=0.3)


def test_label_noise_is_biased_late_not_symmetric():
    rul = np.full(400, 60.0)
    units = np.repeat(np.arange(20), 20)
    out, info = ROB.noisy_labels(rul, units, sd_cycles=5.0, late_bias_cycles=6.0)
    assert out.mean() > rul.mean(), "recorded failures are systematically late"


# ---------------------------------------------------------------------------
# mixture + sequence models
# ---------------------------------------------------------------------------

def test_soft_gate_spans_hard_split_to_uniform():
    d = np.array([[0.0, 3.0], [3.0, 0.0]])
    hard = MIX.soft_gate(d, temperature=0.01)
    soft = MIX.soft_gate(d, temperature=1000.0)
    assert hard[0, 0] > 0.99 and MIX.gate_entropy(hard) < 0.05
    assert MIX.gate_entropy(soft) > 0.99


def test_effective_sample_size_beats_a_hard_split():
    """The whole argument for a mixture: experts do not starve."""
    d = np.abs(np.random.default_rng(0).normal(0, 1, (100, 2)))
    ess = MIX.effective_sample_size(MIX.soft_gate(d, temperature=1.0))
    assert ess.min() > 50, "a soft gate should beat the 50/50 of a hard split"


def test_blend_weights_stay_on_the_simplex():
    y = np.arange(50, dtype=float)
    preds = {"a": y + 1, "b": y - 1, "c": y * 2}
    out = SEQ.blend_weights(preds, y, step=0.1)
    w = out["weights"]
    assert all(v >= -1e-9 for v in w.values())
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


def test_tcn_feature_map_is_causal():
    """A feature at position t must not move when inputs AFTER t change.

    This is the chomp, tested where it actually lives. A TCN pads both ends of
    each dilated conv and trims the surplus off the right; forget the trim and
    position t sees the future. The symptom is a validation score that improves,
    which is the direction nobody investigates.

    Tested on the internal feature map rather than on the scalar head, because
    the head reads only the final timestep -- where there is no future left to
    leak, so the head cannot expose the bug.
    """
    import torch
    torch.manual_seed(0)
    m = SEQ.RULTCN(n_features=3, channels=(8, 8))
    m.eval()
    x = torch.randn(1, 30, 3)
    with torch.no_grad():
        h1 = m.tcn(x.transpose(1, 2))
        x2 = x.clone()
        x2[:, 20:, :] += 10.0                    # perturb the future only
        h2 = m.tcn(x2.transpose(1, 2))
    assert torch.allclose(h1[:, :, :20], h2[:, :, :20], atol=1e-6),         "positions before the perturbation must be unchanged"
    assert not torch.allclose(h1[:, :, 20:], h2[:, :, 20:], atol=1e-6),         "positions at and after the perturbation must move"


def test_tcn_receptive_field_is_what_it_claims():
    """Inputs outside the receptive field must not reach the prediction.

    Not a bug hunt -- a documentation check. The head reads the last timestep, so
    with channels=(8,8) and kernel 3 it can see 13 cycles and no more. Asserting
    that keeps `receptive_field` honest, and it is the number that decides whether
    the window length is doing anything.
    """
    import torch
    torch.manual_seed(0)
    m = SEQ.RULTCN(n_features=3, channels=(8, 8))
    m.eval()
    rf = m.receptive_field
    x = torch.randn(1, rf + 12, 3)
    with torch.no_grad():
        base = float(m(x))
        inside = x.clone()
        inside[:, -1, :] += 5.0
        outside = x.clone()
        outside[:, : x.shape[1] - rf, :] += 5.0
        assert float(m(inside)) != pytest.approx(base), "the last step must matter"
        assert float(m(outside)) == pytest.approx(base, abs=1e-6),             "steps older than the receptive field must not reach the output"
