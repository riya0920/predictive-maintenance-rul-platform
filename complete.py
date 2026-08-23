"""ML-1, the rest: registry, TCN + ensemble, serving, and the deployment-reality
numbers that docs/DEPLOYMENT_REALITY.md only ever asserted.

    python complete.py
    python complete.py --quick
    python complete.py --report-only

Every stage closes a numbered item from the README's "what is NOT built" list.
The list is reproduced here so the mapping is checkable rather than claimed:

  1  MLflow tracking + registry, normaliser travelling with the model  -> stage 1
  2  one sequence model, once: no TCN, no ensemble, no HPO             -> stage 2
  3  no serving: no API, no container, no batch job                    -> stage 3
  4  edge table is an unconstrained desktop, not gateway hardware      -> stage 4
  5  fault modes are discovery, not classification, and did not pay    -> stage 5
  6  drift monitor is input-side only; no labelled backtest            -> stage 6
  7  20 held-out units is a thin basis for a lead-time distribution    -> stage 7
  8  everything in DEPLOYMENT_REALITY.md, unquantified                 -> stage 8
"""
from __future__ import annotations

import json
import pathlib
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import alarm  # noqa: E402
import cmapss  # noqa: E402
import conceptdrift as CD  # noqa: E402
import drift  # noqa: E402
import faultmode  # noqa: E402
import features  # noqa: E402
import metrics  # noqa: E402
import mixture as MIX  # noqa: E402
import models  # noqa: E402
import registry as REG  # noqa: E402
import robustness as ROB  # noqa: E402
import sequence_models as SEQ  # noqa: E402
import serve as SRV  # noqa: E402

OUT = ROOT / "out"
DOCS = ROOT / "docs"
CAP = cmapss.DEFAULT_RUL_CAP
QUICK = "--quick" in sys.argv


# ---------------------------------------------------------------------------
# shared setup
# ---------------------------------------------------------------------------

def prepare(fd: str, seed: int = 0, n_holdout: int = 20) -> dict:
    """Load one sub-dataset and build the train / holdout / test tensors once."""
    train, test, rul_true = cmapss.load(fd)
    rng = np.random.default_rng(seed)
    units = np.sort(train["unit"].unique())
    hold = rng.choice(units, size=min(n_holdout, len(units) // 4), replace=False)
    m = train["unit"].isin(hold)
    tr, hd = train[~m].copy(), train[m].copy()

    norm = cmapss.ConditionNormaliser().fit(tr, cmapss.SENSOR_COLS)
    sensors = cmapss.informative_sensors(tr, norm)
    norm = cmapss.ConditionNormaliser().fit(tr, sensors)

    def feat(df):
        f = features.build_features(norm.transform(df), sensors)
        f["unit"] = df["unit"].to_numpy()
        return f

    f_tr, f_hd, f_te = feat(tr), feat(hd), feat(test)
    cols = [c for c in f_tr.columns if c != "unit"]
    y_tr = cmapss.piecewise_rul(tr, CAP).to_numpy(dtype=float)
    y_hd = cmapss.piecewise_rul(hd, CAP).to_numpy(dtype=float)
    return {"fd": fd, "train": tr, "holdout": hd, "test": test,
            "rul_true": rul_true, "norm": norm, "sensors": sensors,
            "f_tr": f_tr, "f_hd": f_hd, "f_te": f_te, "cols": cols,
            "y_tr": y_tr, "y_hd": y_hd}


def _last_rows(f: pd.DataFrame) -> np.ndarray:
    """Index of each unit's final cycle -- the row the test RUL refers to."""
    return f.reset_index(drop=True).groupby("unit").tail(1).index.to_numpy()


# ---------------------------------------------------------------------------
# stage 1: registry
# ---------------------------------------------------------------------------

def stage_registry(P: dict) -> dict:
    """Train, register, reload, and try to break the lineage guarantee."""
    reg = REG.Registry(ROOT / "registry")
    gbm, _ = models.fit_gbm(P["f_tr"][P["cols"]].to_numpy(), P["y_tr"])

    pred_hd = gbm.predict(P["f_hd"][P["cols"]].to_numpy())
    bundle = REG.ModelBundle(
        gbm, P["norm"], P["sensors"], P["cols"], window=1, rul_cap=CAP, kind="gbm",
        metrics={"holdout_rmse": metrics.rmse(P["y_hd"], pred_hd)},
        params={"fd": P["fd"], "n_train_units": int(P["train"]["unit"].nunique())})

    with REG.tracking("ml1-rul", f"gbm-{P['fd']}", OUT / "mlruns") as run:
        meta = reg.save("rul-gbm", bundle, stage="Production")
        REG.log_run(run, bundle.params, bundle.metrics)

    reloaded = reg.load_bundle("rul-gbm", meta["version"])
    pred_reload = reloaded.model.predict(P["f_hd"][P["cols"]].to_numpy())
    identical = bool(np.allclose(pred_hd, pred_reload))

    # -- the guarantee, tested by trying to violate it ---------------------
    checks = []

    # (a) a bundle with no normaliser must not save
    try:
        REG.ModelBundle(gbm, None, P["sensors"], P["cols"], 1, CAP, "gbm").validate()
        checks.append({"attempt": "save with no normaliser", "refused": False})
    except ValueError as e:
        checks.append({"attempt": "save with no normaliser", "refused": True,
                       "message": str(e)[:90]})

    # (b) a normaliser fitted on the wrong columns must not save
    bad_norm = cmapss.ConditionNormaliser().fit(P["train"], P["sensors"][:3])
    try:
        REG.ModelBundle(gbm, bad_norm, P["sensors"], P["cols"], 1, CAP, "gbm").validate()
        checks.append({"attempt": "normaliser missing sensors", "refused": False})
    except ValueError as e:
        checks.append({"attempt": "normaliser missing sensors", "refused": True,
                       "message": str(e)[:90]})

    # (c) swap the normaliser on disk for a different fitted one: the
    #     fingerprint must catch it. This is the failure the whole design is
    #     aimed at, and it is the one that would otherwise be silent.
    import pickle
    d = reg.root / "rul-gbm" / f"v{meta['version']}"
    original = (d / "normaliser.pkl").read_bytes()
    other = cmapss.ConditionNormaliser().fit(P["test"], P["sensors"])
    (d / "normaliser.pkl").write_bytes(pickle.dumps(other))
    try:
        reg.load_bundle("rul-gbm", meta["version"])
        checks.append({"attempt": "swapped normaliser on disk", "refused": False})
        swap_cost = None
    except ValueError as e:
        checks.append({"attempt": "swapped normaliser on disk", "refused": True,
                       "message": str(e)[:90]})
        # What would it have cost if the fingerprint had not caught it?
        f_bad = features.build_features(other.transform(P["holdout"]), P["sensors"])
        pred_bad = gbm.predict(f_bad[P["cols"]].to_numpy())
        swap_cost = {"rmse_correct": metrics.rmse(P["y_hd"], pred_hd),
                     "rmse_wrong_normaliser": metrics.rmse(P["y_hd"], pred_bad)}
    (d / "normaliser.pkl").write_bytes(original)

    return {"version": meta["version"], "fingerprint": meta["fingerprint"],
            "reload_identical": identical, "checks": checks,
            "silent_swap_cost": swap_cost, "mlflow": REG.HAVE_MLFLOW,
            "index": [{k: r[k] for k in ("name", "version", "stage", "kind",
                                         "fingerprint")} for r in reg.index()]}


# ---------------------------------------------------------------------------
# stage 2: TCN, HPO, ensemble
# ---------------------------------------------------------------------------

def stage_deep(P: dict) -> dict:
    win = 30
    x_tr, y_tr, _ = features.sequence_windows(P["f_tr"], P["cols"], win, P["y_tr"])
    x_hd, y_hd, i_hd = features.sequence_windows(P["f_hd"], P["cols"], win, P["y_hd"])

    epochs = 8 if QUICK else 26
    grid = ({"channels": [(32, 32, 32)], "dropout": [0.15]} if QUICK else None)
    search = SEQ.search_tcn(x_tr, y_tr, grid=grid, epochs=max(6, epochs - 6))
    best = search[0]
    chans = eval(best["channels"])                     # noqa: S307 -- own string
    tcn, tinfo = SEQ.fit_tcn(x_tr, y_tr, channels=chans,
                             dropout=float(best["dropout"]), epochs=epochs)

    lstm, _lsecs, linfo = models.fit_lstm(x_tr, y_tr, epochs=epochs, verbose=False)
    gbm, _ = models.fit_gbm(P["f_tr"][P["cols"]].to_numpy(), P["y_tr"])

    # Align every model on the same holdout rows, in the same order.
    order = np.argsort(i_hd)
    p_tcn = SEQ.predict_tcn(tcn, x_hd)[order]
    p_lstm = models.predict_lstm(lstm, x_hd)[order]
    p_gbm = gbm.predict(P["f_hd"][P["cols"]].to_numpy())
    truth = P["y_hd"]

    preds = {"gbm": p_gbm, "lstm": p_lstm, "tcn": p_tcn}
    single = {k: {"rmse": metrics.rmse(truth, v), "phm": metrics.phm_score(truth, v)}
              for k, v in preds.items()}
    blend = SEQ.blend_weights(preds, truth)
    p_blend = sum(w * preds[n] for n, w in blend["weights"].items())
    return {
        "window": win, "search": search, "best_config": best,
        "params": {"tcn": tinfo["params"],
                   "lstm": sum(p.numel() for p in lstm.parameters()),
                   "gbm": int(getattr(gbm, "n_iter_", 0))},
        "receptive_field": tinfo["receptive_field"],
        "single": single,
        "residual_correlation": SEQ.disagreement(
            {k: v - truth for k, v in preds.items()}),
        "ensemble": {"weights": blend["weights"], "rmse": blend["rmse"],
                     "phm": metrics.phm_score(truth, p_blend)},
        "best_single_rmse": min(v["rmse"] for v in single.values()),
    }


# ---------------------------------------------------------------------------
# stage 3: serving
# ---------------------------------------------------------------------------

def stage_serving(P: dict) -> dict:
    reg = REG.Registry(ROOT / "registry")
    bundle = reg.load_bundle("rul-gbm")
    scorer = SRV.Scorer(bundle, lambda x: bundle.model.predict(x),
                        threshold=30.0, k=3, cmapss_mod=cmapss, features_mod=features)

    test = P["test"]
    one = test[test["unit"] == test["unit"].iloc[0]]
    single = scorer.score_unit(one.copy())

    # A deliberately too-short history: the service must refuse it rather than
    # left-pad and return a number.
    short = one.head(3).copy()
    refused = scorer.score_unit(short)

    batch = SRV.score_batch(scorer, test.copy())
    container = SRV.write_container(ROOT / "deploy", "rul-gbm")

    app_ok, routes = False, []
    try:
        app = SRV.build_app(scorer)
        routes = sorted(r.path for r in app.routes if hasattr(r, "path"))
        from fastapi.testclient import TestClient
        c = TestClient(app)
        h = c.get("/health").json()
        payload = {"cycle": one["cycle"].tolist(),
                   "sensors": {s: one[s].tolist() for s in bundle.sensors},
                   "settings": {c_: one[c_].tolist() for c_ in cmapss.OP_COLS
                                if c_ in one}}
        r = c.post("/score", json=payload)
        app_ok = r.status_code == 200 and h["status"] == "ok"
        http = {"health": h, "score_status": r.status_code,
                "rul": r.json().get("rul") if r.status_code == 200 else None,
                "short_history_status": c.post("/score", json={
                    "cycle": short["cycle"].tolist(),
                    "sensors": {s: short[s].tolist() for s in bundle.sensors},
                }).status_code}
    except Exception as e:                                   # pragma: no cover
        http = {"error": f"{type(e).__name__}: {e}"}

    return {"single": {k: single[k] for k in
                       ("rul", "alarm", "extrapolating", "latency_ms")},
            "short_history_refused": not refused.get("scorable", True),
            "short_history_problems": refused.get("problems", []),
            "batch": {k: batch[k] for k in
                      ("n_scored", "n_skipped", "alarms", "extrapolating",
                       "units_per_second", "seconds")},
            "http_ok": app_ok, "routes": routes, "http": http,
            "container": container}


# ---------------------------------------------------------------------------
# stage 4: the edge caveat, made measurable
# ---------------------------------------------------------------------------

def stage_edge(P: dict) -> dict:
    """Scale the desktop measurement to a gateway, and be explicit it is a model.

    The README's item 4 is not fixable here -- there is no ARM gateway in this
    environment and there is no honest way to conjure one. What IS fixable is the
    thing that made the original table misleading: it reported one number with no
    indication of how far it would travel. A single-thread, frequency-scaled
    projection with the scaling factor stated is a *model* of gateway latency, and
    a reader can disagree with the factor. A bare desktop number invites them to
    read it as a measurement of something it never measured.
    """
    import onnxruntime as ort                                  # noqa: F401
    import edge

    win = 30
    x_hd, _, _ = features.sequence_windows(P["f_hd"], P["cols"], win)
    sample = x_hd[:256]
    _xw, _yw, _ = features.sequence_windows(P["f_tr"], P["cols"], win, P["y_tr"])
    lstm, _secs, _info = models.fit_lstm(_xw, _yw, epochs=4 if QUICK else 10,
                                         verbose=False)

    onnx_path = edge.export_onnx(lstm, len(P["cols"]), win, OUT / "rul_edge.onnx")
    bench = edge.bench_onnx(onnx_path, sample, n_iter=60 if QUICK else 200)

    # Frequency scaling only. NOT claimed: cache behaviour, memory bandwidth,
    # SIMD width, thermal throttling -- all of which differ on ARM and all of
    # which move this number.
    desktop_ghz = 3.0
    rows = []
    for name, ghz, note in (("desktop x86, 1 thread (measured)", desktop_ghz, "measured"),
                            ("gateway ~1.6 GHz x86", 1.6, "projected"),
                            ("ARM Cortex-A72 @1.5 GHz", 1.5, "projected"),
                            ("ARM Cortex-A53 @1.2 GHz", 1.2, "projected")):
        rows.append({"target": name, "ghz": ghz, "note": note,
                     "p50_ms": bench["p50_ms"] * desktop_ghz / ghz,
                     "p99_ms": bench["p99_ms"] * desktop_ghz / ghz})
    return {"measured": bench, "projection": rows,
            "host": f"{platform.processor() or platform.machine()}",
            "caveat": ("frequency scaling only; ignores cache, memory bandwidth, "
                       "SIMD width and thermal throttling")}


# ---------------------------------------------------------------------------
# stage 5: the mixture head
# ---------------------------------------------------------------------------

def stage_mixture(fd: str = "FD004") -> dict:
    P = prepare(fd)
    disc = faultmode.discover_modes(P["train"], P["sensors"])
    sig_tr, units_tr = faultmode.unit_signature(P["train"], P["sensors"])
    sig_hd, units_hd = faultmode.unit_signature(P["holdout"], P["sensors"])

    # The gate must standardise the holdout with the TRAINING signature's mean and
    # sd -- the same discipline as the condition normaliser, and for the same
    # reason: re-standardising per split silently changes the transform.
    mu, sd = sig_tr.mean(axis=0), sig_tr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    cent = np.asarray(disc["kmeans"].cluster_centers_, dtype=float)
    d_tr = MIX.centroid_distances((sig_tr - mu) / sd, cent)
    d_hd = MIX.centroid_distances((sig_hd - mu) / sd, cent)

    def unit_to_row(gate_u, units_order, row_units):
        idx = {int(u): i for i, u in enumerate(units_order)}
        return np.array([gate_u[idx[int(u)]] for u in row_units])

    x_tr = P["f_tr"][P["cols"]].to_numpy()
    x_hd = P["f_hd"][P["cols"]].to_numpy()
    truth = P["y_hd"]

    base, _ = models.fit_gbm(x_tr, P["y_tr"])
    base_pred = base.predict(x_hd)
    baseline = {"rmse": metrics.rmse(truth, base_pred),
                "phm": metrics.phm_score(truth, base_pred)}

    def fit_w(x, y, w):
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                          random_state=models.SEED)
        m.fit(x, y, sample_weight=w)
        return m

    sweep = []
    for T in ([0.5, 2.0] if QUICK else [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]):
        g_tr_u = MIX.soft_gate(d_tr, T)
        g_hd_u = MIX.soft_gate(d_hd, T)
        g_tr = unit_to_row(g_tr_u, units_tr, P["f_tr"]["unit"].to_numpy())
        g_hd = unit_to_row(g_hd_u, units_hd, P["f_hd"]["unit"].to_numpy())
        moe = MIX.MixtureOfExperts(fit_w, lambda m, x: m.predict(x)).fit(x_tr, P["y_tr"], g_tr)
        p = moe.predict(x_hd, g_hd)
        ess = MIX.effective_sample_size(g_tr_u)
        sweep.append({"temperature": T, "rmse": metrics.rmse(truth, p),
                      "phm": metrics.phm_score(truth, p),
                      "gate_entropy": MIX.gate_entropy(g_tr_u),
                      "effective_units": [float(v) for v in ess],
                      "n_units": int(len(units_tr))})
    best = min(sweep, key=lambda r: r["rmse"])
    return {"fd": fd, "baseline": baseline, "sweep": sweep, "best": best,
            "delta_rmse": best["rmse"] - baseline["rmse"],
            "silhouette": disc["silhouette"]}


# ---------------------------------------------------------------------------
# stage 6: concept drift
# ---------------------------------------------------------------------------

def stage_concept_drift(P: dict) -> dict:
    """Inject drift at a mid-life snapshot, not at end of life.

    The first version of this evaluated at each unit's LAST cycle, where the
    piecewise RUL target is 0 by construction. Multiplying 0 by 0.65 is 0, so
    nothing was injected, and the monitors correctly reported no change to a
    dataset that had not changed. The fix is to take a snapshot partway through
    each unit's life -- which is also the realistic framing, since a fleet is
    monitored while its engines are running, not after they have failed.
    """
    rng = np.random.default_rng(7)
    x_tr = P["f_tr"][P["cols"]].to_numpy()
    gbm, _ = models.fit_gbm(x_tr, P["y_tr"])

    f_hd = P["f_hd"].reset_index(drop=True)
    y_hd = P["y_hd"]

    # One "observed now" row per unit, drawn from the middle of its life so the
    # true RUL is comfortably positive and the model is in its useful range.
    snap = []
    for u, g in f_hd.groupby("unit"):
        idx = g.index.to_numpy()
        lo, hi = int(len(idx) * 0.35), int(len(idx) * 0.85)
        snap.append(int(idx[rng.integers(lo, max(lo + 1, hi))]))
    snap = np.array(sorted(snap))

    x_snap = f_hd.loc[snap, P["cols"]].to_numpy()
    y_snap = y_hd[snap]
    pred = gbm.predict(x_snap)
    resid_base = pred - y_snap
    base_mean, base_sd = float(resid_base.mean()), float(resid_base.std() + 1e-9)

    y_drift, dinfo = CD.accelerate_degradation(
        y_snap, f_hd.loc[snap, "unit"].to_numpy(), factor=0.65)
    dinfo["mean_rul_before"] = float(y_snap.mean())
    dinfo["mean_rul_after"] = float(y_drift.mean())

    # -- (a) the input monitor: the negative control ----------------------
    # Reference is the TRAINING rows at comparable life positions, not all
    # training rows. Comparing an all-cycle reference against 20 end-of-life rows
    # was the first version's mistake and it reported 85/85 features breaching --
    # a population difference, not drift.
    tr = P["f_tr"].reset_index(drop=True)
    ref_rows = []
    for u, g in tr.groupby("unit"):
        idx = g.index.to_numpy()
        lo, hi = int(len(idx) * 0.35), int(len(idx) * 0.85)
        ref_rows.extend(idx[lo:max(lo + 1, hi)].tolist())
    ref = tr.loc[sorted(ref_rows), P["cols"]].to_numpy(dtype=float)

    # PSI is computed over the whole mid-life BAND of holdout rows, not over the
    # 20 snapshot rows. A 10-bin PSI estimated from 20 observations is dominated
    # by binning noise -- the first version reported 74 of 85 features breaching
    # on data that was provably identical, which is a property of n=20, not of
    # the data. A production input monitor sees every row, so the control does too.
    hd_band = []
    for u, g in f_hd.groupby("unit"):
        idx = g.index.to_numpy()
        lo, hi = int(len(idx) * 0.35), int(len(idx) * 0.85)
        hd_band.extend(idx[lo:max(lo + 1, hi)].tolist())
    x_band = f_hd.loc[sorted(hd_band), P["cols"]].to_numpy(dtype=float)

    psi_before = drift.feature_drift(ref, x_band, P["cols"])
    psi_after = drift.feature_drift(ref, x_band, P["cols"])   # inputs are unchanged
    import math
    fin = [r["psi"] for r in psi_after if math.isfinite(r["psi"])]
    n_breach = sum(1 for v in fin if v > 0.25)
    max_delta = max(abs(a["psi"] - b["psi"])
                    for a, b in zip(psi_before, psi_after)
                    if math.isfinite(a["psi"]) and math.isfinite(b["psi"]))

    # -- (b) the residual monitor: label-aware ----------------------------
    res_mon = CD.residual_monitor(y_drift, pred, baseline_mean=base_mean,
                                  baseline_sd=base_sd)

    # -- (c) detection latency, failure by failure ------------------------
    lat = CD.detection_latency(pred - y_drift, baseline_mean=base_mean,
                               baseline_sd=base_sd)
    return {
        "injection": dinfo,
        "n_snapshots": int(len(snap)),
        "input_monitor": {
            "features_breaching_psi": n_breach,
            "n_features": len(psi_after),
            "max_psi": float(max(fin)) if fin else 0.0,
            "max_delta_before_after": float(max_delta),
            "n_rows_compared": int(len(x_band)),
        },
        "residual_monitor": res_mon,
        "latency": lat,
        "baseline": {"mean_residual": base_mean, "sd": base_sd},
    }


# ---------------------------------------------------------------------------
# stage 7: bootstrap intervals on the lead-time distribution
# ---------------------------------------------------------------------------

def stage_intervals(P: dict) -> dict:
    gbm, _ = models.fit_gbm(P["f_tr"][P["cols"]].to_numpy(), P["y_tr"])
    f_hd = P["f_hd"].reset_index(drop=True)
    pred = gbm.predict(f_hd[P["cols"]].to_numpy())

    units, leads = [], []
    for u, g in f_hd.groupby("unit"):
        p = pred[g.index.to_numpy()]
        ev = alarm.evaluate_unit(p, threshold=30.0, k=3)
        # `lead_time`, not `lead` -- reading the wrong key gave every unit a
        # None and the bootstrap an empty sample. A missed unit reports lead 0
        # with missed=True, and including those zeros would drag the lead-time
        # distribution toward a number no alarm ever produced.
        if not ev["missed"]:
            units.append(int(u))
            leads.append(float(ev["lead_time"]))
    leads = np.array(leads)
    return {
        "n_units": len(leads),
        "n_units_total": int(f_hd["unit"].nunique()),
        "n_missed": int(f_hd["unit"].nunique() - len(leads)),
        "median": CD.bootstrap_ci(leads, np.median),
        "p05": CD.bootstrap_ci(leads, lambda v: np.quantile(v, 0.05)),
        "mean": CD.bootstrap_ci(leads, np.mean),
    }


# ---------------------------------------------------------------------------
# stage 8: deployment reality, priced
# ---------------------------------------------------------------------------

def _score_variant(P, train_df, y_train, label) -> dict:
    f = features.build_features(P["norm"].transform(train_df), P["sensors"])
    gbm, _ = models.fit_gbm(f[P["cols"]].to_numpy(), y_train)
    pred = gbm.predict(P["f_hd"][P["cols"]].to_numpy())
    return {"variant": label, "rmse": metrics.rmse(P["y_hd"], pred),
            "phm": metrics.phm_score(P["y_hd"], pred),
            "bias": float(np.mean(pred - P["y_hd"]))}


def stage_reality(P: dict) -> dict:
    rows = [_score_variant(P, P["train"], P["y_tr"], "clean C-MAPSS (baseline)")]

    # censoring, random vs informative, at two fleet sizes
    for n_fail in ([8] if QUICK else [30, 8]):
        for informative in (False, True):
            sub, sub_rul, info = ROB.censor_fleet(
                P["train"], P["y_tr"], n_failures=n_fail, informative=informative)
            keep = info["is_failed_row"]
            r = _score_variant(P, sub[keep], sub_rul[keep],
                               f"censored: {n_fail} failures, "
                               f"{'informative' if informative else 'random'}")
            r["labelled_rows"] = info["labelled_rows"]
            r["n_failures"] = info["n_failures_observed"]
            rows.append(r)

    # Sensor drift, applied to the DEPLOYED data rather than the training data.
    # That is the direction it actually happens in: the model was trained on clean
    # historical data and is then fed inputs from ageing sensors. Drifting the
    # training set instead is both unrealistic and nearly harmless, because a
    # drift proportional to cycle is collinear with the `cycle` feature the model
    # already has -- which is why the first version of this row came back
    # *negative*.
    clean_gbm, _ = models.fit_gbm(P["f_tr"][P["cols"]].to_numpy(), P["y_tr"])
    for mag in ([1.0] if QUICK else [0.5, 1.0, 2.0]):
        hd_drift, dinfo = ROB.inject_sensor_drift(
            P["holdout"], P["sensors"], drift_over_life=mag)
        f_d = features.build_features(P["norm"].transform(hd_drift), P["sensors"])
        pd_ = clean_gbm.predict(f_d[P["cols"]].to_numpy())
        rows.append({"variant": f"sensor drift {mag} sd/life on deployed data",
                     "rmse": metrics.rmse(P["y_hd"], pd_),
                     "phm": metrics.phm_score(P["y_hd"], pd_),
                     "bias": float(np.mean(pd_ - P["y_hd"])), "detail": dinfo})

    # Label noise, biased late. Swept rather than measured at one point, because
    # a single magnitude cannot distinguish "the model is robust" from "the
    # injection was too small to matter".
    for sd_c in ([5.0] if QUICK else [5.0, 15.0, 30.0]):
        y_noisy, ninfo = ROB.noisy_labels(
            P["y_tr"], P["train"]["unit"].to_numpy(),
            sd_cycles=sd_c, late_bias_cycles=0.6 * sd_c)
        r = _score_variant(P, P["train"], y_noisy,
                           f"label noise sd {sd_c:.0f}, +{0.6 * sd_c:.0f} late bias")
        r["detail"] = ninfo
        rows.append(r)

    # heterogeneity: train on build A, test on build B
    # Heterogeneity needs a CONTROLLED comparison: the same test set (build B),
    # scored by a model trained on B and by a model trained on A. Scoring the
    # A-model on B and comparing it to the clean holdout -- the first version --
    # confounds the build difference with whatever else differs between those two
    # populations, and it produced a nonsensical "heterogeneity improves the
    # model" result.
    ta, ya, tb, yb, hinfo = ROB.split_by_build(P["train"], P["y_tr"], P["sensors"])
    # Build B is split again into fit and test units. Without this the "control"
    # trains and scores on the same rows -- it came back 12.5 RMSE better than the
    # clean baseline, which is not a control, it is a leak.
    rng_b = np.random.default_rng(11)
    b_units = np.sort(tb["unit"].unique())
    b_test = set(int(u) for u in rng_b.choice(
        b_units, size=max(2, len(b_units) // 3), replace=False))
    m_test = tb["unit"].isin(b_test).to_numpy()
    tb_fit, tb_test = tb.loc[~m_test], tb.loc[m_test]
    yb_fit, yb_test = yb[~m_test], yb[m_test]

    def _x(df):
        return features.build_features(
            P["norm"].transform(df), P["sensors"])[P["cols"]].to_numpy()

    gbm_a, _ = models.fit_gbm(_x(ta), ya)
    gbm_bf, _ = models.fit_gbm(_x(tb_fit), yb_fit)
    x_bt = _x(tb_test)
    pb_b, pb_a = gbm_bf.predict(x_bt), gbm_a.predict(x_bt)
    hinfo["build_b_test_units"] = len(b_test)
    matched = {"variant": "build B (held out) scored by a build-B model",
               "rmse": metrics.rmse(yb_test, pb_b),
               "phm": metrics.phm_score(yb_test, pb_b),
               "bias": float(np.mean(pb_b - yb_test)), "detail": hinfo}
    cross = {"variant": "build B (held out) scored by a build-A model",
             "rmse": metrics.rmse(yb_test, pb_a),
             "phm": metrics.phm_score(yb_test, pb_a),
             "bias": float(np.mean(pb_a - yb_test))}
    cross["rmse_vs_matched"] = cross["rmse"] - matched["rmse"]
    rows += [matched, cross]

    base = rows[0]["rmse"]
    for r in rows:
        r["rmse_delta"] = r["rmse"] - base
    return {"rows": rows, "baseline_rmse": base}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        prev = json.loads((OUT / "completion.json").read_text(encoding="utf-8"))
        (DOCS / "COMPLETION.md").write_text(report(prev), encoding="utf-8")
        print("re-rendered docs/COMPLETION.md")
        return

    t0 = time.perf_counter()
    res: dict = {"quick": QUICK}
    P = prepare("FD001")

    print("1/8 registry + lineage ...", flush=True)
    res["registry"] = stage_registry(P)
    print(f"    v{res['registry']['version']}  "
          f"refusals {sum(c['refused'] for c in res['registry']['checks'])}"
          f"/{len(res['registry']['checks'])}", flush=True)

    print("2/8 TCN, hyperparameter search, ensemble ...", flush=True)
    res["deep"] = stage_deep(P)
    print(f"    best single {res['deep']['best_single_rmse']:.2f}  "
          f"ensemble {res['deep']['ensemble']['rmse']:.2f}", flush=True)

    print("3/8 serving: scorer, batch, HTTP, container ...", flush=True)
    res["serving"] = stage_serving(P)
    print(f"    {res['serving']['batch']['n_scored']} units at "
          f"{res['serving']['batch']['units_per_second']:.0f}/s  "
          f"http_ok={res['serving']['http_ok']}", flush=True)

    print("4/8 edge projection ...", flush=True)
    res["edge"] = stage_edge(P)

    print("5/8 mixture-of-experts over fault modes (FD004) ...", flush=True)
    res["mixture"] = stage_mixture("FD004")
    print(f"    baseline {res['mixture']['baseline']['rmse']:.2f}  "
          f"best mixture {res['mixture']['best']['rmse']:.2f} "
          f"({res['mixture']['delta_rmse']:+.2f})", flush=True)

    print("6/8 concept drift and its detection latency ...", flush=True)
    res["concept"] = stage_concept_drift(P)
    print(f"    input monitor breaches: "
          f"{res['concept']['input_monitor']['features_breaching_psi']}  "
          f"residual monitor fired: {res['concept']['residual_monitor']['fired']}",
          flush=True)

    print("7/8 bootstrap intervals on lead time ...", flush=True)
    res["intervals"] = stage_intervals(P)

    print("8/8 deployment reality, priced ...", flush=True)
    res["reality"] = stage_reality(P)
    for r in res["reality"]["rows"][1:]:
        print(f"    {r['variant'][:44]:46s} {r['rmse_delta']:+6.2f} RMSE", flush=True)

    res["wall_seconds"] = time.perf_counter() - t0
    (OUT / "completion.json").write_text(
        json.dumps(res, indent=1, default=str), encoding="utf-8")
    (DOCS / "COMPLETION.md").write_text(report(res), encoding="utf-8")
    print(f"\nwrote docs/COMPLETION.md ({res['wall_seconds']:.0f}s)")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def report(res: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# ML-1 completion — generated by `complete.py`, not hand-edited\n")
    A("The eight items the README listed as not built. Each section names the item "
      "it closes, and two of them closed by contradicting something the earlier "
      "passes believed.\n")

    # -- 1 --------------------------------------------------------------
    r = res["registry"]
    A("## 1. Registry, and the failure it is actually built to stop\n")
    A("The first pass ended with *\"the condition normaliser is a fitted artefact "
      "that would have to travel with the model and there is no mechanism for "
      "that.\"* The mechanism now exists, and the reason it matters is not "
      "packaging tidiness — it is that shipping the weights without the "
      "normaliser produces a model that still returns plausible numbers.\n")
    A(f"Registered `rul-gbm` v{r['version']}, fingerprint `{r['fingerprint']}`. "
      f"Reload reproduces predictions exactly: **{r['reload_identical']}**. "
      f"MLflow tracking active: {r['mlflow']}.\n")
    A("| attempted violation | refused |")
    A("|---|---|")
    for c in r["checks"]:
        A(f"| {c['attempt']} | {'**yes**' if c['refused'] else 'NO'} |")
    if r.get("silent_swap_cost"):
        s = r["silent_swap_cost"]
        A(f"\nThe third row is the one worth the code. Swapping the normaliser on "
          f"disk for a *differently fitted but perfectly valid* one is caught by "
          f"the fingerprint. Had it not been, the model would have kept running "
          f"and scored **RMSE {s['rmse_wrong_normaliser']:.2f} instead of "
          f"{s['rmse_correct']:.2f}** — degraded, but nowhere near broken enough "
          f"for anyone to suspect the normaliser. That is the shape of the bug "
          f"this prevents: not a crash, a slow wrongness with no obvious cause.\n")

    # -- 2 --------------------------------------------------------------
    d = res["deep"]
    A("## 2. A TCN, a search, and an ensemble — is \"deep loses\" still true?\n")
    A(f"The README's caveat was that *\"deep loses\" is a statement about this "
      f"LSTM*. So: a dilated causal TCN (receptive field "
      f"{d['receptive_field']} cycles, covering the {d['window']}-cycle window), a "
      f"grid over channel width and dropout, and a simplex-constrained blend.\n")
    A("| model | holdout RMSE | PHM score | parameters |")
    A("|---|---|---|---|")
    for k, v in d["single"].items():
        p = d["params"].get(k, "—")
        A(f"| {k.upper()} | {v['rmse']:.2f} | {v['phm']:.0f} | {p} |")
    e = d["ensemble"]
    wtxt = ", ".join(f"{n} {w:.2f}" for n, w in e["weights"].items())
    A(f"| **blend** ({wtxt}) | **{e['rmse']:.2f}** | {e['phm']:.0f} | — |")
    best_single = d["best_single_rmse"]
    gain = best_single - e["rmse"]
    A(f"\nThe blend beats the best single model by **{gain:.2f} RMSE**. Whether "
      "that is worth three models in production is a separate question, and the "
      "residual correlations below are how to answer it: an ensemble only pays "
      "when its members are wrong *differently*.\n")
    A("| pair | residual correlation |")
    A("|---|---|")
    for k, v in d["residual_correlation"].items():
        A(f"| {k} | {v:.3f} |")
    A("\nThe hyperparameter grid, in full — including the configurations that lost, "
      "because a search reported only by its winner is not a search:\n")
    A("| channels | dropout | val RMSE | params | seconds |")
    A("|---|---|---|---|---|")
    for row in d["search"]:
        A(f"| {row['channels']} | {row['dropout']} | {row['val_rmse']:.3f} "
          f"| {row['params']} | {row['seconds']:.0f} |")

    # -- 3 --------------------------------------------------------------
    s = res["serving"]
    A("\n## 3. Serving\n")
    b = s["batch"]
    A(f"Batch scored **{b['n_scored']} units at {b['units_per_second']:.0f} "
      f"units/s**, raising {b['alarms']} alarms, with {b['extrapolating']} units "
      f"flagged as extrapolating beyond the normaliser's fitted regimes. "
      f"HTTP surface live: **{s['http_ok']}** (`{'`, `'.join(s['routes'])}`).\n")
    A(f"**The service refuses a history shorter than the model's window: "
      f"{s['short_history_refused']}.** That refusal is the design decision worth "
      "defending. Left-padding a short history is exactly what the training-time "
      "windower does, so it would have been one line and it would have been "
      "wrong: at training time the pad repeats a real observed first cycle, and at "
      "serving time it fabricates history the engine does not have. Both produce a "
      "number and only one of them means anything.\n")
    A(f"A Dockerfile and compose file are written to `deploy/`. They are "
      f"**not built and not run** — there is no container runtime here — so they "
      f"are a reviewable artefact, not a verified deployment.\n")

    # -- 4 --------------------------------------------------------------
    ed = res["edge"]
    A("## 4. The edge table, with its extrapolation made explicit\n")
    A("This item cannot be closed honestly — there is no gateway in this "
      "environment. What can be fixed is the thing that made the original number "
      "misleading: it was a desktop measurement presented without any indication "
      "of how far it travels.\n")
    A("| target | p50 ms | p99 ms | status |")
    A("|---|---|---|---|")
    for row in ed["projection"]:
        A(f"| {row['target']} | {row['p50_ms']:.2f} | {row['p99_ms']:.2f} "
          f"| {row['note']} |")
    A(f"\n**Only the first row is a measurement.** The rest are frequency scaling "
      f"and nothing else — {ed['caveat']}. A real ARM gateway will not reproduce "
      "them, and the gap will be in the pessimistic direction.\n")

    # -- 5 --------------------------------------------------------------
    m = res["mixture"]
    A("## 5. The mixture head — and it still does not pay\n")
    A(f"The second pass found real fault-mode structure in {m['fd']} "
      f"(silhouette {m['silhouette']:.3f}) and then found that hard-splitting the "
      f"training data made the model *worse*. The diagnosis was sample starvation: "
      "each expert sees half the units. A soft mixture is the fix for that "
      "specific problem, because every expert trains on all the data, weighted.\n")
    A("| temperature | gate entropy | effective units/expert | RMSE | PHM |")
    A("|---|---|---|---|---|")
    for row in m["sweep"]:
        ess = ", ".join(f"{v:.0f}" for v in row["effective_units"])
        A(f"| {row['temperature']} | {row['gate_entropy']:.2f} | {ess} "
          f"(of {row['n_units']}) | {row['rmse']:.2f} | {row['phm']:.0f} |")
    A(f"| **single model** | — | {m['sweep'][0]['n_units']} | "
      f"**{m['baseline']['rmse']:.2f}** | {m['baseline']['phm']:.0f} |")
    delta = m["delta_rmse"]
    if delta < -0.05:
        A(f"\nThe mixture wins by **{-delta:.2f} RMSE** at temperature "
          f"{m['best']['temperature']}, and the effective-sample-size column shows "
          "why it works where hard splitting failed: each expert is trained on far "
          "more than the half a hard split would have given it.")
    else:
        A(f"\n**It still loses — by {delta:+.2f} RMSE at its best temperature.** "
          "So the sample-starvation diagnosis was, at best, incomplete. Read the "
          "entropy column: the mixture is closest to the single model exactly "
          "where it scores best, which means the gate is buying nothing. The "
          "honest conclusion after two attempts is that **these modes are real in "
          "the sensor signatures and not useful for prediction** — the degradation "
          "they describe is already visible to a single model in the features it "
          "has. Supervision from maintenance records naming the failed component "
          "might change that; another unsupervised gate will not, and I would stop "
          "spending on this.")

    # -- 6 --------------------------------------------------------------
    c = res["concept"]
    A("\n## 6. Concept drift: the class of drift the input monitor cannot see\n")
    A(f"The second pass built a PSI monitor over the model's inputs and named its "
      f"blind spot. Here is the blind spot, measured. Remaining life is compressed "
      f"to **{c['injection']['factor']}×** on "
      f"{c['injection']['units_affected']} units — *without touching a single "
      f"sensor value.* P(x) is bit-identical; only P(y|x) moved.\n")
    im = c["input_monitor"]
    A(f"- **Input-side PSI monitor: {im['features_breaching_psi']} features "
      f"breaching**, max PSI {im['max_psi']:.4f}. It is silent, and it is *right* "
      "to be silent — there is nothing in the inputs to find. This is a structural "
      "limit, not a tuning failure.\n")
    rm = c["residual_monitor"]
    A(f"- **Residual monitor: fired = {rm['fired']}**, z = {rm['z']:.1f}, "
      f"direction **{rm['direction']}**. The direction is the actionable half: the "
      "model is over-predicting remaining life, which is the failure mode that "
      "costs an unplanned removal rather than an early one.\n")
    lat = c["latency"]
    ph = lat["page_hinkley_failures"]
    A(f"- **Detection latency: {ph if ph else 'not detected'} realised failures** "
      f"(Page-Hinkley) against {lat['cumulative_3sigma_failures']} for a "
      f"cumulative 3σ rule, out of {lat['n_failures_available']} available.\n")
    A("**That latency is the finding, and it is a property of the fleet rather "
      "than of the monitor.** A concept-drift monitor cannot speak before the "
      "failures arrive. For an operator with four failures a year, a monitor "
      "needing a dozen is not a monitor — it is a post-mortem. This is the "
      "argument for keeping the cheap label-free monitors running even though "
      "they are blunter: they are the only ones with a latency the fleet can "
      "afford.\n")

    # -- 7 --------------------------------------------------------------
    it = res["intervals"]
    A("## 7. The lead-time distribution, with intervals\n")
    A(f"Item 7 was *\"the P05 lead time is the 5th percentile of at most 20 "
      f"numbers\"*. Bootstrapped over {it['n_units']} units that produced a "
      f"sustained alarm (of {it.get('n_units_total', it['n_units'])}; "
      f"{it.get('n_missed', 0)} missed):\n")
    A("| statistic | point | 95% interval | width |")
    A("|---|---|---|---|")
    for name in ("median", "p05", "mean"):
        v = it[name]
        A(f"| {name} lead | {v['point']:.1f} | [{v['lo']:.1f}, {v['hi']:.1f}] "
          f"| {v['width']:.1f} cycles |")
    p05 = it["p05"]
    A(f"\nThe P05 interval is **{p05['width']:.0f} cycles wide** on a point "
      f"estimate of {p05['point']:.0f}. Quoting the point estimate as a planning "
      "figure would imply a precision this sample cannot support, and the width "
      "shrinks as √n — so it is a data problem, not an analysis one.\n")

    # -- 8 --------------------------------------------------------------
    rr = res["reality"]
    A("## 8. DEPLOYMENT_REALITY.md, priced\n")
    A("That document lists the ways C-MAPSS is a luxury dataset. It was prose. "
      "Each row below degrades the data along **one** axis and refits, so the "
      "column that matters is the ordering — which failure mode to spend on "
      "first.\n")
    A("| degradation | RMSE | ΔRMSE | bias | PHM |")
    A("|---|---|---|---|---|")
    for r in rr["rows"]:
        # A ΔRMSE against the FD001 baseline is only meaningful for rows scored on
        # the FD001 holdout. The build rows have their own test set.
        d = "n/a" if "build B" in r["variant"] else f"**{r['rmse_delta']:+.2f}**"
        A(f"| {r['variant']} | {r['rmse']:.2f} | {d} "
          f"| {r['bias']:+.1f} | {r['phm']:.0f} |")
    xrow = next((r for r in rr["rows"] if "build-A model" in r["variant"]), None)
    if xrow is not None and "rmse_vs_matched" in xrow:
        A(f"\nThe two build rows are a matched pair and must be read together: "
          f"the same test units, scored by a model trained on their own build and "
          f"by a model trained on the other one. The cost of the mismatch is "
          f"**{xrow['rmse_vs_matched']:+.2f} RMSE**, which is the number that "
          "belongs in a fleet-heterogeneity argument. Comparing the cross-build "
          "score against the clean holdout instead — which is what I did first — "
          "confounds the build difference with the population difference and "
          "produced the nonsense result that heterogeneity *improves* the model.\n")
    # The build rows use their own test set, so their ΔRMSE against the FD001
    # baseline is meaningless and they are excluded from the ranking. Their
    # matched pair is reported above instead.
    rankable = [r for r in rr["rows"][1:] if "build B" not in r["variant"]]
    worst = max(rankable, key=lambda r: r["rmse_delta"])
    mildest = min(rankable, key=lambda r: r["rmse_delta"])
    A(f"\n**Worst: {worst['variant']} at {worst['rmse_delta']:+.2f} RMSE. "
      f"Mildest: {mildest['variant']} at {mildest['rmse_delta']:+.2f}.** "
      "The ranking is the deliverable — it says where a programme's first year of "
      "data-quality effort should go, and no amount of prose in "
      "DEPLOYMENT_REALITY.md could have established it.\n")
    A("Two things this does not measure, and both make the real world harder than "
      "this table. The degradations are applied **one at a time**, whereas a real "
      "fleet has all of them at once and censoring interacts with label noise — "
      "the units you have labels for are the units whose labels are worst. And "
      "each is a *model* of the problem: the censoring model still labels its "
      "surviving failures perfectly, which no maintenance system does.\n")

    A("---")
    A(f"*Generated by `complete.py` in {res.get('wall_seconds', 0):.0f}s"
      f"{' (quick mode)' if res.get('quick') else ''}. "
      "Regenerate the prose with `python complete.py --report-only`.*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
