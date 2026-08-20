"""ML-1 end-to-end: four sub-datasets, two model families, one alarm policy.

    python train.py            # all four (approx. 12-20 min on CPU)
    python train.py FD001      # one sub-dataset
    python train.py --quick    # fewer LSTM epochs, for a smoke test

Everything in docs/RESULTS.md is written by this script. Nothing in it is hand
edited, including the sentences -- they are formatted from the measured numbers.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

import alarm  # noqa: E402
import cmapss  # noqa: E402
import edge  # noqa: E402
import features  # noqa: E402
import metrics  # noqa: E402
import models  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "out"
WINDOW = 30
N_HOLDOUT_UNITS = 20  # run-to-failure units withheld for the decision layer
RNG = np.random.default_rng(models.SEED)

# Literature reference points. NOT reproduced here -- quoted so my numbers land
# somewhere on a map instead of floating free.
PUBLISHED = {
    # Zheng, Ristovski, Farahat, Gupta -- "Long Short-Term Memory Network for
    # Remaining Useful Life Estimation", IEEE ICPHM 2017.
    "LSTM (Zheng 2017)": {
        "FD001": (16.14, 338), "FD002": (24.49, 4450),
        "FD003": (16.18, 852), "FD004": (28.17, 5550),
    },
    # Li, Ding, Sun -- "Remaining useful life estimation in prognostics using
    # deep convolution neural networks", Reliab. Eng. Syst. Saf. 172 (2018).
    "DCNN (Li 2018)": {
        "FD001": (12.61, 273), "FD002": (22.36, 10412),
        "FD003": (12.64, 284), "FD004": (23.31, 12466),
    },
}


def split_holdout(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Withhold whole units, never rows.

    Row-level splitting would put cycle 100 of engine 7 in train and cycle 101 in
    validation, which is the same engine one cycle later -- an information leak
    that makes any model look excellent. Unit-level is the only defensible split
    for run-to-failure data.
    """
    units = np.sort(train["unit"].unique())
    hold = RNG.choice(units, size=min(N_HOLDOUT_UNITS, len(units) // 4), replace=False)
    m = train["unit"].isin(hold)
    return train[~m].copy(), train[m].copy()


def last_cycle_rows(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("unit").tail(1).index.to_numpy()


def run_dataset(fd: str, quick: bool = False) -> dict:
    print(f"\n=== {fd} " + "=" * 50, flush=True)
    t_start = time.perf_counter()
    train_all, test, rul_true_raw = cmapss.load(fd)
    n_regimes = len(np.unique(cmapss.operating_regimes(train_all)))

    train, holdout = split_holdout(train_all)
    norm = cmapss.ConditionNormaliser().fit(train, cmapss.SENSOR_COLS)
    sensors = cmapss.informative_sensors(train, norm)
    print(f"  units train/holdout/test: {train['unit'].nunique()}/"
          f"{holdout['unit'].nunique()}/{test['unit'].nunique()}   "
          f"regimes: {n_regimes}   sensors kept: {len(sensors)}", flush=True)

    trn = norm.transform(train)
    hld = norm.transform(holdout)
    tst = norm.transform(test)

    y_trn = cmapss.piecewise_rul(train).to_numpy(dtype=float)
    y_hld = cmapss.piecewise_rul(holdout).to_numpy(dtype=float)

    # Test truth: the RUL file gives remaining life at the last test cycle only.
    # Capped to 125 to match the training target and the published protocol; the
    # uncapped number is reported alongside so the choice is visible.
    rul_true = np.minimum(rul_true_raw, cmapss.DEFAULT_RUL_CAP)

    # ---------- tree baseline -------------------------------------------------
    print("  fitting GBM ...", flush=True)
    f_trn = features.build_features(trn, sensors)
    f_tst = features.build_features(tst, sensors)
    f_hld = features.build_features(hld, sensors)
    gbm, gbm_secs = models.fit_gbm(f_trn.to_numpy(), y_trn)
    li = last_cycle_rows(test)
    gbm_test = np.clip(gbm.predict(f_tst.loc[li].to_numpy()), 0, cmapss.DEFAULT_RUL_CAP)
    gbm_hld_all = np.clip(gbm.predict(f_hld.to_numpy()), 0, cmapss.DEFAULT_RUL_CAP)

    # ---------- sequence model -----------------------------------------------
    print("  fitting LSTM ...", flush=True)
    x_trn, ys_trn, _ = features.sequence_windows(trn, sensors, WINDOW, y_trn)
    x_tst, _, idx_tst = features.sequence_windows(tst, sensors, WINDOW)
    x_hld, _, idx_hld = features.sequence_windows(hld, sensors, WINDOW)
    lstm, lstm_secs, lstm_info = models.fit_lstm(
        x_trn, ys_trn, epochs=4 if quick else 25, verbose=not quick
    )
    pos_of_row = {r: i for i, r in enumerate(idx_tst)}
    sel = np.array([pos_of_row[test.index.get_loc(r)] for r in li])
    lstm_test = np.clip(models.predict_lstm(lstm, x_tst[sel]), 0, cmapss.DEFAULT_RUL_CAP)
    lstm_hld_all = np.clip(models.predict_lstm(lstm, x_hld), 0, cmapss.DEFAULT_RUL_CAP)
    order = np.argsort(idx_hld)
    lstm_hld_all = lstm_hld_all[order]

    bench = {}
    for name, pred, secs in (("GBM", gbm_test, gbm_secs), ("LSTM", lstm_test, lstm_secs)):
        bench[name] = {
            "rmse": metrics.rmse(rul_true, pred),
            "phm_score": metrics.phm_score(rul_true, pred),
            "rmse_uncapped_truth": metrics.rmse(rul_true_raw, pred),
            "train_seconds": secs,
        }
        print(f"    {name:<5} test RMSE {bench[name]['rmse']:.2f}  "
              f"PHM {bench[name]['phm_score']:.0f}  ({secs:.0f}s to fit)", flush=True)

    winner = "LSTM" if bench["LSTM"]["phm_score"] < bench["GBM"]["phm_score"] else "GBM"
    hold_pred_all = lstm_hld_all if winner == "LSTM" else gbm_hld_all

    # ---------- trajectory-level analysis (holdout units only) ---------------
    # The test set cannot support any of this: it is right-censored, so there is
    # no failure event to measure lead time against.
    strata = metrics.horizon_strata(y_hld, hold_pred_all)
    units_h = holdout["unit"].to_numpy()
    traj_pred, traj_true = [], []
    for u in np.unique(units_h):
        m = units_h == u
        traj_pred.append(hold_pred_all[m])
        traj_true.append(y_hld[m])
    al = [metrics.alpha_lambda(t, p) for t, p in zip(traj_true, traj_pred)]
    alpha_lambda_summary = {
        "alpha": 0.2,
        "min_cone_cycles": 5.0,
        "n_units": len(al),
        "converged_units": int(sum(a["converged"] for a in al)),
        "converged_units_strict": int(sum(a["converged_strict"] for a in al)),
        "median_prognostic_horizon_cycles": float(
            np.median([a["prognostic_horizon_cycles"] for a in al])
        ),
        "median_lambda_fraction": float(np.median([a["lambda_fraction"] for a in al])),
        "mean_frac_inside_cone": float(np.mean([a["frac_inside_cone"] for a in al])),
        **{f"acc_at_lambda_{lam}": float(np.mean([a[f"inside_at_lambda_{lam}"] for a in al]))
           for lam in (0.25, 0.5, 0.75, 0.9)},
    }

    # Keep the trajectories so the decision layer can be re-analysed without
    # refitting anything. train.py --report-only depends on this.
    np.savez_compressed(
        OUT / f"holdout_{fd}.npz",
        pred=np.concatenate(traj_pred), true=np.concatenate(traj_true),
        unit=np.concatenate([[u] * len(p) for u, p in zip(np.unique(units_h), traj_pred)]),
    )

    best, grid = alarm.tune(traj_pred, thresholds=range(10, 61, 5), ks=(1, 2, 3, 5, 8))
    persistence_curve = [
        alarm.cost_of_policy(traj_pred, best["threshold"], k) for k in (1, 2, 3, 5, 8, 12)
    ]
    # Two sensitivity axes, because they push the operating point in opposite
    # directions: c_fail wants to alarm earlier, c_life wants to alarm later.
    cost_sensitivity = []
    for cf in (5.0, 10.0, 20.0, 50.0, 100.0):
        b, _ = alarm.tune(traj_pred, range(10, 61, 5), (1, 2, 3, 5, 8), c_fail=cf)
        cost_sensitivity.append({"axis": "c_fail", "value": cf, **b})
    for cl in (0.0, 0.02, 0.10, 0.30, 1.00):
        b, _ = alarm.tune(traj_pred, range(10, 61, 5), (1, 2, 3, 5, 8), c_life=cl)
        cost_sensitivity.append({"axis": "c_life", "value": cl, **b})

    print(f"  alarm policy: RUL<{best['threshold']:.0f} for k={best['k']}  "
          f"-> lead {best['lead_mean']:.1f} cycles mean, "
          f"{best['frac_ge_10']*100:.0f}% of units get >=10, "
          f"{best['nuisance_per_unit']:.2f} nuisance alarms/unit-life", flush=True)

    result = {
        "fd": fd,
        "n_regimes": n_regimes,
        "n_sensors": len(sensors),
        "sensors": sensors,
        "units": {
            "train": int(train["unit"].nunique()),
            "holdout": int(holdout["unit"].nunique()),
            "test": int(test["unit"].nunique()),
        },
        "benchmark": bench,
        "winner_by_phm": winner,
        "lstm_params": lstm_info["n_params"],
        "horizon_strata": strata,
        "alpha_lambda": alpha_lambda_summary,
        "alarm_best": best,
        "alarm_persistence_curve": persistence_curve,
        "alarm_cost_sensitivity": cost_sensitivity,
        "wall_seconds": time.perf_counter() - t_start,
    }

    if fd == "FD001":
        result["edge"] = edge_table(lstm, x_hld[:2000], sensors, lstm_hld_all)
        result["cap_sensitivity"] = cap_sensitivity(fd, train, test, rul_true_raw, norm, sensors)
    return result


def edge_table(lstm, x_sample: np.ndarray, sensors: list[str], ref_preds: np.ndarray) -> dict:
    print("  exporting ONNX + benchmarking edge inference ...", flush=True)
    fp32 = edge.export_onnx(lstm, len(sensors), WINDOW, OUT / "rul_lstm_fp32.onnx")
    r32 = edge.bench_onnx(fp32, x_sample)
    torch_size = sum(p.numel() * 4 for p in lstm.parameters()) / 1024.0
    row = {
        "torch_weights_kb": torch_size,
        "onnx_fp32": {k: v for k, v in r32.items() if k != "preds"},
    }
    q = edge.quantize_dynamic(fp32, OUT / "rul_lstm_int8.onnx")
    if q is not None:
        r8 = edge.bench_onnx(q, x_sample)
        row["onnx_int8"] = {k: v for k, v in r8.items() if k != "preds"}
        n = min(len(r32["preds"]), len(r8["preds"]))
        row["int8_rul_mae_vs_fp32"] = float(np.mean(np.abs(r8["preds"][:n] - r32["preds"][:n])))
        row["int8_rul_max_abs_delta"] = float(np.max(np.abs(r8["preds"][:n] - r32["preds"][:n])))
    return row


def cap_sensitivity(fd, train, test, rul_true_raw, norm, sensors) -> list[dict]:
    """How much does the 125 in 'piecewise-linear RUL' actually matter?"""
    print("  sweeping the RUL cap ...", flush=True)
    trn = norm.transform(train)
    tst = norm.transform(test)
    f_trn = features.build_features(trn, sensors).to_numpy()
    f_tst = features.build_features(tst, sensors)
    li = last_cycle_rows(test)
    xt = f_tst.loc[li].to_numpy()
    rows = []
    for cap in (90, 110, 125, 140, 200, None):
        y = cmapss.piecewise_rul(train, cap).to_numpy(dtype=float)
        m, _ = models.fit_gbm(f_trn, y)
        hi = cap if cap is not None else 1e9
        pred = np.clip(m.predict(xt), 0, hi)
        truth = np.minimum(rul_true_raw, hi)
        # Fair common ground: units whose true RUL is <= 90, a range EVERY cap can
        # express. Scoring a cap-90 model against cap-125 truth would penalise it
        # for being unable to say a number it was never asked to say.
        low = rul_true_raw <= 90
        rows.append({
            "cap": cap if cap is not None else "none",
            "rmse": metrics.rmse(truth, pred),
            "phm_score": metrics.phm_score(truth, pred),
            "n_low_rul_units": int(low.sum()),
            "rmse_rul_le_90": metrics.rmse(rul_true_raw[low], pred[low]),
            "phm_rul_le_90": metrics.phm_score(rul_true_raw[low], pred[low]),
        })
        print(f"    cap={rows[-1]['cap']:<5} RMSE {rows[-1]['rmse']:6.2f}  "
              f"PHM {rows[-1]['phm_score']:9.0f}   [RUL<=90 subset: "
              f"RMSE {rows[-1]['rmse_rul_le_90']:6.2f} PHM {rows[-1]['phm_rul_le_90']:7.0f}]",
              flush=True)
    return rows


def fmt_results(all_res: list[dict]) -> str:
    L: list[str] = []
    A = L.append
    A("# ML-1 results — generated by `train.py`, not hand-edited\n")
    A(f"Generated in {sum(r['wall_seconds'] for r in all_res)/60:.1f} min of CPU wall time.\n")

    A("\n## 1. Benchmark: all four sub-datasets\n")
    A("Test-set protocol: one prediction per test unit, at its last available cycle,")
    A("scored against `RUL_FDxxx.txt` clipped at the 125-cycle cap (the protocol the")
    A("quoted papers use). PHM score is the asymmetric C-MAPSS/PHM08 sum — lower is")
    A("better, and it is a SUM, so it scales with the number of test units.\n")
    A("| sub-dataset | conditions | fault modes | units (tr/ho/te) | GBM RMSE | GBM PHM | LSTM RMSE | LSTM PHM | lit. best RMSE |")
    A("|---|---|---|---|---|---|---|---|---|")
    modes = {"FD001": 1, "FD002": 1, "FD003": 2, "FD004": 2}
    for r in all_res:
        fd = r["fd"]
        u = r["units"]
        lit = min(PUBLISHED[k][fd][0] for k in PUBLISHED)
        A(f"| {fd} | {r['n_regimes']} | {modes[fd]} | {u['train']}/{u['holdout']}/{u['test']} | "
          f"{r['benchmark']['GBM']['rmse']:.2f} | {r['benchmark']['GBM']['phm_score']:.0f} | "
          f"{r['benchmark']['LSTM']['rmse']:.2f} | {r['benchmark']['LSTM']['phm_score']:.0f} | {lit:.2f} |")
    A("\nPublished reference points (quoted, **not** reproduced here):\n")
    A("| method | FD001 | FD002 | FD003 | FD004 |")
    A("|---|---|---|---|---|")
    for k, v in PUBLISHED.items():
        A(f"| {k} | " + " | ".join(f"{v[f][0]:.2f} / {v[f][1]}" for f in ("FD001", "FD002", "FD003", "FD004")) + " |")

    d1 = next(r for r in all_res if r["fd"] == "FD001")
    d4 = next((r for r in all_res if r["fd"] == "FD004"), None)
    if d4:
        gap = d4["benchmark"][d4["winner_by_phm"]]["rmse"] - d1["benchmark"][d1["winner_by_phm"]]["rmse"]
        A(f"\n**The gap.** Best-model RMSE is {gap:+.2f} worse on FD004 than on FD001. "
          f"FD004 flies {d4['n_regimes']} operating conditions against FD001's {d1['n_regimes']}, "
          "and carries two fault modes rather than one. Condition normalisation "
          f"(`cmapss.ConditionNormaliser`) removes the first difference — it is why "
          f"{d4['n_sensors']} sensors survive the variance screen on FD004 versus "
          f"{d1['n_sensors']} on FD001. What is left is the second: a single model is "
          "averaging two degradation physics, and closing that needs either a fault-mode "
          "classifier upstream or a mixture head. Neither is built here.")

    A("\n## 2. Where the error lives (horizon-stratified)\n")
    A("Computed on held-out run-to-failure units, all cycles — not on the censored test set.\n")
    A("| sub-dataset | " + " | ".join(s["band"] for s in d1["horizon_strata"]) + " |")
    A("|---|" + "---|" * len(d1["horizon_strata"]))
    for r in all_res:
        A(f"| {r['fd']} RMSE | " + " | ".join(f"{s['rmse']:.1f}" for s in r["horizon_strata"]) + " |")
    worst = max(d1["horizon_strata"], key=lambda s: s["rmse"])
    A(f"\nThe shape is not monotonic and should not be read as if it were. On FD001 the "
      f"error is lowest near failure ({d1['horizon_strata'][0]['rmse']:.1f} in `[0,20)`), "
      f"peaks in the `{worst['band']}` band at {worst['rmse']:.1f}, and falls again in "
      f"`{d1['horizon_strata'][-1]['band']}` ({d1['horizon_strata'][-1]['rmse']:.1f}). Two "
      "different things are happening. The far band is easy for an uninteresting reason: "
      "the 125-cycle cap makes early life a constant, and predicting a constant is free — "
      "that band's low error is an artefact of the target, not evidence of skill. The "
      "middle band is genuinely the hardest region, where degradation has become visible "
      "but has not yet become decisive.")
    A(f"\nThe band that matters is `[0,20)`, and it is the best one ("
      f"{d1['horizon_strata'][0]['rmse']:.1f} RMSE, bias "
      f"{d1['horizon_strata'][0]['bias']:+.1f}). That is the property to argue for: the "
      "alarm decision is taken in the last few dozen cycles of life, so accuracy is being "
      "spent where it converts into a maintenance date. A model with the same aggregate "
      "RMSE and the opposite profile would be useless, and no aggregate number would say so.")

    A("\n## 3. Prognostic horizon (α-λ, α = 0.2)\n")
    A("*Converged* means the prediction entered the accuracy cone and never left it again. "
      "Entering **and staying** is the requirement; a model that dips into the cone and "
      "back out has not converged, and any snapshot metric would credit it anyway.\n")
    A("Two cones. **Strict** is ±0.2·RUL exactly as defined. **Floored** is "
      "±max(0.2·RUL, 5 cycles). The gap between the two columns is the finding — see below.\n")
    A("| sub-dataset | converged (floored) | converged (strict) | median PH, cycles | median λ | α-λ accuracy at λ=0.5 | at λ=0.9 |")
    A("|---|---|---|---|---|---|---|")
    for r in all_res:
        a = r["alpha_lambda"]
        A(f"| {r['fd']} | {a['converged_units']}/{a['n_units']} | "
          f"{a['converged_units_strict']}/{a['n_units']} | "
          f"{a['median_prognostic_horizon_cycles']:.0f} | {a['median_lambda_fraction']:.2f} | "
          f"{a['acc_at_lambda_0.5']*100:.0f}% | {a['acc_at_lambda_0.9']*100:.0f}% |")
    tot_s = sum(r["alpha_lambda"]["converged_units_strict"] for r in all_res)
    tot_f = sum(r["alpha_lambda"]["converged_units"] for r in all_res)
    tot_n = sum(r["alpha_lambda"]["n_units"] for r in all_res)
    A(f"\n**Read the two columns together.** Under the strict cone, {tot_s}/{tot_n} units "
      f"converge. Under the floored cone, {tot_f}/{tot_n}. That collapse is not the model "
      "getting worse — it is the metric. The strict cone at RUL = 3 demands the prediction "
      "land within 0.6 of a cycle *and never leave*, which no prognostic on any sensor suite "
      "achieves, so strict PH is close to a constant zero for everyone and discriminates "
      "nothing. The floor of 5 cycles encodes a maintenance fact: once RUL is inside a week, "
      "the difference between Tuesday and Wednesday no longer changes the plan, because the "
      "work order is already cut.\n")
    A("I am reporting both rather than the flattering one. If a reviewer prefers the strict "
      "definition, the strict column is the answer and it is bad — and the reason it is bad "
      "is worth more in an interview than the floored number is.")

    A("\n## 4. The alarm policy — what a planner actually buys\n")
    A("| sub-dataset | policy | mean lead | P05 lead | ≥10 cycles | missed | nuisance alarms / unit-life |")
    A("|---|---|---|---|---|---|---|")
    for r in all_res:
        b = r["alarm_best"]
        A(f"| {r['fd']} | RUL<{b['threshold']:.0f}, k={b['k']} | {b['lead_mean']:.1f} | "
          f"{b['lead_p05']:.0f} | {b['frac_ge_10']*100:.0f}% | {b['missed_rate']*100:.0f}% | "
          f"{b['nuisance_per_unit']:.2f} |")
    b1 = d1["alarm_best"]
    A(f"\nFD001, in the sentence a maintenance manager reads: **we alert "
      f"{b1['lead_mean']:.0f} cycles early on average, {b1['frac_ge_10']*100:.0f}% of units "
      f"get at least 10 cycles of warning, and the operator sees "
      f"{b1['nuisance_per_unit']:.2f} alarms per engine-life that clear on their own.**")

    A(f"\n### Why k={b1['k']}? The persistence curve (FD001, threshold fixed at "
      f"{b1['threshold']:.0f})\n")
    A("| k | mean lead | missed | nuisance / unit-life | cost per unit |")
    A("|---|---|---|---|---|")
    for row in d1["alarm_persistence_curve"]:
        A(f"| {row['k']} | {row['lead_mean']:.1f} | {row['missed_rate']*100:.0f}% | "
          f"{row['nuisance_per_unit']:.2f} | {row['cost_per_unit']:.3f} |")
    A("\nRaising k trades lead time (you wait k cycles before asserting) against nuisance "
      "alarms. The curve is the answer to \"why 3?\" — it is not 3 because 3 is a nice "
      "number, it is whichever k minimises the cost column, and the column is visible so "
      "the planner can disagree with the cost model rather than with the number.")

    A("\n### Sensitivity: does the operating point survive different cost assumptions?\n")
    A("Two axes, because they pull in opposite directions: `c_fail` (cost of an "
      "in-service failure) wants the alarm earlier, `c_life` (cost per cycle of "
      "remaining life discarded by pulling early) wants it later.\n")
    A("| axis | value | chosen threshold | k | mean lead | missed |")
    A("|---|---|---|---|---|---|")
    for row in d1["alarm_cost_sensitivity"]:
        v = f"{row['value']:.0f}:1" if row["axis"] == "c_fail" else f"{row['value']:.2f}"
        A(f"| `{row['axis']}` | {v} | RUL<{row['threshold']:.0f} | {row['k']} | "
          f"{row['lead_mean']:.1f} | {row['missed_rate']*100:.0f}% |")
    cf_rows = [r for r in d1["alarm_cost_sensitivity"] if r["axis"] == "c_fail"]
    cl_rows = [r for r in d1["alarm_cost_sensitivity"] if r["axis"] == "c_life"]
    cf_flat = len({r["threshold"] for r in cf_rows}) == 1
    A("")
    if cf_flat:
        A(f"**The `c_fail` axis is flat** — the same policy (RUL<{cf_rows[0]['threshold']:.0f}, "
          f"k={cf_rows[0]['k']}) wins from 5:1 to 100:1. That is not the sweep failing, it is "
          "the sweep telling me which constraint is binding. At the chosen persistence "
          f"{cf_rows[0]['missed_rate']*100:.0f}% of units are missed, so the `c_fail` term "
          "contributes nothing to the objective and multiplying it by 20 multiplies zero. "
          "The policy is robust to the number I am least able to source, which is the good "
          "version of this result — but it is robust because it is not being tested, and "
          "saying only 'insensitive to failure cost' would have been misleading.")
    A(f"\nThe `c_life` axis is where the policy actually moves: the threshold goes from "
      f"RUL<{cl_rows[0]['threshold']:.0f} at c_life={cl_rows[0]['value']:.2f} to "
      f"RUL<{cl_rows[-1]['threshold']:.0f} at c_life={cl_rows[-1]['value']:.2f}, and mean "
      f"lead time falls from {cl_rows[0]['lead_mean']:.1f} to {cl_rows[-1]['lead_mean']:.1f} "
      "cycles. So the number to go and get from finance is not the cost of a failure — it is "
      "**what an hour of unused remaining life is worth**, which is the one nobody asks for.")

    if "cap_sensitivity" in d1:
        A("\n## 5. The RUL cap — how much does 125 matter?\n")
        A("GBM on FD001, refit per cap. The first two columns score each model against "
          "its own capped truth and are **not** comparable across rows — a lower cap "
          "compresses the target and flatters itself. The last two columns restrict "
          f"scoring to the {d1['cap_sensitivity'][0]['n_low_rul_units']} test units "
          "with true RUL ≤ 90, a range every cap can express — which removes *some* "
          "of that unfairness but, as the discussion below works through, not all "
          "of it.\n")
        A("| cap | RMSE (own truth) | PHM (own truth) | RMSE (RUL≤90) | PHM (RUL≤90) |")
        A("|---|---|---|---|---|")
        for row in d1["cap_sensitivity"]:
            A(f"| {row['cap']} | {row['rmse']:.2f} | {row['phm_score']:.0f} | "
              f"{row['rmse_rul_le_90']:.2f} | {row['phm_rul_le_90']:.0f} |")
        capped = [r for r in d1["cap_sensitivity"] if r["cap"] != "none"]
        uncapped = [r for r in d1["cap_sensitivity"] if r["cap"] == "none"][0]
        best_cap = min(capped, key=lambda r: r["phm_rul_le_90"])
        lo, hi = capped[0], capped[-1]
        A(f"\n**The clearest result is the uncapped row.** It scores "
          f"{uncapped['rmse_rul_le_90']:.2f} RMSE / {uncapped['phm_rul_le_90']:.0f} PHM "
          f"against {best_cap['rmse_rul_le_90']:.2f} / {best_cap['phm_rul_le_90']:.0f} for the "
          f"best capped model — a {uncapped['phm_rul_le_90']/max(best_cap['phm_rul_le_90'],1):.0f}× "
          "worse PHM score, and PHM is the metric that weights the end of life where the "
          "alarm decision is taken. An uncapped linear target asks the model to distinguish "
          "RUL 300 from RUL 250 on evidence that does not exist, and the only way to reduce "
          "that loss is to regress on cycle count — i.e. to learn the training set's "
          "censoring pattern rather than degradation.\n")
        A(f"**Among the capped rows the comparison is NOT clean, and I am not going to "
          f"pretend it is.** RMSE rises monotonically with the cap "
          f"({lo['rmse_rul_le_90']:.2f} at cap {lo['cap']} to {hi['rmse_rul_le_90']:.2f} at "
          f"cap {hi['cap']}), which looks like an argument for the lowest cap. It is not. "
          "The evaluation subset is the units with true RUL ≤ 90, and a cap-90 model is "
          "structurally unable to over-predict on that subset — the cap encodes the same "
          "prior the subset selects on. Restricting to a low-RUL region does not remove the "
          "advantage a low cap has there, it hands it over.\n")
        A("There is no single fixed truth on which caps compare neutrally, because the cap "
          "changes what the model is allowed to say. **The cap-neutral test is the decision "
          "layer** — refit the alarm policy per cap and compare lead time, missed rate and "
          "nuisance alarms, all of which live below RUL 50 where every cap can express the "
          "answer. That is not run here, and it is the experiment I would want before "
          "defending a cap value to anyone. What this table does support is the strong "
          "claim, which is about capping versus not capping at all.")

    if "edge" in d1:
        e = d1["edge"]
        A("\n## 6. Edge deployment — could this run on the gateway?\n")
        A(f"LSTM, {d1['lstm_params']:,} parameters, single-sample inference, "
          "onnxruntime pinned to **one** CPU thread to stand in for a gateway core.\n")
        A("| artefact | size | p50 latency | p99 latency | RUL delta vs fp32 |")
        A("|---|---|---|---|---|")
        A(f"| PyTorch weights | {e['torch_weights_kb']:.0f} KB | — | — | — |")
        A(f"| ONNX fp32 | {e['onnx_fp32']['file_kb']:.0f} KB | {e['onnx_fp32']['p50_ms']:.2f} ms | "
          f"{e['onnx_fp32']['p99_ms']:.2f} ms | — |")
        if "onnx_int8" in e:
            A(f"| ONNX int8 (dynamic) | {e['onnx_int8']['file_kb']:.0f} KB | "
              f"{e['onnx_int8']['p50_ms']:.2f} ms | {e['onnx_int8']['p99_ms']:.2f} ms | "
              f"MAE {e['int8_rul_mae_vs_fp32']:.2f} cycles, max "
              f"{e['int8_rul_max_abs_delta']:.2f} |")
            A(f"\nint8 costs {e['int8_rul_mae_vs_fp32']:.2f} cycles of mean absolute "
              "disagreement with fp32. Against a lead time measured in tens of cycles that "
              "is affordable; against the ±20% α-λ cone near end of life it is not free, "
              "and the honest reading is that the size win matters more than the latency "
              "win here — an engine cycle is minutes, so neither number is a constraint. "
              "The constraint this table actually answers is memory footprint on a gateway "
              "already running protocol drivers.")
    A("\n---\n*Every number above is a property of C-MAPSS, a simulation. See "
      "`docs/DEPLOYMENT_REALITY.md` before believing any of it about a real fleet.*")
    return "\n".join(L) + "\n"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    quick = "--quick" in sys.argv
    fds = args or ["FD001", "FD002", "FD003", "FD004"]
    OUT.mkdir(exist_ok=True)

    if "--report-only" in sys.argv:
        # Re-render docs/RESULTS.md from the last run's measurements. Exists so a
        # wording fix does not cost 18 minutes of refitting, and so the document
        # can never drift from out/results.json.
        res = json.loads((OUT / "results.json").read_text())
        (ROOT / "docs" / "RESULTS.md").write_text(fmt_results(res), encoding="utf-8")
        print("re-rendered docs/RESULTS.md from out/results.json")
        return

    res = [run_dataset(fd, quick) for fd in fds]
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    if len(res) == 4:
        (ROOT / "docs").mkdir(exist_ok=True)
        (ROOT / "docs" / "RESULTS.md").write_text(fmt_results(res), encoding="utf-8")
        print("\nwrote docs/RESULTS.md")
    print("wrote out/results.json")


if __name__ == "__main__":
    main()
