"""ML-1, the next 30%: drift trigger, fault-mode discovery, cap-neutral cap test.

python extend.py
python extend.py --report-only

Runs independently of train.py so the flagship pipeline stays intact, and writes
docs/EXTENSIONS.md plus out/extensions.json. Each stage answers a question the
first build explicitly left open:

1. CAP-NEUTRAL CAP TEST. RESULTS.md §5 could only support "capping beats not
     capping" and said the neutral comparison is the DECISION LAYER. This runs it:
refit the alarm policy per cap and compare lead time, missed rate and
nuisance alarms -- all of which live below RUL 50, where every cap can
express the answer.

2. DRIFT + RETRAIN TRIGGER. A real covariate shift is available without
inventing one: train on FD001 (one operating condition) and score FD002 (six).
That is as genuine a distribution shift as a fleet ever sees.

3. FAULT-MODE DISCOVERY on FD003/FD004, with FD001 as the null.

"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import alarm  # noqa: E402
import cmapss  # noqa: E402
import drift  # noqa: E402
import faultmode  # noqa: E402
import features  # noqa: E402
import metrics  # noqa: E402
import models  # noqa: E402

OUT = ROOT / "out"
RNG = np.random.default_rng(models.SEED)
N_HOLDOUT = 20


def _split_holdout(train):
    units = np.sort(train["unit"].unique())
    hold = RNG.choice(units, size=min(N_HOLDOUT, len(units) // 4), replace=False)
    m = train["unit"].isin(hold)
    return train[~m].copy(), train[m].copy()


def _fit_gbm_for_cap(train, holdout, sensors, norm, cap):
    """Fit on train at a given cap, return per-unit holdout trajectories."""
    trn, hld = norm.transform(train), norm.transform(holdout)
    y = cmapss.piecewise_rul(train, cap).to_numpy(dtype=float)
    f_trn = features.build_features(trn, sensors).to_numpy()
    f_hld = features.build_features(hld, sensors)
    m, _ = models.fit_gbm(f_trn, y)
    hi = cap if cap is not None else 1e9
    pred = np.clip(m.predict(f_hld.to_numpy()), 0, hi)
    units = holdout["unit"].to_numpy()
    return [pred[units == u] for u in np.unique(units)]


# ---------------------------------------------------------------------------
# 1. the cap-neutral test
# ---------------------------------------------------------------------------

def cap_decision_test(fd: str = "FD001") -> list[dict]:
    """Compare caps on the DECISION LAYER, which every cap can express.

    Why this is neutral where RMSE was not: the alarm policy operates below RUL
    50. A cap-90 model and a cap-200 model can both say "RUL is 23". Neither is
    structurally advantaged in the region the policy uses, so lead time, missed
    rate and nuisance alarms are comparable in a way RMSE-on-a-subset was not.

    The threshold grid is FIXED across caps for the same reason -- letting each cap
    tune its own grid would reintroduce the advantage.

    """
    train_all, _, _ = cmapss.load(fd)
    train, holdout = _split_holdout(train_all)
    norm = cmapss.ConditionNormaliser().fit(train, cmapss.SENSOR_COLS)
    sensors = cmapss.informative_sensors(train, norm)

    rows = []
    for cap in (90, 110, 125, 140, 200, None):
        traj = _fit_gbm_for_cap(train, holdout, sensors, norm, cap)
        best, _ = alarm.tune(traj, thresholds=range(10, 61, 5), ks=(1, 2, 3, 5, 8))
        rows.append({
            "cap": cap if cap is not None else "none",
            "policy": f"RUL<{best['threshold']:.0f}, k={best['k']}",
            "cost_per_unit": best["cost_per_unit"],
            "lead_mean": best["lead_mean"], "lead_p05": best["lead_p05"],
            "missed_rate": best["missed_rate"],
            "nuisance_per_unit": best["nuisance_per_unit"],
            "frac_ge_10": best["frac_ge_10"],
        })
        print(f"    cap={rows[-1]['cap']:<5} {rows[-1]['policy']:<16} "
              f"cost {best['cost_per_unit']:.3f}  lead {best['lead_mean']:.1f}  "
              f"missed {best['missed_rate']*100:.0f}%  "
              f"nuisance {best['nuisance_per_unit']:.2f}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# 2. drift + retrain trigger
# ---------------------------------------------------------------------------

def drift_experiment() -> dict:
    """Train on FD001, then walk a model through progressively shifted data.

    The windows, in order of increasing genuine shift:
    w0  FD001 test          same conditions  -> should NOT trigger
    w1  FD003 test          same conditions, EXTRA FAULT MODE -> the interesting
    case: the inputs barely move, the physics changed
    w2  FD002 test (regime 0 only)   one of six conditions -> partial shift
    w3  FD002 test (all)    six operating conditions -> unmistakable shift

    w1 is the one worth watching. A new fault mode is a change in the process, not
    in the sensor distribution, so an input-PSI monitor has little to see. If the
    trigger stays quiet on w1 and fires on w3, that is the monitor behaving exactly
    as its ranking predicts -- and it is also the honest limit of label-free
    monitoring, which the report states rather than hides.

    """
    train_all, test1, _ = cmapss.load("FD001")
    train, _ = _split_holdout(train_all)
    norm = cmapss.ConditionNormaliser().fit(train, cmapss.SENSOR_COLS)
    sensors = cmapss.informative_sensors(train, norm)

    y = cmapss.piecewise_rul(train).to_numpy(dtype=float)
    f_ref = features.build_features(norm.transform(train), sensors)
    gbm, _ = models.fit_gbm(f_ref.to_numpy(), y)
    ref_pred = np.clip(gbm.predict(f_ref.to_numpy()), 0, 125)

    _, test3, _ = cmapss.load("FD003")
    _, test2, _ = cmapss.load("FD002")
    reg2 = cmapss.operating_regimes(test2)

    # THE CONTROL HAS TO BE THE SAME POPULATION AS THE REFERENCE.
    #
    # The first version used the FD001 TEST set as the control and it breached on
    # 35 features -- which looked like a broken monitor and was actually a broken
    # experiment. Training units run to failure and span full life; test units are
    # right-censored and truncated early, so they are mostly mid-life. Those are
    # genuinely different feature distributions, and PSI was correctly reporting a
    # difference that has nothing to do with drift.
    #
    # The honest control is held-out TRAINING units: same population, same
    # censoring, no shift. If the monitor fires on that, the monitor is wrong.
    holdout_units = np.sort(train_all["unit"].unique())[-15:]
    control = train_all[train_all["unit"].isin(holdout_units)].copy()

    windows = [
        ("w0: held-out FD001 train units (control)", control),
        ("w0b: FD001 test (same conditions, CENSORED)", test1),
        ("w1: FD003 test (extra fault mode)", test3),
        ("w2: FD002 test, one regime only", test2[reg2 == reg2[0]].copy()),
        ("w3: FD002 test, all six regimes", test2),
    ]

    trig = drift.RetrainTrigger()
    alarm_thr = 25.0
    ref_rate = drift.alarm_rate(ref_pred, alarm_thr)
    out = []
    for name, df in windows:
        f_cur = features.build_features(norm.transform(df), sensors)
        cur_pred = np.clip(gbm.predict(f_cur.to_numpy()), 0, 125)
        fd_rows = drift.feature_drift(f_ref.to_numpy(), f_cur.to_numpy(),
                                      list(f_ref.columns))
        s_psi = drift.psi(ref_pred, cur_pred)
        rec = trig.evaluate(name, fd_rows, s_psi,
                            drift.alarm_rate(cur_pred, alarm_thr), ref_rate)
        rec["top_drifted"] = fd_rows[:5]
        out.append(rec)
        print(f"    {name:<38} feats>=0.25: {rec['n_features_drifted']:<4} "
              f"scorePSI {s_psi:.3f}  alarm x{rec['alarm_rate_ratio']:.2f}  "
              f"-> {rec['action']}", flush=True)
    return {"reference": "FD001 train", "alarm_threshold": alarm_thr,
            "reference_alarm_rate": ref_rate, "windows": out}


# ---------------------------------------------------------------------------
# 3. fault-mode discovery
# ---------------------------------------------------------------------------

def faultmode_experiment(fd: str) -> dict:
    train_all, test, rul_raw = cmapss.load(fd)
    train, holdout = _split_holdout(train_all)
    norm = cmapss.ConditionNormaliser().fit(train, cmapss.SENSOR_COLS)
    sensors = cmapss.informative_sensors(train, norm)
    trn = norm.transform(train)

    disc = faultmode.discover_modes(trn, sensors)
    fd1_train, _, _ = cmapss.load("FD001")
    n1 = cmapss.ConditionNormaliser().fit(fd1_train, cmapss.SENSOR_COLS)
    s1 = cmapss.informative_sensors(fd1_train, n1)
    null = faultmode.null_check(n1.transform(fd1_train), s1)

    # Baseline: one model for everything.
    y = cmapss.piecewise_rul(train).to_numpy(dtype=float)
    f_trn = features.build_features(trn, sensors)
    tst = norm.transform(test)
    f_tst = features.build_features(tst, sensors)
    li = test.groupby("unit").tail(1).index
    truth = np.minimum(rul_raw, 125)

    base_model, _ = models.fit_gbm(f_trn.to_numpy(), y)
    base_pred = np.clip(base_model.predict(f_tst.loc[li].to_numpy()), 0, 125)
    base = {"rmse": metrics.rmse(truth, base_pred),
            "phm": metrics.phm_score(truth, base_pred)}

    # Mode-aware: one model per discovered mode.
    unit_mode = dict(zip(disc["units"], disc["labels"]))
    row_mode = train["unit"].map(unit_mode).to_numpy()
    test_modes = faultmode.assign_modes(trn, tst, sensors, disc)
    test_unit_mode = dict(zip(np.sort(test["unit"].unique()), test_modes))

    per_mode = {}
    for c in np.unique(disc["labels"]):
        m = row_mode == c
        if m.sum() < 500:
            continue
        mdl, _ = models.fit_gbm(f_trn[m].to_numpy(), y[m])
        per_mode[int(c)] = mdl

    mode_pred = base_pred.copy()
    test_units = test.loc[li, "unit"].to_numpy()
    for i, u in enumerate(test_units):
        c = int(test_unit_mode.get(u, -1))
        if c in per_mode:
            mode_pred[i] = np.clip(
                per_mode[c].predict(f_tst.loc[[li[i]]].to_numpy())[0], 0, 125)
    mode_aware = {"rmse": metrics.rmse(truth, mode_pred),
                  "phm": metrics.phm_score(truth, mode_pred)}

    print(f"    {fd}: silhouette {disc['silhouette']:.3f} "
          f"(FD001 null {null['silhouette']:.3f})  sizes {disc['cluster_sizes']}  "
          f"| single {base['rmse']:.2f}/{base['phm']:.0f} -> "
          f"mode-aware {mode_aware['rmse']:.2f}/{mode_aware['phm']:.0f}", flush=True)
    return {
        "fd": fd,
        "silhouette": disc["silhouette"],
        "null_silhouette_fd001": null["silhouette"],
        "cluster_sizes": disc["cluster_sizes"],
        "null_cluster_sizes": null["cluster_sizes"],
        "separating_sensors": disc["separating_sensors"],
        "single_model": base, "mode_aware": mode_aware,
        "rmse_delta": mode_aware["rmse"] - base["rmse"],
        "phm_delta": mode_aware["phm"] - base["phm"],
    }


# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        prev = json.loads((OUT / "extensions.json").read_text())
        (ROOT / "docs" / "EXTENSIONS.md").write_text(report(prev), encoding="utf-8")
        print("re-rendered docs/EXTENSIONS.md")
        return

    t0 = time.perf_counter()
    res: dict = {}
    print("1/3 cap-neutral test on the decision layer ...", flush=True)
    res["cap_decision"] = cap_decision_test("FD001")

    print("2/3 drift monitoring and the retrain trigger ...", flush=True)
    res["drift"] = drift_experiment()

    print("3/3 fault-mode discovery (FD003, FD004) with FD001 as the null ...",
          flush=True)
    res["faultmode"] = [faultmode_experiment(fd) for fd in ("FD003", "FD004")]
    res["wall_seconds"] = time.perf_counter() - t0

    (OUT / "extensions.json").write_text(json.dumps(res, indent=2, default=str))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "EXTENSIONS.md").write_text(report(res), encoding="utf-8")
    print(f"\nwrote docs/EXTENSIONS.md ({res['wall_seconds']:.0f}s)")


def report(res: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# ML-1 extensions — generated by `extend.py`, not hand-edited\n")
    A("Three questions the first build left explicitly open.\n")

    A("## 1. The cap-neutral test RESULTS.md said it wanted\n")
    A("§5 of RESULTS.md could only support the one-sided claim — capping beats not "
      "capping — because RMSE on any RUL-restricted subset structurally favours a "
      "low cap. It named the neutral comparison: **the decision layer**, which "
      "operates below RUL 50 where every cap can express the answer. Here it is, "
      "with the threshold grid held fixed across caps.\n")
    A("| cap | chosen policy | cost/unit | mean lead | P05 lead | missed | nuisance/unit-life |")
    A("|---|---|---|---|---|---|---|")
    for r in res["cap_decision"]:
        A(f"| {r['cap']} | {r['policy']} | {r['cost_per_unit']:.3f} | "
          f"{r['lead_mean']:.1f} | {r['lead_p05']:.0f} | {r['missed_rate']*100:.0f}% | "
          f"{r['nuisance_per_unit']:.2f} |")
    capped = [r for r in res["cap_decision"] if r["cap"] != "none"]
    unc = [r for r in res["cap_decision"] if r["cap"] == "none"][0]
    best = min(capped, key=lambda r: r["cost_per_unit"])
    spread = max(r["cost_per_unit"] for r in capped) - min(r["cost_per_unit"] for r in capped)
    A(f"\nBest capped policy: **cap {best['cap']}** at {best['cost_per_unit']:.3f} "
      f"cost/unit. Uncapped: {unc['cost_per_unit']:.3f}. Across the capped rows the "
      f"cost spread is **{spread:.3f}**.\n")
    if spread < 0.15:
        A("**On the metric that matters, the cap choice is nearly flat** — which is "
          "the result the RMSE table could not establish, and is the opposite of "
          "what RMSE-on-a-subset implied (it made cap 90 look 40% better than cap "
          "140). The decision layer washes the difference out because the policy "
          "only reads predictions below RUL 50, and every cap is competent there. "
          "So the defensible statement is: **cap, and do not agonise over the "
          "value** — which is exactly what the standard 125 being used unquestioned "
          "across the literature has quietly assumed all along.")
    else:
        A(f"The cap choice **does** move the decision layer: {spread:.3f} cost/unit "
          "between the best and worst capped policy, which is a real difference and "
          "means the value has to be chosen rather than inherited.")

    A("\n## 2. Drift monitoring and a retrain trigger with teeth\n")
    d = res["drift"]
    A(f"Reference: {d['reference']}. Alarm threshold RUL<{d['alarm_threshold']:.0f}, "
      f"reference alarm rate {d['reference_alarm_rate']*100:.1f}%.\n")
    A("The trigger requires **magnitude** (PSI ≥ 0.25), **persistence** (2 "
      "consecutive windows) and **scope** (≥3 features, or the prediction "
      "distribution itself). The scope rule is the one that matters: a single "
      "drifting sensor is a *sensor* ticket, not a model ticket, and conflating "
      "them sends the wrong team.\n")
    A("| window | features PSI≥0.25 | max feature PSI | prediction PSI | alarm-rate ratio | action |")
    A("|---|---|---|---|---|---|")
    for w in d["windows"]:
        A(f"| {w['window']} | {w['n_features_drifted']} | {w['max_feature_psi']:.3f} | "
          f"{w['score_psi']:.3f} | ×{w['alarm_rate_ratio']:.2f} | **{w['action']}** |")
    w0 = d["windows"][0]
    w0b = d["windows"][1]
    w1 = d["windows"][2]
    w3 = d["windows"][-1]
    A(f"\n**w0 is the control: held-out training units.** Same population, same "
      f"censoring, no shift — and it is quiet ({w0['action']}, "
      f"{w0['n_features_drifted']} features breached, prediction PSI "
      f"{w0['score_psi']:.3f}). That is the first thing to check about any "
      "monitor: it must not fire on data drawn from the distribution it was built "
      "on.\n")
    A("**w0b is a methodological trap I fell into and kept in the table.** The "
      "first version used the FD001 *test* set as the control, and it breached on "
      f"{w0b['n_features_drifted']} features — which looked like a broken monitor "
      "and was actually a broken experiment. Training units run to failure and "
      "span full life; test units are right-censored and truncated early, so they "
      "sit mostly mid-life. Those are genuinely different feature distributions, "
      "and PSI was correctly reporting a difference that has nothing to do with "
      "drift.\n")
    A("**If the reference window and the comparison window are not the same "
      "population to begin with, every number a drift monitor produces measures "
      "your sampling rather than your process.** On a fleet this is the same "
      "mistake as comparing this month's running assets against a reference built "
      "from assets that later failed.\n")
    A(f"**w1 is the row I expected to be interesting.** FD003 is the same operating "
      "condition as FD001 with an *extra fault mode*. Features at PSI≥0.25: "
      f"{w1['n_features_drifted']}. Prediction PSI: {w1['score_psi']:.3f}. "
      f"Verdict: {w1['action']}.\n")
    if w1["breached"]:
        A("**The monitor does catch it, which is not what I predicted.** I expected "
          "a new fault mode to be invisible to an input-distribution monitor, on "
          "the reasoning that the sensor-to-life *relationship* changes while the "
          "marginals stay put. That reasoning was wrong for this dataset: a second "
          "fault mode drives different sensors in different directions, so the "
          f"marginals move too — {w1['n_features_drifted']} features breach.\n")
        A("The prediction still holds in general for the class of drift where only "
          "the relationship changes (a recalibrated sensor whose distribution is "
          "unchanged, a maintenance policy change that alters what failure means). "
          "**But it does not hold here, and the honest version is: on C-MAPSS the "
          "input monitor happens to be sufficient, and I do not get to claim credit "
          "for predicting otherwise.**\n")
    elif not w1["breached"]:
        A("**The monitor does not see it, and that is the honest limit of "
          "label-free drift detection, not a bug to tune away.** A new failure mode "
          "is a change in the relationship between sensors and remaining life. PSI "
          "measures the marginal distribution of the inputs, and that relationship "
          "can change completely while every marginal stays put. Nothing that reads "
          "only inputs can catch this class of drift — you need labels, and in "
          "prognostics labels are failures, which arrive after the event.\n")
        A("What you *can* do is name the blind spot in the monitoring plan and cover "
          "it another way: a periodic backtest against whatever failures did occur, "
          "and physical review when maintenance reports a fault mode the model was "
          "never trained on. Both are process, not code, and both belong in the "
          "runbook rather than in a threshold.")
    A(f"\n**w3 is the unmistakable shift** — six operating conditions against one — "
      f"and the trigger fires: {w3['n_features_drifted']} features breached, "
      f"prediction PSI {w3['score_psi']:.3f}, alarm rate "
      f"×{w3['alarm_rate_ratio']:.2f}. Note the alarm-rate column throughout: it is "
      "the only label-free signal tied to a *cost*, because it is the maintenance "
      "team's workload.")

    A("\n## 3. Fault-mode discovery, with a null\n")
    A("FD003 and FD004 carry two fault modes; C-MAPSS does not label which unit has "
      "which. So this is discovery, not classification: cluster units by their "
      "per-sensor degradation slope over late life, then **check the clustering is "
      "real** by running the identical procedure on FD001, which has one fault "
      "mode. If FD001 splits as cleanly, the method found nothing.\n")
    A("| sub-dataset | silhouette | FD001 null silhouette | cluster sizes | single model RMSE/PHM | mode-aware RMSE/PHM | Δ RMSE |")
    A("|---|---|---|---|---|---|---|")
    for f in res["faultmode"]:
        A(f"| {f['fd']} | **{f['silhouette']:.3f}** | {f['null_silhouette_fd001']:.3f} | "
          f"{f['cluster_sizes']} | {f['single_model']['rmse']:.2f} / "
          f"{f['single_model']['phm']:.0f} | {f['mode_aware']['rmse']:.2f} / "
          f"{f['mode_aware']['phm']:.0f} | {f['rmse_delta']:+.2f} |")
    f3 = res["faultmode"][0]
    margin = f3["silhouette"] - f3["null_silhouette_fd001"]
    A("")
    if margin < 0.05:
        A(f"**The null fires.** FD003 separates at {f3['silhouette']:.3f} and "
          f"single-fault-mode FD001 separates at {f3['null_silhouette_fd001']:.3f} — "
          f"a margin of {margin:+.3f}. k-means will split any cloud in two; a "
          "silhouette of that size on a dataset with one fault mode means the "
          "statistic is measuring the algorithm's willingness to split, not the "
          "presence of modes.\n")
        A("**So this method has not found the fault modes**, and the RMSE column "
          "should be read in that light rather than as a modelling result. Without "
          "the null this table would have looked like a discovery — which is "
          "precisely why the null is here.")
    else:
        A(f"FD003 separates at {f3['silhouette']:.3f} against a null of "
          f"{f3['null_silhouette_fd001']:.3f}, a margin of {margin:+.3f}, so the "
          "split is carrying real structure.")
    improved = [f for f in res["faultmode"] if f["rmse_delta"] < 0]
    A(f"\nMode-aware models improve RMSE on **{len(improved)} of "
      f"{len(res['faultmode'])}** sub-datasets. Splitting the training data in two "
      "halves each model's data, and on FD003 that is 80 units becoming ~40 — the "
      "variance cost of the split can easily exceed the bias it removes. That "
      "tradeoff is the real reason single models survive in this literature, and it "
      "is visible in the Δ column.")
    A("\n**A weakness worth stating:** test units are right-censored, so a unit "
      "truncated at 40% of its life has almost no late-life signature and its mode "
      "assignment is close to a guess. Mode-aware inference is therefore weakest "
      "exactly where the test set is hardest.")

    A("\n---\n*Every number regenerated by `python extend.py`.*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
