"""Fourth pass: calibrated uncertainty (conformal) and survival analysis.

    python uq_survival.py              # all four sub-datasets (~20 min on CPU)
    python uq_survival.py --quick      # FD001 only, one seed
    python uq_survival.py --report-only

Writes out/uq_survival.json, docs/UNCERTAINTY_SURVIVAL.md and two figures in
docs/img/. Every number in the document is formatted from the JSON.

Part A, conformal: does the 90% lower bound really hold 90% of the time on
engines it never saw, including near end of life, and does alarming on it buy
anything a plain threshold change could not?

Part B, survival: with most engines still running (censored), does using them
through a survival model beat dropping them?
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import alarm  # noqa: E402
import cmapss  # noqa: E402
import conformal as CF  # noqa: E402
import features  # noqa: E402
import metrics  # noqa: E402
import models  # noqa: E402
import robustness as ROB  # noqa: E402
import survival as SV  # noqa: E402

OUT = ROOT / "out"
DOCS = ROOT / "docs"
IMG = DOCS / "img"
CAP = cmapss.DEFAULT_RUL_CAP
QUICK = "--quick" in sys.argv
FDS = ["FD001"] if QUICK else ["FD001", "FD002", "FD003", "FD004"]
N_HOLDOUT = 20
SURV_FAILURES = [5, 10, 20]
SURV_SEEDS = [0] if QUICK else [0, 1, 2]


def _feat(norm, sensors, df):
    return features.build_features(norm.transform(df), sensors)


def _last_idx(df: pd.DataFrame) -> np.ndarray:
    return df.reset_index(drop=True).groupby("unit").tail(1).index.to_numpy()


def _split_units(df, n, rng):
    units = np.sort(df["unit"].unique())
    pick = rng.choice(units, size=n, replace=False)
    m = df["unit"].isin(pick)
    return df[~m].copy(), df[m].copy()


# ---------------------------------------------------------------------------
# Part A: conformal lower bounds
# ---------------------------------------------------------------------------

N_FOLDS = 5


def conformal_fd(fd: str) -> dict:
    """Calibrate on out-of-fold predictions, grouped by ENGINE.

    A single 25% calibration split leaves ~20 engines to calibrate on. Rows of one
    engine are strongly correlated, so the effective sample is closer to 20 than
    to the ~4,000 rows, and the realised coverage then swings by several points
    from one split to the next. (The first version did this and came in at 86-87%
    against a 90% target.) Five engine-grouped folds give every training engine
    an out-of-fold residual, so calibration uses ~80 engines instead of ~20. The
    point and quantile predictions on new engines are the average of the five
    fold models, the cross-conformal / CV+ construction (Barber et al., 2021).
    """
    rng = np.random.default_rng(models.SEED)
    train_all, test, rul_test = cmapss.load(fd)
    rest, hold = _split_units(train_all, min(N_HOLDOUT, train_all["unit"].nunique() // 4), rng)

    norm = cmapss.ConditionNormaliser().fit(rest, cmapss.SENSOR_COLS)
    sensors = cmapss.informative_sensors(rest, norm)
    norm = cmapss.ConditionNormaliser().fit(rest, sensors)
    f_rest, f_hd, f_te = (_feat(norm, sensors, d) for d in (rest, hold, test))
    x_rest = f_rest.to_numpy()
    y_rest = cmapss.piecewise_rul(rest, CAP).to_numpy(float)
    y_hd = cmapss.piecewise_rul(hold, CAP).to_numpy(float)
    li = _last_idx(test)
    y_te = np.minimum(rul_test, CAP).astype(float)

    units_rest = rest["unit"].to_numpy()
    uniq = rng.permutation(np.unique(units_rest))
    fold_of = {int(u): i % N_FOLDS for i, u in enumerate(uniq)}
    fold = np.array([fold_of[int(u)] for u in units_rest])

    def pr(m, x):
        return np.clip(m.predict(x), 0, CAP)

    p_cal = np.zeros(len(y_rest))
    q_cal = np.zeros(len(y_rest))
    p_hd = np.zeros(len(y_hd))
    q_hd = np.zeros(len(y_hd))
    p_te = np.zeros(len(li))
    q_te = np.zeros(len(li))
    x_hd, x_te = f_hd.to_numpy(), f_te.to_numpy()[li]
    for k in range(N_FOLDS):
        tr = fold != k
        pm, _ = models.fit_gbm(x_rest[tr], y_rest[tr])
        qm = CF.fit_quantile_gbm(x_rest[tr], y_rest[tr])
        p_cal[~tr] = pr(pm, x_rest[~tr])
        q_cal[~tr] = pr(qm, x_rest[~tr])
        p_hd += pr(pm, x_hd) / N_FOLDS
        q_hd += pr(qm, x_hd) / N_FOLDS
        p_te += pr(pm, x_te) / N_FOLDS
        q_te += pr(qm, x_te) / N_FOLDS
    y_cal = y_rest

    cover, lowers = {}, {}
    for kind in ("split", "mondrian", "cqr"):
        lb = CF.LowerBound(kind).calibrate(y_cal, p_cal, q_cal)
        lo_hd = lb.predict(p_hd, q_hd)
        lo_te = lb.predict(p_te, q_te)
        lowers[kind] = lo_hd
        cover[kind] = {
            "holdout": CF.coverage_report(y_hd, lo_hd, p_hd),
            "test": CF.coverage_report(y_te, lo_te, p_te),
            "q": {str(k): v for k, v in lb.q_.items()},
        }
        # Uncalibrated quantile model, to show what the conformal step fixes.
    cover["quantile_gbm_raw"] = {
        "holdout": CF.coverage_report(y_hd, q_hd, p_hd),
        "test": CF.coverage_report(y_te, q_te, p_te)}

    units = hold["unit"].to_numpy()
    traj = {"point": [], "split": [], "mondrian": [], "cqr": [], "true": [], "unit": []}
    for u in np.unique(units):
        m = units == u
        traj["point"].append(p_hd[m])
        traj["true"].append(y_hd[m])
        traj["unit"].append(int(u))
        for kind in ("split", "mondrian", "cqr"):
            traj[kind].append(lowers[kind][m])

    print(f"  {fd} conformal: calibration/holdout units "
          f"{rest['unit'].nunique()}/{hold['unit'].nunique()}  "
          + "  ".join(f"{k} cov {cover[k]['holdout']['coverage']:.3f} "
                      f"(RUL<25: {cover[k]['holdout']['true_rul_0_25']:.3f})"
                      for k in ("split", "mondrian", "cqr")), flush=True)
    return {"fd": fd, "coverage": cover, "traj": traj,
            "n_units": {"calibration": int(rest["unit"].nunique()),
                        "holdout": int(hold["unit"].nunique()), "test": int(len(li))}}


def policy_grid(trajs: list[np.ndarray], ks=(1, 3, 5, 8), thresholds=range(5, 121)) -> list[dict]:
    rows = []
    for k in ks:
        for t in thresholds:
            per = [alarm.evaluate_unit(p, t, k) for p in trajs]
            lead = np.array([r["lead_time"] for r in per])  # 0 when never alarmed
            rows.append({"k": k, "threshold": t,
                         "lead_mean": float(lead.mean()),
                         "lead_min": float(lead.min()),
                         "lead_p05": float(np.percentile(lead, 5)),
                         "missed": int(sum(r["missed"] for r in per)),
                         "nuisance_per_unit": float(np.mean([r["nuisance_alarms"] for r in per]))})
    return rows


def matched(grid: list[dict], target: float, tol: float = 1.5) -> dict | None:
    """Best worst-case warning among policies whose AVERAGE warning is ~target.

    Matching on average warning is what makes the comparison fair: any signal can
    buy more worst-case warning by alarming earlier on average, which throws away
    engine life. The question is what each signal gets for the same average.
    """
    cand = [r for r in grid if abs(r["lead_mean"] - target) <= tol]
    if not cand:
        return None
    return max(cand, key=lambda r: (-r["missed"], r["lead_min"], r["lead_p05"],
                                    -r["nuisance_per_unit"]))


def bootstrap_matched(trajs_a, trajs_b, target, n_boot=300, seed=0):
    """Unit bootstrap of the worst-5% warning difference (b - a) at matched mean."""
    rng = np.random.default_rng(seed)
    n = len(trajs_a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ga = policy_grid([trajs_a[i] for i in idx], ks=(5,), thresholds=range(5, 121, 2))
        gb = policy_grid([trajs_b[i] for i in idx], ks=(5,), thresholds=range(5, 121, 2))
        a, b = matched(ga, target, 2.0), matched(gb, target, 2.0)
        if a and b:
            diffs.append(b["lead_p05"] - a["lead_p05"])
    d = np.array(diffs)
    return {"median": float(np.median(d)), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)), "n": int(len(d))}


def alarm_comparison(conf: list[dict]) -> dict:
    pooled = {s: [t for c in conf for t in c["traj"][s]] for s in ("point", "mondrian", "cqr")}
    grids = {s: policy_grid(v) for s, v in pooled.items()}
    targets = [15, 20, 25, 30, 40]
    table = []
    for tgt in targets:
        row = {"target_mean_lead": tgt}
        for s in grids:
            row[s] = matched(grids[s], tgt)
        table.append(row)

    # Default cost model, tuned per signal: the project's usual operating point.
    tuned = {}
    for s, v in pooled.items():
        best, _ = alarm.tune(v, thresholds=range(5, 121), ks=(1, 3, 5, 8))
        tuned[s] = best

    boot = {} if QUICK else {str(t): bootstrap_matched(pooled["point"], pooled["cqr"], t)
                             for t in (20, 30)}
    return {"n_units": len(pooled["point"]), "matched": table, "tuned": tuned,
            "bootstrap_cqr_minus_point_p05": boot,
            "frontier": {s: [{"lead_mean": r["lead_mean"], "lead_p05": r["lead_p05"],
                              "missed": r["missed"]}
                             for r in g if r["k"] == 5] for s, g in grids.items()}}


def zero_miss_frontier(conf: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """The safety question, asked directly: how much average warning (engine life
    thrown away) does each signal need before NO engine is missed?

    "Missed" is the project's definition: less than 10 cycles of warning, or none.
    Every (threshold, k) policy is evaluated once per engine; a bootstrap over
    engines then only re-indexes that table, so 1,000 resamples cost seconds.
    """
    ks, ts = (1, 3, 5, 8), np.arange(5, 121)
    pols = [(t, k) for k in ks for t in ts]
    out = {"n_boot": n_boot}
    tables = {}
    for s in ("point", "mondrian", "cqr"):
        trajs = [np.asarray(t) for c in conf for t in c["traj"][s]]
        lead = np.zeros((len(pols), len(trajs)))
        miss = np.zeros((len(pols), len(trajs)), dtype=bool)
        for i, (t, k) in enumerate(pols):
            for j, p in enumerate(trajs):
                r = alarm.evaluate_unit(p, t, k)
                lead[i, j], miss[i, j] = r["lead_time"], r["missed"]
        tables[s] = (lead, miss)

    def need(lead, miss, idx):
        ok = ~miss[:, idx].any(axis=1)
        return float(lead[:, idx].mean(axis=1)[ok].min()) if ok.any() else np.inf

    n = tables["point"][0].shape[1]
    full = np.arange(n)
    for s, (lead, miss) in tables.items():
        out[s] = need(lead, miss, full)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(need(*tables["point"], idx) - need(*tables["cqr"], idx))
    d = np.array(diffs)
    out["saving_point_minus_cqr"] = {
        "estimate": out["point"] - out["cqr"], "median": float(np.median(d)),
        "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
        "frac_positive": float(np.mean(d > 0))}
    out["n_units"] = int(n)

    # Out of sample. The frontier above picks each policy on the same engines it
    # scores, which flatters both signals. Here the policy is chosen on a random
    # half of the engines (least average warning with zero misses there) and
    # then scored on the other half, 200 times.
    oos = {s: {"missed": [], "lead_mean": []} for s in tables}
    for _ in range(200):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        for s, (lead, miss) in tables.items():
            ok = ~miss[:, a].any(axis=1)
            cand = np.flatnonzero(ok)
            i = cand[np.argmin(lead[cand][:, a].mean(axis=1))]
            oos[s]["missed"].append(int(miss[i, b].sum()))
            oos[s]["lead_mean"].append(float(lead[i, b].mean()))
    out["out_of_sample"] = {
        s: {"mean_missed_of_40": float(np.mean(v["missed"])),
            "frac_splits_with_a_miss": float(np.mean(np.array(v["missed"]) > 0)),
            "mean_warning": float(np.mean(v["lead_mean"]))}
        for s, v in oos.items()}
    return out


# ---------------------------------------------------------------------------
# Part B: survival analysis on a censored fleet
# ---------------------------------------------------------------------------

def survival_fd(fd: str) -> dict:
    train_all, test, rul_test = cmapss.load(fd)
    y_te = np.minimum(rul_test, CAP).astype(float)
    y_all = cmapss.piecewise_rul(train_all, CAP).to_numpy(float)

    def evaluate(pred):
        return {"rmse": metrics.rmse(y_te, pred), "phm": metrics.phm_score(y_te, pred),
                "cindex": SV.concordance(rul_test.astype(float), pred)}

    def prep(df_fit):
        norm = cmapss.ConditionNormaliser().fit(df_fit, cmapss.SENSOR_COLS)
        sensors = cmapss.informative_sensors(df_fit, norm)
        norm = cmapss.ConditionNormaliser().fit(df_fit, sensors)
        return norm, sensors

    li = _last_idx(test)

    # Reference: every engine run to failure (the luxury C-MAPSS normally gives).
    norm, sensors = prep(train_all)
    f_all = _feat(norm, sensors, train_all)
    f_te = _feat(norm, sensors, test)
    gbm, _ = models.fit_gbm(f_all.to_numpy(), y_all)
    ref = {"gbm_full": evaluate(np.clip(gbm.predict(f_te.to_numpy())[li], 0, CAP))}
    scols = SV.survival_columns(list(f_all.columns))
    dur, ev = SV.landmark_targets(train_all, set(int(u) for u in train_all["unit"].unique()))
    wb = SV.WeibullRUL().fit(f_all[scols], dur, ev)
    ref["weibull_full"] = evaluate(wb.predict_rul(f_te[scols].iloc[li]))

    rows = []
    for n_fail in SURV_FAILURES:
        for seed in SURV_SEEDS:
            t0 = time.perf_counter()
            sub, sub_y, info = ROB.censor_fleet(train_all, y_all, n_failures=n_fail,
                                                informative=False, seed=seed)
            failed = set(info["failed_units"])
            is_f = info["is_failed_row"]
            # Everything is fitted on what this fleet has actually observed.
            norm, sensors = prep(sub)
            f_sub = _feat(norm, sensors, sub)
            f_te = _feat(norm, sensors, test)
            x_te = f_te.to_numpy()
            scols = SV.survival_columns(list(f_sub.columns))
            dur, ev = SV.landmark_targets(sub, failed)
            r = {"n_failures": n_fail, "seed": seed,
                 "n_censored": info["n_censored"]}

            # 1. drop the censored engines (what stage 8 of complete.py did)
            g1, _ = models.fit_gbm(f_sub.to_numpy()[is_f], sub_y[is_f])
            r["gbm_failed_only"] = evaluate(np.clip(g1.predict(x_te)[li], 0, CAP))

            # 2. the trap: pretend censored engines failed when observation stopped
            naive_y = np.minimum(dur - 1.0, CAP)
            g2, _ = models.fit_gbm(f_sub.to_numpy(), naive_y)
            r["gbm_naive"] = evaluate(np.clip(g2.predict(x_te)[li], 0, CAP))

            # 3. survival model on failed engines only (the control)
            w_f = SV.WeibullRUL().fit(f_sub[scols][is_f], dur[is_f], ev[is_f])
            r["weibull_failed_only"] = evaluate(w_f.predict_rul(f_te[scols].iloc[li]))

            # 4. survival model using the censored engines
            w_a = SV.WeibullRUL().fit(f_sub[scols], dur, ev, seed=seed)
            r["weibull_censored"] = evaluate(w_a.predict_rul(f_te[scols].iloc[li]))

            # 5. GBM trained on real labels + survival-imputed labels for censored
            y_imp = sub_y.copy()
            cm = ~is_f
            y_imp[cm] = w_a.imputed_capped_rul(f_sub[scols][cm], dur[cm])
            g5, _ = models.fit_gbm(f_sub.to_numpy(), y_imp)
            r["gbm_imputed"] = evaluate(np.clip(g5.predict(x_te)[li], 0, CAP))

            # 6. same, but the imputing Weibull saw failed engines only. The
            # conditioning on "survived at least this long" still uses the
            # censored engines; only the fitted life curve differs. Separates
            # "censoring-aware labels help" from "this particular fit helps".
            y_imp2 = sub_y.copy()
            y_imp2[cm] = w_f.imputed_capped_rul(f_sub[scols][cm], dur[cm])
            g6, _ = models.fit_gbm(f_sub.to_numpy(), y_imp2)
            r["gbm_imputed_failfit"] = evaluate(np.clip(g6.predict(x_te)[li], 0, CAP))
            r["seconds"] = time.perf_counter() - t0
            rows.append(r)
            print(f"  {fd} n_fail={n_fail:>2} seed={seed}  RMSE  drop {r['gbm_failed_only']['rmse']:.1f}"
                  f"  naive {r['gbm_naive']['rmse']:.1f}"
                  f"  wb-fail {r['weibull_failed_only']['rmse']:.1f}"
                  f"  wb-cens {r['weibull_censored']['rmse']:.1f}"
                  f"  imputed {r['gbm_imputed']['rmse']:.1f}"
                  f"  imputed-ff {r['gbm_imputed_failfit']['rmse']:.1f}   ({r['seconds']:.0f}s)", flush=True)
    return {"fd": fd, "reference": ref, "rows": rows,
            "n_train_units": int(train_all["unit"].nunique())}


METHODS = ["gbm_failed_only", "gbm_naive", "weibull_failed_only", "weibull_censored",
           "gbm_imputed", "gbm_imputed_failfit"]


def summarise_survival(surv: list[dict]) -> list[dict]:
    out = []
    for s in surv:
        for n in SURV_FAILURES:
            rs = [r for r in s["rows"] if r["n_failures"] == n]
            row = {"fd": s["fd"], "n_failures": n, "n_censored": rs[0]["n_censored"]}
            for m in METHODS:
                v = np.array([r[m]["rmse"] for r in rs])
                c = np.array([r[m]["cindex"] for r in rs])
                row[m] = {"rmse_mean": float(v.mean()), "rmse_sd": float(v.std()),
                          "cindex_mean": float(c.mean())}
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# figures and report
# ---------------------------------------------------------------------------

def figures(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    IMG.mkdir(parents=True, exist_ok=True)
    c = res["conformal"][0]
    tr = c["traj"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, i in zip(axes, range(3)):
        true, point, lo = tr["true"][i], tr["point"][i], tr["cqr"][i]
        x = np.arange(1, len(true) + 1)
        ax.fill_between(x, lo, point, color="#1f77b4", alpha=0.18, label="90% safe range")
        ax.plot(x, true, color="#999999", lw=2, label="true RUL")
        ax.plot(x, point, color="#1f77b4", lw=1.1, label="prediction")
        ax.plot(x, lo, color="#d62728", lw=1.1, label="90% lower bound")
        ax.set_title(f"Engine {tr['unit'][i]} ({c['fd']})", loc="left", fontsize=11)
        ax.set_xlabel("engine cycle")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("remaining life (cycles)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    cov = c["coverage"]["cqr"]["holdout"]["coverage"]
    fig.suptitle(f"Conformal lower bound: the red line is below the truth "
                 f"{cov * 100:.0f}% of the time on unseen engines (target 90%)", fontsize=12)
    fig.tight_layout()
    fig.savefig(IMG / "conformal_demo.png", dpi=130)
    plt.close(fig)

    # The safety trade-off: for each amount of average warning, the fewest engines
    # each signal can miss. Lower-left is better.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    style = {"point": ("#7f7f7f", "alarm on the prediction"),
             "cqr": ("#d62728", "alarm on the 90% lower bound (CQR)")}
    for sname, (col, lab) in style.items():
        trajs = [np.asarray(t) for c in res["conformal"] for t in c["traj"][sname]]
        grid = policy_grid(trajs)
        xs = np.arange(10, 46)
        ys = []
        for x in xs:
            cand = [r["missed"] for r in grid if r["lead_mean"] <= x]
            ys.append(min(cand) if cand else np.nan)
        ax.step(xs, ys, where="post", color=col, lw=2, label=lab)
    z = res["alarm"].get("zero_miss")
    if z:
        for sname, (col, _) in style.items():
            ax.axvline(z[sname], color=col, ls=":", lw=1)
        ax.annotate("", xy=(z["cqr"], 3.2), xytext=(z["point"], 3.2),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1))
        ax.text((z["cqr"] + z["point"]) / 2, 3.45,
                f"{z['point'] - z['cqr']:.0f} fewer cycles thrown away" + "\n" + "for zero missed engines",
                ha="center", fontsize=9)
    ax.set_xlabel("average warning per engine (cycles of engine life given up)")
    ax.set_ylabel(f"engines missed (of {res['alarm']['n_units']})")
    ax.set_ylim(-0.3, 4.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Same safety, less waste: alarming on the lower bound", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(IMG / "safety_tradeoff.png", dpi=130)
    plt.close(fig)

    if len(res["survival_summary"]) == 0:
        return
    summ = res["survival_summary"]
    fds = sorted({r["fd"] for r in summ})
    fig, axes = plt.subplots(1, len(fds), figsize=(3.4 * len(fds), 3.8), sharey=False, squeeze=False)
    colors = {"gbm_failed_only": "#7f7f7f", "gbm_naive": "#ff7f0e",
              "weibull_censored": "#2ca02c", "gbm_imputed_failfit": "#1f77b4"}
    names = {"gbm_failed_only": "GBM, drop running engines",
             "gbm_naive": "GBM, treat running as failed",
             "weibull_censored": "Weibull survival",
             "gbm_imputed_failfit": "GBM + survival-imputed labels"}
    for ax, fd in zip(axes[0], fds):
        rs = [r for r in summ if r["fd"] == fd]
        xs = [r["n_failures"] for r in rs]
        for m, col in colors.items():
            ax.errorbar(xs, [r[m]["rmse_mean"] for r in rs], yerr=[r[m]["rmse_sd"] for r in rs],
                        color=col, marker="o", ms=4, lw=1.3, capsize=2, label=names[m])
        full = next(s for s in res["survival"] if s["fd"] == fd)["reference"]["gbm_full"]["rmse"]
        ax.axhline(full, color="black", ls=":", lw=1, label="GBM, every engine failed (luxury)")
        ax.set_title(fd, loc="left")
        ax.set_xlabel("engines observed to failure")
        ax.set_xticks(xs)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("test RMSE (lower is better)")
    axes[0][-1].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle("When most engines have not failed yet: does using them help?", fontsize=12)
    fig.tight_layout()
    fig.savefig(IMG / "survival_demo.png", dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    path = OUT / "uq_survival.json"
    if "--report-only" in sys.argv:
        res = json.loads(path.read_text(encoding="utf-8"))
    else:
        t0 = time.perf_counter()
        print("Part A: conformal lower bounds", flush=True)
        conf = [conformal_fd(fd) for fd in FDS]
        print("  alarm comparison ...", flush=True)
        alarm_cmp = alarm_comparison(conf)
        print("Part B: survival on a censored fleet", flush=True)
        surv = [survival_fd(fd) for fd in FDS]
        res = {"quick": QUICK, "conformal": conf, "alarm": alarm_cmp, "survival": surv,
               "survival_summary": summarise_survival(surv),
               "wall_seconds": time.perf_counter() - t0}
        for c in res["conformal"]:
            c["traj"] = {k: [np.asarray(a).tolist() for a in v] if k != "unit" else v
                         for k, v in c["traj"].items()}
        path.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for c in res["conformal"]:
        c["traj"] = {k: [np.asarray(a) for a in v] if k != "unit" else v
                     for k, v in c["traj"].items()}
    if "zero_miss" not in res["alarm"]:
        print("  zero-miss frontier + bootstrap ...", flush=True)
        res["alarm"]["zero_miss"] = zero_miss_frontier(res["conformal"])
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["alarm"]["zero_miss"] = res["alarm"]["zero_miss"]
        path.write_text(json.dumps(saved, indent=1, default=float), encoding="utf-8")
    figures(res)
    import uq_report
    (DOCS / "UNCERTAINTY_SURVIVAL.md").write_text(uq_report.render(res), encoding="utf-8")
    print("wrote docs/UNCERTAINTY_SURVIVAL.md", flush=True)


if __name__ == "__main__":
    main()
