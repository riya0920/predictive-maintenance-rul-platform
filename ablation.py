"""Why is the tree baseline beating published deep models on FD002/FD004?

The first run produced GBM RMSE 13.17 on FD002 against 22.36 for the DCNN of
Li et al. (2018) and 24.49 for the LSTM of Zheng et al. (2017). A 40% improvement
over published work from a HistGradientBoostingRegressor with rolling means is not
a plausible result, and the correct response to an implausible result is to attack
it rather than to publish it.

Two candidate explanations, both testable:

  (A) INFORMATION SET. The published sequence models consume a fixed window of
      sensor readings and nothing else. My feature table also contains `cycle` --
      the absolute age of the engine. That is not leakage (you always know how
      many hours are on an asset), but it is a different problem. C-MAPSS engines
      have a life distribution, so `cycle` alone is a decent RUL predictor by
      regression-to-the-mean, and any comparison against window-only models is
      then apples-to-oranges.

  (B) CONDITION NORMALISATION. FD002/FD004 fly six operating conditions.
      Per-regime z-scoring removes the between-condition variance that otherwise
      swamps degradation. Papers that do not normalise pay for it on exactly these
      two sub-datasets, which is where the published FD001-vs-FD002 gap comes from.

Run: python ablation.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

import cmapss  # noqa: E402
import features  # noqa: E402
import metrics  # noqa: E402
import models  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "out"


def one(fd: str, use_cycle: bool, normalise: bool) -> dict:
    train, test, rul_raw = cmapss.load(fd)
    norm = cmapss.ConditionNormaliser().fit(train, cmapss.SENSOR_COLS)
    sensors = cmapss.informative_sensors(train, norm if normalise else None)
    trn = norm.transform(train) if normalise else train
    tst = norm.transform(test) if normalise else test

    y = cmapss.piecewise_rul(train).to_numpy(dtype=float)
    f_trn = features.build_features(trn, sensors)
    f_tst = features.build_features(tst, sensors)
    if not use_cycle:
        f_trn = f_trn.drop(columns=["cycle"])
        f_tst = f_tst.drop(columns=["cycle"])

    m, _ = models.fit_gbm(f_trn.to_numpy(), y)
    li = test.groupby("unit").tail(1).index
    pred = np.clip(m.predict(f_tst.loc[li].to_numpy()), 0, cmapss.DEFAULT_RUL_CAP)
    truth = np.minimum(rul_raw, cmapss.DEFAULT_RUL_CAP)
    return {
        "fd": fd, "use_cycle": use_cycle, "normalise": normalise,
        "n_features": f_trn.shape[1],
        "rmse": metrics.rmse(truth, pred), "phm": metrics.phm_score(truth, pred),
    }


def cycle_only(fd: str) -> dict:
    """The floor: predict RUL from engine age alone. If this is good, the dataset
    leaks its own life distribution and everyone's numbers are partly this."""
    train, test, rul_raw = cmapss.load(fd)
    y = cmapss.piecewise_rul(train).to_numpy(dtype=float)
    x = train[["cycle"]].to_numpy(dtype=float)
    m, _ = models.fit_gbm(x, y)
    li = test.groupby("unit").tail(1).index
    pred = np.clip(m.predict(test.loc[li, ["cycle"]].to_numpy()), 0, cmapss.DEFAULT_RUL_CAP)
    truth = np.minimum(rul_raw, cmapss.DEFAULT_RUL_CAP)
    return {"fd": fd, "variant": "cycle only", "rmse": metrics.rmse(truth, pred),
            "phm": metrics.phm_score(truth, pred)}


def main() -> None:
    rows = []
    for fd in ("FD001", "FD002", "FD003", "FD004"):
        rows.append(cycle_only(fd))
        for use_cycle in (True, False):
            for normalise in (True, False):
                r = one(fd, use_cycle, normalise)
                r["variant"] = (
                    f"{'+cycle' if use_cycle else '-cycle'} "
                    f"{'+condnorm' if normalise else '-condnorm'}"
                )
                rows.append(r)
        for r in [x for x in rows if x["fd"] == fd]:
            print(f"  {fd}  {r['variant']:<22} RMSE {r['rmse']:6.2f}  PHM {r['phm']:8.0f}",
                  flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "ablation.json").write_text(json.dumps(rows, indent=2))
    print("\nwrote out/ablation.json")


if __name__ == "__main__":
    main()
