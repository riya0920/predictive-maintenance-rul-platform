# ML-1 — Predictive Maintenance Platform (RUL + deployment reality)

**Status: ~50% slice.** The measurement, the maintenance-decision layer, the
edge-export story, **drift monitoring with a retrain trigger**, and **fault-mode
discovery** are built and run end to end on **real NASA C-MAPSS data, all four
sub-datasets**. MLflow, serving, and a second sequence model are not. See
[what is NOT built](#what-is-not-built-the-other-50) — and read
[docs/DEPLOYMENT_REALITY.md](docs/DEPLOYMENT_REALITY.md) before believing any
number here about a real fleet.

The second build's results are in **[docs/EXTENSIONS.md](docs/EXTENSIONS.md)**
(`python extend.py`): the cap-neutral test this README previously said was
missing, a retrain trigger with magnitude/persistence/scope rules, and fault-mode
discovery with a null.

```bash
python train.py
```

~18 minutes on CPU. Everything in [docs/RESULTS.md](docs/RESULTS.md) is written by
that command, including the narrative sentences — they are formatted from the
measured numbers rather than written ahead of them. `python train.py --report-only`
re-renders the document from `out/results.json` without refitting.

```bash
python ablation.py    # the audit described below
```

## The data is real

`data/CMAPSS/` holds the actual NASA Turbofan Engine Degradation Simulation
dataset — `train_FD001..FD004`, `test_FD001..FD004`, `RUL_FD001..FD004`, 43 MB.
Not a generator, not a subset. Provenance and the format check are in
[data/CMAPSS/SOURCE.md](data/CMAPSS/SOURCE.md).

| | conditions | fault modes | train units | test units |
|---|---|---|---|---|
| FD001 | 1 | 1 | 100 | 100 |
| FD002 | 6 | 1 | 260 | 259 |
| FD003 | 1 | 2 | 100 | 100 |
| FD004 | 6 | 2 | 249 | 248 |

That table is the entire difficulty gradient, which is why results on FD001 alone
say nothing about a method.

**The censoring, stated because it drives the design:** training units run to
failure; test units are truncated at a random point and only their RUL *at that
instant* is given. There is no failure event anywhere in the test set. So lead
time, prognostic horizon, and everything else trajectory-shaped is computed on
**20 held-out training units per sub-dataset**, never on the test set. A portfolio
reporting lead time "on the C-MAPSS test set" has either redefined the metric or
not read the data description.

## Headline results

| sub-dataset | GBM RMSE | GBM PHM score | LSTM RMSE | LSTM PHM | best published RMSE quoted |
|---|---|---|---|---|---|
| FD001 | **12.03** | 216 | 14.34 | 413 | 12.61 |
| FD002 | **13.17** | 794 | 17.15 | 2198 | 22.36 |
| FD003 | **12.77** | 289 | 15.25 | 1198 | 12.64 |
| FD004 | **14.48** | 1052 | 15.26 | 1147 | 23.31 |

Published values quoted from Zheng et al. (ICPHM 2017) and Li et al. (RESS 2018);
neither was reproduced here.

The alarm policy, which is the number a maintenance planner actually buys:

| | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| policy | RUL<25, k=5 | RUL<30, k=8 | RUL<25, k=8 | RUL<45, k=8 |
| mean lead time | 16.9 cycles | 20.4 | 14.4 | 28.9 |
| units with ≥10 cycles warning | 100% | 100% | 100% | 100% |
| nuisance alarms per unit-life | 0.05 | 0.00 | 0.00 | 0.30 |

## I did not believe the FD002/FD004 numbers, so I attacked them

A `HistGradientBoostingRegressor` with rolling means beating a published DCNN by
40% on FD002 is not a plausible result. `ablation.py` exists to find out why, and
the answer is not flattering to the headline:

RMSE, lower is better. The **6-condition** sub-datasets are FD002 and FD004:

| | FD001 (1 cond.) | FD002 (6 cond.) | FD003 (1 cond.) | FD004 (6 cond.) |
|---|---|---|---|---|
| cycle-count only (the floor) | 30.96 | 29.63 | 33.80 | — |
| +cycle +condition-normalisation | 12.53 | 13.10 | 12.90 | 14.40 |
| +cycle −condition-normalisation | 12.44 | 15.54 | 12.59 | 17.19 |
| −cycle +condition-normalisation | 13.51 | 13.89 | 14.17 | 16.16 |
| −cycle −condition-normalisation | 13.44 | **16.32** | 14.11 | **18.10** |

Three things fall out:

1. **Condition normalisation is worth ~2.4 RMSE on FD002 and ~2.8 on FD004, and
   nothing on FD001/FD003.** Exactly as it should be, and the split falls precisely
   along the 1-condition / 6-condition line: FD001 and FD003 fly one operating
   condition, so there is nothing to normalise. Per-regime z-scoring is the single
   change that closes most of the published FD001→FD002 gap.
2. **The `cycle` feature is worth ~0.8 RMSE, and it is an information-set
   difference, not a bug.** The sequence models I quote consume a fixed window of
   sensor readings and nothing else; my feature table also knows how many cycles
   the engine has run. That is legitimate — you always know an asset's hours — but
   it makes the comparison apples-to-oranges, and the `−cycle` rows are the fair
   ones. The cycle-only floor (RMSE ~30) shows the model is not merely regressing
   on age.
3. **Even stripped to `−cycle −condnorm`, FD002 lands at 16.32 and FD004 at 18.10,
   versus the quoted 22.36 and 23.31.** So the remaining gap is the engineered
   degradation features themselves. The honest framing: **these two papers are from
   2017 and 2018 and are quoted as map references, not as the state of the art.**
   Later work reports better. I am not claiming to have beaten the field; I am
   claiming my numbers are in a sensible place and I can tell you which design
   decision bought which part of them.

## What is built

- **All four sub-datasets** with per-regime condition normalisation
  (`src/cmapss.py`). Regimes are recovered by rounding the operating settings —
  they are discrete by construction in C-MAPSS, so k-means would be a slower way
  to get the same six clusters. Sensor selection is by post-normalisation
  variance, which is why FD002/FD004 keep 16 sensors and FD001 keeps 14: `s6` and
  `s10` only become informative once the regime offset is removed.
- **Piecewise-linear RUL target**, implemented *and* justified *and* swept.
  §5 of RESULTS.md scores caps from 90 to uncapped, restricted to the 54 test units
  with true RUL ≤ 90 so a cap-90 model is not penalised for being unable to say a
  number it was never asked to say.

  **The finding is one-sided, and the report says so.** Uncapped scores 2,297 PHM
  against 42 for the best capped model — a 54× gap, concentrated in the metric that
  weights end of life. But *among* the capped rows the comparison is still
  contaminated: RMSE rises monotonically with the cap, which looks like an argument
  for the lowest one, and isn't — a cap-90 model is structurally unable to
  over-predict on a subset selected for RUL ≤ 90, so the restriction hands it the
  advantage rather than removing it. There is no fixed truth on which caps compare
  neutrally. **The cap-neutral test is the decision layer** — refit the alarm policy
  per cap and compare lead time, missed rate and nuisance alarms, all of which live
  below RUL 50.

  **That test is now run** ([EXTENSIONS.md §1](docs/EXTENSIONS.md)) and it settles
  the question: the cost spread across caps 90–200 is **0.08 per unit, nearly
  flat**, where RMSE-on-a-subset had made cap 90 look 40% better than cap 140. So
  the defensible statement is *cap, and do not agonise over the value* — which is
  what the literature's unquestioned 125 has quietly assumed all along.
- **GBM vs LSTM, both reported** (`src/models.py`). The GBM wins on all four
  sub-datasets, on both RMSE and the PHM score, while fitting in 5–12 seconds
  against the LSTM's 87–442. That is the practitioner artifact: deep did not beat
  trees here, and saying so is the point.
- **The PHM08 asymmetric scoring function** (`src/metrics.py`), implemented with
  the asymmetry explained in maintenance terms *and* the conditions under which it
  inverts (cheap asset, cheap failure, expensive spares → running to failure is
  correct and early prediction is the expensive error).
- **Horizon-stratified error.** Error is lowest near failure (3.3 RMSE in `[0,20)`
  on FD001) — which is where the alarm decision is taken. RESULTS.md also refuses
  to read the shape as monotonic: the far band looks easy only because the cap
  makes early life a constant.
- **α-λ prognostic horizon**, computed under *two* cones — the strict ±0.2·RUL and
  a version floored at ±5 cycles. Strict: 13/80 units converge. Floored: 69/80.
  The collapse is a property of the metric, not the model: the strict cone at
  RUL=3 demands sub-cycle accuracy forever after. Both columns are in the document
  rather than the flattering one.
- **The maintenance-decision layer** (`src/alarm.py`): threshold + k-of-consecutive
  persistence, tuned by expected cost, reporting lead-time distribution, missed
  rate, and **nuisance alarms per unit-lifetime**. The persistence curve answers
  "why k=5?" with a table rather than a preference.
- **Two-axis cost sensitivity**, which produced the most useful finding in the
  project: the operating point is completely **insensitive** to the failure cost
  from 5:1 to 100:1 — because at the chosen persistence nothing is missed, so that
  term multiplies zero — and completely **sensitive** to `c_life`, the value of
  discarded remaining life. The number to go and get from finance is the one
  nobody asks for.
- **Edge export**: ONNX fp32 and int8-dynamic, benchmarked through onnxruntime
  pinned to one CPU thread, with size, p50/p99 latency, and the RUL disagreement
  int8 costs.

## Built in the second pass — see [docs/EXTENSIONS.md](docs/EXTENSIONS.md)

- **The cap-neutral test.** RESULTS.md §5 could only support "capping beats not
  capping" and named the decision layer as the neutral comparison. Run: refit the
  alarm policy per cap and compare lead time, missed rate and nuisance alarms.
  **The cost spread across caps 90–200 is 0.08 per unit — nearly flat**, which is
  the opposite of what RMSE-on-a-subset implied.
- **Drift monitoring with a retrain trigger** that requires magnitude *and*
  persistence *and* scope — the scope rule being the one that distinguishes "the
  model is stale" from "a thermocouple is dying". The control window (held-out
  training units) is quiet at 0 features breached; a naive control built from the
  *censored test set* breaches on 36, and that trap is documented rather than
  hidden.
- **Fault-mode discovery on FD003/FD004 with FD001 as a null.** The split carries
  real structure (silhouette 0.42 vs 0.34 null) and mode-aware models still make
  RMSE **worse**, because halving the training data costs more variance than the
  bias it removes. Reported as the negative result it is.

## What is NOT built (the other 50%)

1. **No MLflow tracking or registry**, no experiment lineage, no model versioning.
   The condition normaliser is a fitted artefact that would have to travel with
   the model and there is no mechanism for that.
2. **One sequence model, once.** No TCN, no transformer, no ensemble, no
   hyperparameter search on either family. The LSTM is small on purpose (see
   `src/models.py`) but that also means "deep loses" is a statement about *this*
   LSTM.
3. **No serving.** No API, no container, no batch scoring job.
4. **The edge table is a desktop CPU pinned to one thread**, standing in for a
   gateway. It is not a measurement on gateway hardware and an ARM gateway at
   1.2 GHz will not reproduce it.
5. **Fault-mode handling is discovery, not classification, and it did not pay.**
   The clustering finds real structure but mode-aware models score *worse*. A
   mixture head, or supervision from maintenance records that name the failed
   component, is the next thing to try and is not built.
6. **The drift monitor is input-side only.** No labelled backtest against realised
   failures, so the class of drift where only the sensor-to-life *relationship*
   changes remains uncovered — named in EXTENSIONS.md §2 as process rather than
   code.
7. **20 held-out units per sub-dataset** is a thin basis for a lead-time
   distribution. The P05 lead time is the 5th percentile of at most 20 numbers.
8. **Everything in [docs/DEPLOYMENT_REALITY.md](docs/DEPLOYMENT_REALITY.md)**:
   censored fleets with 4 failures ever, sensor drift and replacement, maintenance-
   log label noise, fleet heterogeneity. C-MAPSS is a luxury and that document is
   the list of ways it is one.

## Layout

```
src/cmapss.py     loading, piecewise RUL, condition normalisation, sensor selection
src/features.py   rolling/trend features (vectorised OLS slope), sequence windowing
src/metrics.py    RMSE, PHM08 score, horizon strata, alpha-lambda
src/models.py     GBM and LSTM
src/alarm.py      alarm policy, cost model, lead time, nuisance alarms
src/edge.py       ONNX export, int8 quantisation, single-thread latency benchmark
train.py          orchestration; writes docs/RESULTS.md
ablation.py       the audit of the FD002/FD004 numbers
```
