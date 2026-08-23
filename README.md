# ML-1 — Predictive Maintenance Platform (RUL + deployment reality)

**Status: complete.** The measurement, the maintenance-decision layer, the
edge-export story, **drift monitoring with a retrain trigger**, and **fault-mode
discovery** are built and run end to end on **real NASA C-MAPSS data, all four
sub-datasets** — plus a model registry that enforces artefact lineage, a scoring
service, a TCN and an ensemble, concept-drift detection, and the
deployment-reality numbers priced. See
[what is NOT built](#what-is-not-built) — and read
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

## Completed in the third pass — see [docs/COMPLETION.md](docs/COMPLETION.md)

```bash
python complete.py        # ~45 min; all eight remaining items
```

All eight items this README previously listed as not built. Three of them closed
by producing a result I did not expect, and the run found six bugs of my own.

- **A registry that refuses to ship a model without its normaliser.** The gap was
  named in pass 1 and it is not a packaging nicety: the model is trained on
  per-regime z-scores, so weights without those statistics still return plausible
  numbers. Three attempted violations, three refusals — including swapping the
  normaliser on disk for a *differently fitted but perfectly valid* one, caught by
  an artefact fingerprint. Uncaught, that swap would have scored
  **RMSE 36.1 instead of 14.8**:
  degraded, and nowhere near broken enough for anyone to suspect the normaliser.
- **A TCN, a hyperparameter grid, and an ensemble — and "deep loses" survives.**
  The caveat was that the verdict was a statement about one small LSTM. Given a
  dilated causal TCN with a 61-cycle receptive field and a
  grid over width and dropout, the TCN scores **16.46
  RMSE against the GBM's 14.76** — worse, and the
  simplex-constrained blend gives it a weight of **zero**. The blend
  (14.01) beats the best single model by
  0.75 RMSE, on GBM and LSTM alone.
- **Serving.** 100 units batch-scored, a live HTTP surface,
  and a Dockerfile that is written but **not built** — there is no container
  runtime here and saying otherwise would be the overclaim. The design decision
  worth defending is the refusal: a history shorter than the model's window is
  rejected rather than left-padded, because padding at serving time fabricates
  history the engine does not have, and both paths return a number.
- **Concept drift — the drift class the input monitor structurally cannot see.**
  Remaining life compressed to 0.65× *without touching a single sensor value*, so
  P(x) is bit-identical and only P(y|x) moves. The PSI monitor's output changes by
  **0.0** across
  2163 rows — silent, and correctly so. The
  residual monitor fires at **z = 6.8**, direction
  *optimistic*, after
  **5 realised failures**. That
  latency is the finding and it is a property of the fleet, not the monitor: for
  an operator with four failures a year, a monitor needing five is a post-mortem.
- **Lead time with intervals.** The P05 was the 5th percentile of 20 numbers.
  Bootstrapped: median **24
  [21, 27]**, P05
  **17 [15, 21]** —
  an interval 6 cycles wide on a point estimate of
  17. Quoting the point alone implies a precision this
  sample does not have.
- **DEPLOYMENT_REALITY.md, priced.** That document asserted C-MAPSS is a luxury
  dataset and attached no number to "much harder". Each row now degrades the data
  along one axis and refits, and **the ordering is the deliverable** — it says
  where a programme's first year of data-quality work should go. Censoring to
  8 observed failures costs **+9.0 RMSE**;
  label noise only starts to bite past sd 15.
- **The mixture head, and the fault modes still barely pay.** A soft
  mixture-of-experts fixes the sample-starvation that sank the hard split — every
  expert trains on all the data, weighted — and wins by **0.16
  RMSE**. That is a rounding error, and after two attempts the honest read is that
  these modes are real in the sensor signatures and nearly useless for prediction.
  I would stop spending on unsupervised gating here.
- **The edge table, with its extrapolation made explicit.** This one *cannot* be
  closed honestly — there is no gateway in this environment. What is fixed is the
  thing that made it misleading: a desktop number presented with no indication of
  how far it travels. Only the measured row is labelled measured; the ARM rows are
  frequency scaling and are marked projected.

### Six bugs this pass found in its own work

- **The concept-drift injection did nothing.** It scaled RUL at each unit's *last*
  cycle, where the piecewise target is 0 by construction. `0 × 0.65 = 0`. The
  monitors correctly reported no change to data that had not changed.
- **The PSI control compared two different populations** — all-cycle training rows
  against 20 end-of-life rows — and reported 85 of 85 features breaching. Same
  class of error as the FD001 test-set control in pass 2, which I had already
  written up.
- **PSI on 20 observations is binning noise.** 74 of 85 features "breached" on
  provably identical data.
- **The build-heterogeneity control trained and scored on the same rows.** It came
  back 12.5 RMSE *better* than the clean baseline, which is the tell.
- **The sensor-drift experiment ran backwards** — drifting the training set rather
  than the deployed data — and was also specified at sd/1000 cycles against
  ~200-cycle lives, so the largest drift injected was 0.1 sd.
- **The residual monitor compared a mean against a per-observation SD** instead of
  the standard error of that mean, making it √n ≈ 4.5× too insensitive. It never
  fired, which looks exactly like a calm process.

## What is NOT built

The eight numbered gaps above are closed. What remains is bounded by the
environment or by the dataset, and none of it is closable by writing more code
here:

1. **No gateway hardware.** The ARM latency rows are frequency-scaled projections
   and are labelled as such. Cache behaviour, memory bandwidth, SIMD width and
   thermal throttling all differ and all move the number, in the pessimistic
   direction.
2. **No container runtime.** `deploy/Dockerfile` and `compose.yaml` are emitted
   and reviewable; neither has ever been built or run.
3. **C-MAPSS is still C-MAPSS.** The deployment-reality table injects *models* of
   censoring, drift, label noise and heterogeneity one at a time. A real fleet has
   all of them at once, and they interact: the units you have labels for are the
   units whose labels are worst. That interaction is not simulated.
4. **20 held-out units.** Bootstrapping now reports how thin that is rather than
   hiding it, but the interval only shrinks with more units, not more analysis.
5. **Fault modes need supervision to be worth anything.** Two unsupervised
   attempts, two near-zero results. The next thing that could work is maintenance
   records naming the failed component, and there are none in C-MAPSS.

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
