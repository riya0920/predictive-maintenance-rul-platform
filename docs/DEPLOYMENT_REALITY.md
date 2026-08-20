# What changes when the assets are real

C-MAPSS is a simulation, and more importantly it is a *generous* simulation. Every
number in `RESULTS.md` is a property of that generosity. This document is the list
of things that are true of C-MAPSS and false of a fleet, written before anyone asks.

## 1. Run-to-failure data is the luxury, not the norm

C-MAPSS gives 100–249 engines per sub-dataset, **every one of them run to failure**,
with the failure event unambiguous and the trajectory complete. That is the single
most unrealistic thing about it.

A real fleet of 300 machines that has had 4 failures ever is the normal case, and
almost none of the supervised framing above survives it:

| what I built | why it breaks | what replaces it |
|---|---|---|
| RUL regression on labelled trajectories | 4 labelled trajectories cannot fit a 400-tree model, let alone an LSTM | anomaly-based **health index** trained on healthy data only (which you have in abundance), with RUL left undefined |
| piecewise-linear target with a 125 cap | the cap is calibrated to a life distribution you have not observed | a health index has no target to cap |
| RMSE / PHM score | needs a true RUL per test point | **time-to-event** metrics: concordance index, or Brier score for survival |
| train/test split by unit | 4 events cannot be split | leave-one-event-out, and confidence intervals wide enough to be embarrassing |

The right first move on a low-event fleet is **survival analysis with censoring**
(Kaplan–Meier for a fleet-level baseline, Cox or a survival forest for covariates),
because censored observations — "this machine has run 8,000 hours and has not
failed" — are *evidence*, and every method in this repo throws them away. C-MAPSS
hides the censoring problem by giving you the answer key.

None of that is built here. Naming it is the honest half.

## 2. The test set here is already censored, and that shaped the design

The one piece of realism C-MAPSS does have: test units are truncated at a random
point and only their RUL at that instant is known. There is no failure event in the
test set at all.

This is why the maintenance-decision layer in `src/alarm.py` runs on **held-out
training units**, not on the test set. Lead time is the distance between an alarm
and a failure; with no failure there is no lead time. Any portfolio that reports
lead-time or prognostic-horizon numbers "on the C-MAPSS test set" has either
redefined the metric or not read the data description.

The cost is real and worth stating: 20 held-out units per sub-dataset is a thin
basis for a lead-time distribution. The P05 lead time in `RESULTS.md` is the 5th
percentile of at most 20 numbers. It is directional, not precise.

## 3. Fleet heterogeneity

Every C-MAPSS engine is the same engine model with the same physics and initial
wear drawn from one distribution. A real fleet has:

- **Model and vintage mixes.** Serial 001 and serial 4,000 are different machines
  with the same part number.
- **Duty-cycle mixes.** The same machine in a Nevada plant and a Gulf Coast plant
  degrades differently, and the operating-condition variables that would tell you
  so are often not instrumented.
- **Site-specific installation effects** — foundation, alignment, upstream process
  — that behave like a permanent per-asset offset.

The consequence is that a single fleet model is a mixture model whether you
acknowledge it or not. The FD004 gap in `RESULTS.md` is a small, clean version of
this: one model averaging two fault modes. In the field it is one model averaging
several dozen unlabelled contexts, and the useful structures are per-asset
normalisation and hierarchical/partial-pooling models — neither built here.

## 4. Sensor drift, replacement, and recalibration

C-MAPSS sensors are perfect: no drift, no gaps, no recalibration, no replacement.
In the field, all four happen, and each corrupts the model in a different way:

- **Drift** looks exactly like degradation. A slowly rising thermocouple offset and
  a slowly degrading turbine are the same signal to a model built on trends. The
  defence is a redundancy/consistency check across sensors, not a better model.
- **Replacement** produces a step change with no physical meaning. A model built on
  `s - first_value` features (which `src/features.py` uses) reads a sensor swap as
  a large instantaneous change in machine health.
- **Recalibration** is a step change that is *documented somewhere else*, usually
  in a maintenance system nobody joined to the sensor data.
- **Gaps.** Windowing code silently treats a 3-day gap as adjacent samples. Every
  sequence model in this repo would do exactly that.

None of these detectors exist here. The minimum viable version is a per-tag
plausibility and step-change monitor at ingest — which is what `SE-1` in this
portfolio is about, and the two are deliberately not wired together.

## 5. Maintenance-log label noise

The largest single gap. C-MAPSS failure times are exact. Real failure labels come
from work orders, and work orders say things like:

- "Replaced bearing" on a date that is the date somebody *typed it in*, not the
  date the machine stopped
- "No fault found" after a genuine intermittent
- a preventive replacement that pre-empted a failure that never happened — a
  **right-censored** event recorded as a maintenance event
- one work order covering three repairs, one of which is the one you care about

A model trained on this has label noise that is *correlated with the thing being
predicted* — machines that get attention have better labels — which is worse than
random noise and cannot be averaged away. Practically, the useful step before any
modelling is a labelling review with the maintenance planners: which work orders
represent a functional failure, and what is the timestamp uncertainty on each. That
is weeks of unglamorous work and it dominates the model choice.

## 6. What the alarm-policy numbers assume

`src/alarm.py` minimises expected cost with `c_fail = 20 × c_pm`, `c_life = 0.02`
per cycle, `c_nuisance = 0.5`. **All four are invented.** The sensitivity table in
`RESULTS.md` shows the operating point across a 5:1 → 100:1 failure-to-maintenance
cost ratio, which is the honest way to present an assumption you cannot source.

In a real deployment those numbers come from finance and reliability, not from the
data scientist, and getting them is a negotiation:

- `c_fail` is not one number. It is a distribution over consequences (secondary
  damage, collateral, safety, lost production at that hour's margin).
- `c_life` is only real if the asset is retired at replacement. If it is overhauled,
  early removal costs less than the linear term implies.
- `c_nuisance` is not just the inspection cost. It includes the credibility cost of
  the alarm, which is nonlinear and which the model cannot see: the third false
  alarm costs far more than the first because it is what gets the system turned off.

That last one is the reason `nuisance alarms per unit-lifetime` is a headline metric
here rather than a footnote.

## 7. Things claimed nowhere in this repo

To be explicit about scope, because the surrounding README says "20%":

- No MLflow tracking or model registry, no retraining pipeline, no drift monitor
  with a documented retrain trigger. Named in the README as missing.
- No transformer or TCN — one sequence model, done once.
- The ONNX latency table is measured on a desktop CPU pinned to one thread. That
  stands in for a gateway; it is **not** a measurement on gateway hardware, and an
  ARM gateway at 1.2 GHz will not reproduce it.
- No fleet-scale serving, no data contract, no versioning of the condition
  normaliser (which is a fitted artefact and would need to travel with the model).
