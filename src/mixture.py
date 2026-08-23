"""A soft mixture-of-experts head over fault modes, and why hard splitting failed.

WHAT THE SECOND PASS FOUND. `faultmode.discover_modes` finds real structure in
FD003/FD004 -- the clustering is not noise, and `null_check` confirms it by
running the identical procedure on single-fault-mode FD001 and getting a much
weaker silhouette. Then the mode-aware model, trained by splitting the units into
two groups and fitting one model per group, scored *worse* than a single model.

The README recorded that honestly and named the next thing to try:

    "A mixture head, or supervision from maintenance records that name the failed
     component, is the next thing to try and is not built."

WHY HARD SPLITTING LOSES, which is the reasoning this module is built on. Two
effects fight each other:

  + specialisation   each expert fits one degradation mode instead of averaging
                     two, so its predictions can be sharper
  - sample starvation each expert sees HALF the units. With ~100 training units
                     per sub-dataset, that is ~50 -- and the variance cost of
                     halving the data is large

Hard splitting pays the full sample cost to buy the full specialisation benefit.
It also throws away everything the modes have in common, which for two
degradation modes of the SAME engine is most of the signal: both are still an
engine wearing out, and the sensors still mean what they meant.

A soft mixture is the version that does not make that trade at full price:

    prediction = sum_m  gate_m(unit) * expert_m(x)

Every expert is trained on ALL the data, weighted by the gate. A unit that is 0.9
mode-A and 0.1 mode-B contributes strongly to expert A and weakly to expert B --
so no expert starves, and a unit sitting between modes is not forced into one.

WHAT WOULD BE BETTER AND IS NOT HERE. Joint training of gate and experts by EM or
by backprop through the gate. Here the gate is fitted first (from the clustering
already built) and frozen, which is a two-stage approximation chosen because it
reuses the existing, already-validated mode discovery instead of introducing a
second unvalidated thing at the same time.
"""
from __future__ import annotations

import numpy as np


def soft_gate(distances: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax over negative distance to each mode centroid.

    Temperature is the knob that spans the whole design space, which is why it is
    exposed and swept rather than fixed:

        T -> 0    the gate becomes one-hot; the mixture becomes the hard split
                  that already lost
        T -> inf  the gate becomes uniform; every expert sees the same weighted
                  data and the mixture collapses to a single averaged model

    Somewhere between the two is the useful region, and where it is depends on
    how separated the modes actually are. Sweeping it turns "should we model
    fault modes separately" from an opinion into a curve.
    """
    d = np.asarray(distances, dtype=float)
    z = -d / max(temperature, 1e-9)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def centroid_distances(sig: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Euclidean distance from each unit signature to each mode centroid."""
    s = np.asarray(sig, dtype=float)
    c = np.asarray(centroids, dtype=float)
    return np.sqrt(((s[:, None, :] - c[None, :, :]) ** 2).sum(-1))


class MixtureOfExperts:
    """Weighted experts with a frozen gate.

    `fit_fn(x, y, weights) -> model` and `predict_fn(model, x) -> preds` are
    injected so this works with whichever regressor the caller already trusts,
    rather than importing an opinion about the expert family into a module that
    is about the gating.
    """

    def __init__(self, fit_fn, predict_fn) -> None:
        self.fit_fn = fit_fn
        self.predict_fn = predict_fn
        self.experts: list = []
        self.n_modes = 0

    def fit(self, x: np.ndarray, y: np.ndarray, row_gate: np.ndarray,
            min_weight: float = 1e-3) -> "MixtureOfExperts":
        """Fit one expert per mode on ALL rows, weighted by that mode's gate."""
        self.n_modes = row_gate.shape[1]
        self.experts = []
        for m in range(self.n_modes):
            w = np.clip(row_gate[:, m], min_weight, None)
            self.experts.append(self.fit_fn(x, y, w))
        return self

    def predict(self, x: np.ndarray, row_gate: np.ndarray) -> np.ndarray:
        p = np.stack([self.predict_fn(e, x) for e in self.experts], axis=1)
        return (p * row_gate).sum(axis=1)

    def expert_predictions(self, x: np.ndarray) -> np.ndarray:
        return np.stack([self.predict_fn(e, x) for e in self.experts], axis=1)


def effective_sample_size(gate: np.ndarray) -> np.ndarray:
    """Kish effective sample size per expert: (sum w)^2 / sum w^2.

    This is the number that makes the sample-starvation argument concrete rather
    than rhetorical. A hard split of 100 units gives each expert exactly 50. A
    soft gate at a sensible temperature typically gives each expert an effective
    70-90 -- and the difference between 50 and 80 units is the difference the
    mixture is trying to buy.
    """
    g = np.asarray(gate, dtype=float)
    return (g.sum(0) ** 2) / np.maximum((g ** 2).sum(0), 1e-12)


def gate_entropy(gate: np.ndarray) -> float:
    """Mean normalised entropy of the gate. 0 = hard split, 1 = uniform.

    Reported alongside the score so the temperature sweep is interpretable: an
    RMSE that improves as entropy approaches 1 is not evidence for a mixture, it
    is evidence that a single model was right all along.
    """
    g = np.clip(np.asarray(gate, dtype=float), 1e-12, 1.0)
    h = -(g * np.log(g)).sum(1)
    return float(np.mean(h) / np.log(g.shape[1]))
