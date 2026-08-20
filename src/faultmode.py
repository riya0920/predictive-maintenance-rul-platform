"""Fault-mode discovery for FD003/FD004, and mode-aware RUL models.

The named gap from the first build: FD003 and FD004 carry TWO fault modes (HPC
degradation and fan degradation), a single model averages both, and the RESULTS.md
gap analysis said closing it needs either a fault-mode classifier upstream or a
mixture head. This is the classifier.

THE HARD PART IS THAT THE MODE LABEL DOES NOT EXIST.
C-MAPSS does not tell you which unit has which fault. So this cannot be a
classifier in the supervised sense -- it has to DISCOVER the modes from the
degradation signature and then be checked for whether the discovery was real.

The approach, and why each step is the way it is:

  1. SIGNATURE, not snapshot. A unit's fault mode shows up in WHICH sensors
     degrade and in WHAT DIRECTION, not in any single reading. So each unit is
     summarised by the per-sensor slope over the last third of its life --
     normalised per sensor across the fleet so a sensor with large units does not
     dominate the distance.

  2. CLUSTER IN 2, because the data sheet says two fault modes. This is a place
     where domain knowledge beats model selection: running a silhouette sweep and
     letting it pick k would be defensible-looking and would discard the one piece
     of ground truth available.

  3. VERIFY THE CLUSTERING IS REAL before using it. Two checks:
       - separation: silhouette score of the 2-way split
       - a NULL: the same clustering run on FD001, which has ONE fault mode. If
         FD001 splits just as cleanly, the "modes" are an artifact of the
         clustering and not a property of the data.
     The null is the part that makes this evidence rather than a picture.

  4. Only then fit per-mode models and measure whether the gap closes.

If the null fires -- FD001 splitting as cleanly as FD003 -- the honest conclusion
is that this method has not found fault modes, and the report says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SEED = 20260818


def unit_signature(df: pd.DataFrame, sensors: list[str],
                   tail_fraction: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """One vector per unit: the per-sensor degradation slope over its late life.

    Late life only, because that is where the fault mode expresses itself. Early
    life is healthy for every unit regardless of which component will eventually
    fail, so including it dilutes the signal with a shared prefix -- which is the
    same reason the RUL target is capped.
    """
    units = np.sort(df["unit"].unique())
    sig = np.zeros((len(units), len(sensors)))
    for i, u in enumerate(units):
        d = df[df["unit"] == u]
        n = len(d)
        tail = d.iloc[int(n * (1 - tail_fraction)):]
        x = np.arange(len(tail), dtype=float)
        x = x - x.mean()
        denom = float((x * x).sum()) or 1.0
        for j, c in enumerate(sensors):
            y = tail[c].to_numpy(dtype=float)
            sig[i, j] = float((x * (y - y.mean())).sum() / denom)
    return sig, units


def _standardise(sig: np.ndarray) -> np.ndarray:
    sd = sig.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (sig - sig.mean(axis=0)) / sd


def discover_modes(df: pd.DataFrame, sensors: list[str], k: int = 2,
                   tail_fraction: float = 0.35) -> dict:
    """Cluster units by degradation signature. Returns labels + separation quality."""
    sig, units = unit_signature(df, sensors, tail_fraction)
    z = _standardise(sig)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(z)
    labels = km.labels_
    sil = float(silhouette_score(z, labels)) if len(set(labels)) > 1 else float("nan")
    sizes = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    # Which sensors separate the clusters? The interpretable output -- a mode you
    # cannot describe in terms of sensors is a cluster id, not a fault mode.
    centres = km.cluster_centers_
    sep = np.abs(centres[0] - centres[1]) if k == 2 else np.zeros(len(sensors))
    top = sorted(zip(sensors, sep), key=lambda t: -t[1])[:6]
    return {
        "labels": labels, "units": units, "kmeans": km,
        "silhouette": sil, "cluster_sizes": sizes,
        "separating_sensors": [{"sensor": s, "centroid_gap": float(v)} for s, v in top],
        "signature": sig, "z": z,
    }


def assign_modes(train_df: pd.DataFrame, test_df: pd.DataFrame, sensors: list[str],
                 disc: dict, tail_fraction: float = 0.35) -> np.ndarray:
    """Assign a mode to each TEST unit using the clustering fitted on train.

    Test units are right-censored, so their "late life" is late relative to where
    they were truncated, not relative to failure. That is a real weakness and it is
    stated in the report: a unit truncated at 40% of its life has no late-life
    signature to speak of, and its mode assignment is close to a guess.
    """
    sig, _ = unit_signature(test_df, sensors, tail_fraction)
    tr_sig = disc["signature"]
    mu, sd = tr_sig.mean(axis=0), tr_sig.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return disc["kmeans"].predict((sig - mu) / sd)


def null_check(fd001_train: pd.DataFrame, sensors: list[str],
               tail_fraction: float = 0.35) -> dict:
    """Run the same discovery on FD001, which has ONE fault mode.

    This is the control. If a single-fault-mode dataset splits into two clusters
    just as cleanly, then silhouette on this signature is measuring the clustering
    algorithm's willingness to split, not the presence of modes.
    """
    d = discover_modes(fd001_train, sensors, k=2, tail_fraction=tail_fraction)
    return {"silhouette": d["silhouette"], "cluster_sizes": d["cluster_sizes"]}
