"""MLflow tracking and a registry that refuses to ship a model without its normaliser.

THE GAP THIS CLOSES, and why it is not a formality. The first pass ended with:

    "The condition normaliser is a fitted artefact that would have to travel with
     the model and there is no mechanism for that."

That is the whole problem in one sentence, and it is the most common way a
working RUL model produces garbage in production. `ConditionNormaliser` is fitted
on training data: it holds a mean and a standard deviation per operating regime.
The model never sees a raw sensor value -- it sees `(x - mu_r) / sd_r`. Ship the
weights without those statistics and the serving code has three options, all
silently wrong:

  1. normalise with statistics refitted on production data -- which is a
     different transform, drifts as the fleet changes, and makes the model's
     inputs depend on which engines happen to be flying this month
  2. normalise globally instead of per regime -- the exact mistake the
     normaliser exists to prevent, worth ~9 RMSE on FD002
  3. skip normalisation -- the model receives values ~100x its training scale

None of the three raises an exception. All three produce a number. That is what
makes this a lineage problem rather than a packaging problem: **the failure is
silent, and the model still looks like it is working.**

So the registry here treats the normaliser as part of the model, not as a
neighbouring file, and `load_bundle` cannot return one without the other.

WHAT IS DELIBERATELY NOT CLAIMED. This is MLflow's local file backend -- a
directory of runs, not a tracking server, and `mlflow.pyfunc` rather than a
deployment target. Multi-user concurrency, artifact stores on S3, model-stage
transitions with approvals, and lineage back to a training-data snapshot ID are
all real parts of a registry and none of them are here.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import pathlib
import pickle
import platform
import sys
import time
from typing import Any

import numpy as np

try:
    import mlflow
    HAVE_MLFLOW = True
except Exception:                                            # pragma: no cover
    HAVE_MLFLOW = False


# ---------------------------------------------------------------------------
# the bundle
# ---------------------------------------------------------------------------

class ModelBundle:
    """A model plus every fitted artefact needed to reproduce its inputs.

    The fields are not decoration. Each one is something that, if it drifted
    apart from the weights, would change the model's output without changing the
    model:

      normaliser   per-regime mean/sd -- the transform the model was trained under
      sensors      the selected sensor columns, IN ORDER. Column order is a
                   silent killer: reorder two sensors and the model still runs.
      feature_cols the built feature names, in order, for the same reason
      window       sequence length; a model trained on 30 cycles fed 20 is not
                   an error, it is a shorter history silently padded
      rul_cap      the piecewise-linear cap the targets were built under. A model
                   trained at cap 125 and evaluated against uncapped truth is
                   scored on a different problem than it solved.
    """

    def __init__(self, model: Any, normaliser: Any, sensors: list[str],
                 feature_cols: list[str], window: int, rul_cap: int | None,
                 kind: str, metrics: dict | None = None,
                 params: dict | None = None) -> None:
        self.model = model
        self.normaliser = normaliser
        self.sensors = list(sensors)
        self.feature_cols = list(feature_cols)
        self.window = int(window)
        self.rul_cap = rul_cap
        self.kind = kind
        self.metrics = dict(metrics or {})
        self.params = dict(params or {})

    # -- the check that makes the bundle worth having ----------------------
    def validate(self) -> None:
        """Raise unless the bundle is internally consistent.

        Called on save AND on load. On load is the important one: a bundle can be
        corrupted by an edit to the training code between writing and reading it,
        and the failure mode this guards is a model scoring happily against the
        wrong column order.
        """
        if self.normaliser is None:
            raise ValueError(
                "refusing a bundle with no normaliser: the model was trained on "
                "per-regime z-scores and cannot be scored on raw sensor values")
        ncols = getattr(self.normaliser, "cols", None)
        if not ncols:
            raise ValueError("normaliser has no fitted columns -- it was never fit()")
        missing = [s for s in self.sensors if s not in ncols]
        if missing:
            raise ValueError(
                f"normaliser was fitted without {missing}; it cannot transform the "
                "sensors this model consumes")
        if not getattr(self.normaliser, "stats", None):
            raise ValueError("normaliser has no per-regime statistics")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if not self.feature_cols:
            raise ValueError("bundle has no feature columns recorded")

    # -- content addressing ------------------------------------------------
    def fingerprint(self) -> str:
        """Hash of everything that changes the model's output but not its weights.

        Two bundles with the same weights and different fingerprints will score
        differently. That is the point: it gives the serving side something to
        compare against, so a mismatched normaliser is a startup failure rather
        than a slow accuracy regression nobody attributes to it.
        """
        h = hashlib.sha256()
        h.update(json.dumps({
            "sensors": self.sensors, "feature_cols": self.feature_cols,
            "window": self.window, "rul_cap": self.rul_cap, "kind": self.kind,
        }, sort_keys=True).encode())
        stats = getattr(self.normaliser, "stats", {})
        for r in sorted(stats):
            mu, sd = stats[r]
            h.update(np.asarray(mu, dtype=np.float64).tobytes())
            h.update(np.asarray(sd, dtype=np.float64).tobytes())
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

class Registry:
    """A versioned local model store. One directory per (name, version)."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, name: str, version: int) -> pathlib.Path:
        return self.root / name / f"v{version}"

    def versions(self, name: str) -> list[int]:
        d = self.root / name
        if not d.exists():
            return []
        out = []
        for p in d.iterdir():
            if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit():
                out.append(int(p.name[1:]))
        return sorted(out)

    def save(self, name: str, bundle: ModelBundle, stage: str = "None") -> dict:
        bundle.validate()
        version = (max(self.versions(name)) + 1) if self.versions(name) else 1
        d = self._dir(name, version)
        d.mkdir(parents=True, exist_ok=True)

        # torch modules pickle badly across versions; state_dict is the portable
        # form and the class is reconstructed by the loader.
        payload = {"kind": bundle.kind, "params": bundle.params}
        try:
            import torch
            if hasattr(bundle.model, "state_dict"):
                torch.save(bundle.model.state_dict(), d / "weights.pt")
                payload["weights"] = "weights.pt"
            else:
                (d / "model.pkl").write_bytes(pickle.dumps(bundle.model))
                payload["weights"] = "model.pkl"
        except Exception:                                    # pragma: no cover
            (d / "model.pkl").write_bytes(pickle.dumps(bundle.model))
            payload["weights"] = "model.pkl"

        # The normaliser is written NEXT TO the weights, in the same directory,
        # by the same call. There is no path where one is written and the other
        # is not -- which is the entire mechanism.
        (d / "normaliser.pkl").write_bytes(pickle.dumps(bundle.normaliser))

        meta = {
            "name": name, "version": version, "stage": stage,
            "kind": bundle.kind, "fingerprint": bundle.fingerprint(),
            "sensors": bundle.sensors, "feature_cols": bundle.feature_cols,
            "window": bundle.window, "rul_cap": bundle.rul_cap,
            "metrics": bundle.metrics, "params": bundle.params,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0], "platform": platform.platform(),
            **payload,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return meta

    def load_bundle(self, name: str, version: int | None = None,
                    model_factory=None) -> ModelBundle:
        """Load a bundle, or raise. There is no partial success.

        `model_factory(meta) -> nn.Module` reconstructs a torch model to load a
        state_dict into. Omitting it for a torch model raises rather than
        returning a bundle whose `.model` is None -- a bundle you cannot score
        with is not a bundle.
        """
        vs = self.versions(name)
        if not vs:
            raise FileNotFoundError(f"no versions of {name!r} in {self.root}")
        version = max(vs) if version is None else version
        d = self._dir(name, version)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))

        norm_path = d / "normaliser.pkl"
        if not norm_path.exists():
            raise FileNotFoundError(
                f"{name} v{version} has weights but no normaliser.pkl -- refusing "
                "to load. Scoring without it silently changes the model's inputs.")
        normaliser = pickle.loads(norm_path.read_bytes())

        if meta["weights"].endswith(".pt"):
            import torch
            if model_factory is None:
                raise ValueError(
                    f"{name} v{version} is a torch model; pass model_factory to "
                    "rebuild the architecture before loading its state_dict")
            model = model_factory(meta)
            model.load_state_dict(torch.load(d / meta["weights"], map_location="cpu"))
            model.eval()
        else:
            model = pickle.loads((d / meta["weights"]).read_bytes())

        b = ModelBundle(model, normaliser, meta["sensors"], meta["feature_cols"],
                        meta["window"], meta["rul_cap"], meta["kind"],
                        meta.get("metrics"), meta.get("params"))
        b.validate()
        got = b.fingerprint()
        if got != meta["fingerprint"]:
            raise ValueError(
                f"{name} v{version} fingerprint mismatch: meta says "
                f"{meta['fingerprint']}, loaded artefacts hash to {got}. The "
                "normaliser on disk is not the one this model was trained with.")
        return b

    def set_stage(self, name: str, version: int, stage: str) -> dict:
        d = self._dir(name, version)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        meta["stage"] = stage
        (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return meta

    def production(self, name: str) -> int | None:
        for v in sorted(self.versions(name), reverse=True):
            meta = json.loads((self._dir(name, v) / "meta.json").read_text(encoding="utf-8"))
            if meta.get("stage") == "Production":
                return v
        return None

    def index(self) -> list[dict]:
        rows = []
        for nd in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for v in self.versions(nd.name):
                rows.append(json.loads(
                    (self._dir(nd.name, v) / "meta.json").read_text(encoding="utf-8")))
        return rows


# ---------------------------------------------------------------------------
# mlflow tracking
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def tracking(experiment: str, run_name: str, uri: pathlib.Path):
    """MLflow run context that degrades to a no-op if mlflow is unavailable.

    Degrading rather than failing is deliberate: tracking is observability, and
    observability that can break the training run it observes is a liability.
    """
    if not HAVE_MLFLOW:
        yield None
        return
    mlflow.set_tracking_uri(pathlib.Path(uri).as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_run(run, params: dict, metrics: dict, artefacts: dict | None = None) -> None:
    if run is None or not HAVE_MLFLOW:
        return
    for k, v in params.items():
        with contextlib.suppress(Exception):
            mlflow.log_param(k, v)
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and np.isfinite(v):
            with contextlib.suppress(Exception):
                mlflow.log_metric(k, float(v))
    for name, path in (artefacts or {}).items():
        with contextlib.suppress(Exception):
            mlflow.log_artifact(str(path), artifact_path=name)
