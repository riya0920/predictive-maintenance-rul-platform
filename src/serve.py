"""Scoring service and batch scorer, built on the registry bundle.

WHAT MAKES A RUL SERVICE DIFFERENT from a generic model endpoint, and why this is
more than a FastAPI wrapper:

1. **It is stateful in the input, not the model.** A RUL model consumes a WINDOW
   of history. A request carrying one cycle of sensor data is not scorable, and
   the service has to say so rather than pad it. Left-padding a short history --
   which is exactly what the training-time windower does -- is correct during
   training, where the pad is a real observed first cycle, and wrong at serving
   time, where it fabricates history the engine does not have. The distinction is
   invisible in the output: both produce a number.

2. **The normaliser must match.** See registry.py. The service refuses to start
   against a bundle whose fingerprint does not verify.

3. **A RUL number alone is not actionable.** The alarm policy (threshold + k
   consecutive cycles) is what turns a prediction into a work order, and it lives
   with the serving layer because it is an operating decision, not a model
   property. The response carries the prediction, the policy verdict, and the
   inputs the verdict depended on.

4. **Out-of-distribution input is the normal case, not the exception.** An engine
   in a regime the normaliser never saw gets scored against the global fallback
   statistics, silently. The service reports regime coverage on every request so
   that "this prediction is extrapolation" is visible at the point of use rather
   than discoverable later in a drift report.

NOT BUILT, and stated so the endpoint is not mistaken for a deployment: no
authentication, no rate limiting, no request tracing, no model warm-up on start,
no graceful drain, no horizontal scaling story, and the batch scorer is a loop
rather than a job with checkpointing.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# the scorer -- framework-free, so it is testable without a web server
# ---------------------------------------------------------------------------

class Scorer:
    """Turns raw sensor history into a RUL prediction and an alarm verdict."""

    def __init__(self, bundle, predict_fn, threshold: float, k: int,
                 cmapss_mod, features_mod) -> None:
        self.b = bundle
        self.predict_fn = predict_fn
        self.threshold = float(threshold)
        self.k = int(k)
        self._cmapss = cmapss_mod
        self._features = features_mod
        self.b.validate()
        self._fingerprint = self.b.fingerprint()

    # -- input contract ----------------------------------------------------
    def check_history(self, df: pd.DataFrame) -> dict:
        """Validate a unit's history BEFORE scoring it.

        Returns a dict rather than raising so a batch caller can score what is
        scorable and report the rest, instead of failing the whole batch on one
        short engine.
        """
        problems = []
        need = set(self.b.sensors) | {"cycle"}
        missing = sorted(need - set(df.columns))
        if missing:
            problems.append(f"missing columns: {missing}")
        n = len(df)
        if n < self.b.window:
            problems.append(
                f"history is {n} cycles, model window is {self.b.window}; "
                "left-padding at serving time fabricates history this engine "
                "does not have")
        if "cycle" in df and not df["cycle"].is_monotonic_increasing:
            problems.append("cycle column is not monotonically increasing")
        for c in self.b.sensors:
            if c in df and not np.isfinite(df[c].to_numpy(dtype=float)).all():
                problems.append(f"non-finite values in {c}")
        return {"ok": not problems, "problems": problems, "n_cycles": n}

    # -- regime coverage ---------------------------------------------------
    def regime_coverage(self, df: pd.DataFrame) -> dict:
        """Fraction of rows falling in a regime the normaliser was actually fit on.

        Anything below 1.0 means part of this history was z-scored with global
        fallback statistics instead of its own regime's, which is a quieter
        version of not normalising at all.
        """
        try:
            reg = self._cmapss.operating_regimes(df)
        except Exception:
            return {"known_fraction": float("nan"), "unknown_regimes": []}
        known = set(getattr(self.b.normaliser, "stats", {}))
        seen = [int(r) for r in np.unique(reg)]
        frac = float(np.mean([int(r) in known for r in reg])) if len(reg) else 1.0
        return {"known_fraction": frac,
                "unknown_regimes": sorted(r for r in seen if r not in known)}

    # -- scoring -----------------------------------------------------------
    def score_unit(self, df: pd.DataFrame) -> dict:
        t0 = time.perf_counter()
        chk = self.check_history(df)
        if not chk["ok"]:
            return {"scorable": False, **chk}

        df = df.sort_values("cycle").reset_index(drop=True)
        if "unit" not in df.columns:
            df = df.assign(unit=0)
        norm = self.b.normaliser.transform(df)
        feats = self._features.build_features(norm, self.b.sensors)
        feats["unit"] = df["unit"].to_numpy()

        missing = [c for c in self.b.feature_cols if c not in feats.columns]
        if missing:
            return {"scorable": False, "problems": [f"feature mismatch: {missing}"]}

        if self.b.kind in ("lstm", "tcn"):
            x, _, idx = self._features.sequence_windows(
                feats, self.b.feature_cols, self.b.window)
            preds = self.predict_fn(x)
            order = np.argsort(idx)
            preds = preds[order]
        else:
            preds = self.predict_fn(feats[self.b.feature_cols].to_numpy(dtype=float))

        preds = np.asarray(preds, dtype=float)
        if self.b.rul_cap is not None:
            preds = np.clip(preds, 0.0, float(self.b.rul_cap))

        # Alarm policy: k consecutive cycles below threshold. Applied over the
        # trailing window only, because that is what a live service has seen.
        below = preds < self.threshold
        tail = below[-self.k:] if len(below) >= self.k else below
        alarm = bool(len(tail) == self.k and tail.all())
        cov = self.regime_coverage(df)
        return {
            "scorable": True,
            "rul": float(preds[-1]),
            "rul_trailing": [round(float(v), 2) for v in preds[-min(10, len(preds)):]],
            "alarm": alarm,
            "policy": {"threshold": self.threshold, "k": self.k},
            "cycles_below_threshold_in_tail": int(tail.sum()),
            "regime_coverage": cov,
            "extrapolating": bool(cov["known_fraction"] < 1.0),
            "model": {"kind": self.b.kind, "fingerprint": self._fingerprint,
                      "window": self.b.window, "rul_cap": self.b.rul_cap},
            "latency_ms": (time.perf_counter() - t0) * 1e3,
        }


# ---------------------------------------------------------------------------
# batch scoring
# ---------------------------------------------------------------------------

def score_batch(scorer: Scorer, df: pd.DataFrame, unit_col: str = "unit") -> dict:
    """Score every unit in a frame. Partial failure is reported, not raised."""
    rows, skipped = [], []
    t0 = time.perf_counter()
    for u, g in df.groupby(unit_col, sort=True):
        r = scorer.score_unit(g.drop(columns=[unit_col]).assign(unit=u))
        if r.get("scorable"):
            rows.append({"unit": int(u), "rul": r["rul"], "alarm": r["alarm"],
                         "extrapolating": r["extrapolating"]})
        else:
            skipped.append({"unit": int(u), "problems": r.get("problems", [])})
    secs = time.perf_counter() - t0
    return {
        "scored": rows, "skipped": skipped,
        "n_scored": len(rows), "n_skipped": len(skipped),
        "seconds": secs,
        "units_per_second": len(rows) / secs if secs > 0 else float("inf"),
        "alarms": sum(1 for r in rows if r["alarm"]),
        "extrapolating": sum(1 for r in rows if r["extrapolating"]),
    }


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------

def _history_model():
    """Build the request model at call time, in a module WITHOUT postponed annotations.

    This has to live at module scope rather than inside build_app. This file uses
    `from __future__ import annotations`, so every annotation is a string at
    runtime; FastAPI resolves a handler's parameter annotation by name against the
    module globals, and a class defined inside build_app is not there. The symptom
    is not an error -- FastAPI silently decides the unresolvable parameter must be
    a QUERY parameter and every POST returns 422 'field required'.
    """
    from pydantic import BaseModel

    class History(BaseModel):
        cycle: list[float]
        sensors: dict[str, list[float]]
        settings: dict[str, list[float]] | None = None

    return History


History = None          # bound on first build_app() call; see below


def build_app(scorer: Scorer):
    """FastAPI app. Imported lazily so the rest of the module needs no web stack."""
    from fastapi import FastAPI, HTTPException

    global History
    if History is None:
        History = _history_model()
    globals()["History"] = History

    app = FastAPI(title="RUL scoring", version="1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": scorer.b.kind,
                "fingerprint": scorer._fingerprint, "window": scorer.b.window}

    @app.get("/model")
    def model():
        return {"kind": scorer.b.kind, "fingerprint": scorer._fingerprint,
                "sensors": scorer.b.sensors, "window": scorer.b.window,
                "rul_cap": scorer.b.rul_cap, "policy":
                {"threshold": scorer.threshold, "k": scorer.k},
                "metrics": scorer.b.metrics}

    @app.post("/score")
    def score(h: History):        # noqa: F821 -- bound above, resolved at module scope
        cols = {"cycle": h.cycle, **h.sensors, **(h.settings or {})}
        n = len(h.cycle)
        if any(len(v) != n for v in cols.values()):
            raise HTTPException(400, "all series must have the same length as cycle")
        out = scorer.score_unit(pd.DataFrame(cols))
        if not out.get("scorable"):
            raise HTTPException(422, {"problems": out.get("problems")})
        return out

    return app


def write_container(root: pathlib.Path, model_name: str) -> dict:
    """Emit a Dockerfile and compose file for the service.

    Written to disk rather than built: no container runtime is available in this
    environment, so this is a reviewable artefact and NOT a verified image. Saying
    that plainly is the difference between shipping a deployment story and
    claiming one.
    """
    root.mkdir(parents=True, exist_ok=True)
    dockerfile = f"""# NOT BUILT OR RUN -- emitted by src/serve.py, never executed here.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pandas scikit-learn torch --index-url \\
    https://download.pytorch.org/whl/cpu && \\
    pip install --no-cache-dir fastapi uvicorn
COPY src/ /app/src/
COPY registry/{model_name}/ /app/registry/{model_name}/
ENV MODEL_NAME={model_name}
EXPOSE 8000
# The healthcheck hits /health, which verifies the bundle fingerprint on start.
HEALTHCHECK --interval=30s --timeout=3s CMD python -c \\
    "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    (root / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    compose = """# NOT BUILT OR RUN. Resource limits are the point of including this:
# the edge table in RESULTS.md is measured on an unconstrained desktop, and this
# is where the constraint would be applied and re-measured.
services:
  rul:
    build: .
    ports: ["8000:8000"]
    deploy:
      resources:
        limits: {cpus: "0.50", memory: 512M}
"""
    (root / "compose.yaml").write_text(compose, encoding="utf-8")
    return {"dockerfile": str(root / "Dockerfile"),
            "compose": str(root / "compose.yaml"), "built": False,
            "note": "emitted for review; no container runtime in this environment"}
