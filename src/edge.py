"""Could this run on the gateway next to the asset?

Industrial inference lives at the edge for three reasons worth naming:
  bandwidth      -- streaming 21 sensors x N assets to a cloud costs money on a
                    cellular or satellite link, forever
  latency        -- not for RUL itself (a cycle is minutes), but the same gateway
                    hosts other analytics and the compute budget is shared
  data residency -- process data is a trade secret, and several customers treat
                    "the model comes to the data" as a contractual requirement
                    rather than an optimisation

So the deliverable is not "we exported an ONNX file", it is a table: size,
latency, and what accuracy the smaller number cost.
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import torch


def export_onnx(model, n_features: int, window: int, path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, window, n_features, dtype=torch.float32)
    model.eval()
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["window"],
        output_names=["rul"],
        dynamic_axes={"window": {0: "batch"}, "rul": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return path


def quantize_dynamic(src: pathlib.Path, dst: pathlib.Path):
    """int8 dynamic quantisation of the weights.

    Dynamic (not static) because the activation ranges of an LSTM over a
    condition-normalised window are input dependent and we have no calibration
    protocol on the gateway. Weight-only int8 is the honest edge move here.
    """
    try:
        from onnxruntime.quantization import QuantType
        from onnxruntime.quantization import quantize_dynamic as qd

        qd(str(src), str(dst), weight_type=QuantType.QInt8)
        return dst
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"    [edge] dynamic quantisation unavailable: {exc}")
        return None


def bench_onnx(path: pathlib.Path, x: np.ndarray, n_iter: int = 300) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1  # gateway-class: one core for this workload
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    single = x[:1]
    for _ in range(20):
        sess.run(None, {name: single})
    lat = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        sess.run(None, {name: single})
        lat.append((time.perf_counter() - t0) * 1000.0)
    preds = []
    for b in range(0, len(x), 512):
        preds.append(sess.run(None, {name: x[b : b + 512]})[0].reshape(-1))
    return {
        "file_kb": path.stat().st_size / 1024.0,
        "p50_ms": float(np.percentile(lat, 50)),
        "p99_ms": float(np.percentile(lat, 99)),
        "preds": np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32),
    }
