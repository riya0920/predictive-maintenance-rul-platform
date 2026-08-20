"""Two models on the same target, so the deep-vs-trees question gets an answer
with a complexity cost attached rather than a vibe.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch import nn

SEED = 20260818


def fit_gbm(x: np.ndarray, y: np.ndarray) -> tuple[HistGradientBoostingRegressor, float]:
    m = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.06,
        max_depth=None,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED,
    )
    t0 = time.perf_counter()
    m.fit(x, y)
    return m, time.perf_counter() - t0


class RULLSTM(nn.Module):
    """Two-layer LSTM over a fixed window, mean+last pooled, linear head.

    Deliberately small. The claim under test is 'does a sequence model beat
    engineered features on this data', and answering it with a 20M-parameter
    transformer would answer a different question (does compute beat features).
    """

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        z = torch.cat([out[:, -1, :], out.mean(dim=1)], dim=-1)
        return self.head(z).squeeze(-1)


def fit_lstm(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 25,
    batch: int = 512,
    val_frac: float = 0.1,
    verbose: bool = False,
) -> tuple[RULLSTM, float, dict]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))

    n = len(x)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    vi, ti = perm[:n_val], perm[n_val:]

    xt = torch.from_numpy(x[ti])
    yt = torch.from_numpy(y[ti])
    xv = torch.from_numpy(x[vi])
    yv = torch.from_numpy(y[vi])

    model = RULLSTM(x.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.MSELoss()

    best = (float("inf"), None)
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(xt))
        for b in range(0, len(xt), batch):
            j = idx[b : b + batch]
            opt.zero_grad()
            loss = lossf(model(xt[j]), yt[j])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = float(
                np.mean(
                    [
                        lossf(model(xv[b : b + 2048]), yv[b : b + 2048]).item()
                        for b in range(0, len(xv), 2048)
                    ]
                )
            )
        if vl < best[0]:
            best = (vl, {k: v.detach().clone() for k, v in model.state_dict().items()})
        if verbose:
            print(f"    epoch {ep+1:>2}/{epochs}  val_mse {vl:8.2f}", flush=True)
    if best[1] is not None:
        model.load_state_dict(best[1])
    model.eval()
    return model, time.perf_counter() - t0, {"best_val_mse": best[0], "n_params": sum(p.numel() for p in model.parameters())}


@torch.no_grad()
def predict_lstm(model: RULLSTM, x: np.ndarray, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for b in range(0, len(x), batch):
        out.append(model(torch.from_numpy(x[b : b + batch])).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
