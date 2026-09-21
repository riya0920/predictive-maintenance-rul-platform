"""Draw the README demo figure from the held-out engines.

Predicted vs true remaining life for a few FD001 engines the model never trained
on, with the point where the alarm policy (RUL < 25 for 5 cycles in a row) fires.

    python train.py          # writes out/holdout_FD001.npz
    python make_demo.py      # writes docs/img/rul_demo.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / "docs" / "img"
THRESHOLD, K = 25, 5  # the FD001 policy reported in RESULTS.md


def first_alarm(pred: np.ndarray) -> int | None:
    run = 0
    for i, below in enumerate(pred < THRESHOLD):
        run = run + 1 if below else 0
        if run == K:
            return i
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(ROOT / "out" / "holdout_FD001.npz")
    units = np.unique(d["unit"])[:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharey=True)
    for ax, u in zip(axes.flat, units):
        m = d["unit"] == u
        pred, true = d["pred"][m], d["true"][m]
        cycles = np.arange(1, len(pred) + 1)
        ax.plot(cycles, true, color="#999999", lw=2, label="true RUL")
        ax.plot(cycles, pred, color="#1f77b4", lw=1.2, label="predicted RUL")
        ax.axhline(THRESHOLD, color="#d62728", ls=":", lw=1, label="alarm threshold")
        i = first_alarm(pred)
        if i is not None:
            lead = len(pred) - 1 - i
            ax.axvline(cycles[i], color="#d62728", lw=1.2)
            ax.annotate(f"alarm: {lead} cycles before failure",
                        xy=(cycles[i], 112), xytext=(-8, 0),
                        textcoords="offset points", ha="right",
                        color="#d62728", fontsize=9,
                        bbox=dict(fc="white", ec="none", pad=1))
        ax.set_title(f"Engine {u}", loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="center left")
    for ax in axes[1]:
        ax.set_xlabel("engine cycle")
    for ax in axes[:, 0]:
        ax.set_ylabel("remaining life (cycles)")
    fig.suptitle("Held-out FD001 engines: predicted life tracks the truth, "
                 "and the alarm fires with warning to spare", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "rul_demo.png", dpi=130)
    print("wrote", OUT / "rul_demo.png")


if __name__ == "__main__":
    main()
