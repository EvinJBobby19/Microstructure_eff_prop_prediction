"""
plot_curves.py -- plots for a training run produced by initial_Cnn.py

Separate from the training script: it only reads the files a run already saved,
so the main file stays uncluttered.

Usage:
  python plot_curves.py                         # uses the most recent run in runs/
  python plot_curves.py runs/2026-10-12_18-00-00_standard

Reads from the run folder:
  history.csv      -> training_curve.png   (train & val loss vs epoch)
  predictions.csv  -> parity_plot.png      (predicted vs true kappa, test set)
PNGs are saved back into that same run folder.

Needs matplotlib:  pip install matplotlib
"""

import csv
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR    = "runs"
LABEL_NAMES = ["kxx", "kxy", "kyy"]


def latest_run():
    """Most recently created run folder (names are timestamped, so sorting works)."""
    runs = [d for d in sorted(glob.glob(os.path.join(RUNS_DIR, "*"))) if os.path.isdir(d)]
    if not runs:
        raise SystemExit(f"No run folders found in {RUNS_DIR}/")
    return runs[-1]


# ---------------------------------------------------------------------------
# Training curve: train & validation loss vs epoch
# ---------------------------------------------------------------------------
def plot_training_curve(run_dir):
    path = os.path.join(run_dir, "history.csv")
    if not os.path.exists(path):
        print("  no history.csv found; skipping training curve")
        return
    ep, tr, va = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ep.append(int(row["epoch"]))
            tr.append(float(row["train_loss"]))
            va.append(float(row["val_loss"]))
    ep, tr, va = np.array(ep), np.array(tr), np.array(va)
    best = int(ep[np.argmin(va)])                      # epoch of lowest val loss

    plt.figure(figsize=(7, 4.5))
    plt.plot(ep, tr, label="train loss", color="#378ADD", lw=1.5)
    plt.plot(ep, va, label="validation loss", color="#D85A30", lw=1.5)
    plt.axvline(best, ls="--", color="gray", lw=1, label=f"best epoch ({best})")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss (scaled)")
    plt.yscale("log")                                  # loss spans orders of magnitude; log reads better
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    out = os.path.join(run_dir, "training_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Parity plot: predicted vs true on the test set (the honest "test" plot)
# ---------------------------------------------------------------------------
def plot_parity(run_dir):
    path = os.path.join(run_dir, "predictions.csv")
    if not os.path.exists(path):
        print("  no predictions.csv found; skipping parity plot")
        return
    true = {n: [] for n in LABEL_NAMES}
    pred = {n: [] for n in LABEL_NAMES}
    with open(path) as f:
        for row in csv.DictReader(f):
            for n in LABEL_NAMES:
                true[n].append(float(row[f"true_{n}"]))
                pred[n].append(float(row[f"pred_{n}"]))
    true = {n: np.array(v) for n, v in true.items()}
    pred = {n: np.array(v) for n, v in pred.items()}

    comps = ["kxx", "kyy"]                             # kxy ~ 0, not meaningful to plot
    fig, axes = plt.subplots(1, len(comps), figsize=(5 * len(comps), 4.5))
    if len(comps) == 1:
        axes = [axes]
    for ax, n in zip(axes, comps):
        t, p = true[n], pred[n]
        err = p - t
        mae = np.mean(np.abs(err))
        r2 = 1 - np.sum(err**2) / (np.sum((t - t.mean())**2) + 1e-12)
        lo = min(t.min(), p.min()); hi = max(t.max(), p.max())
        pad = 0.05 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                ls="--", color="gray", lw=1)           # y = x: perfect predictions lie here
        ax.scatter(t, p, s=18, color="#378ADD", alpha=0.7, edgecolors="none")
        ax.set_xlabel(f"true {n}")
        ax.set_ylabel(f"predicted {n}")
        ax.set_title(f"{n}:  R2={r2:.3f}   MAE={mae:.4f}")
        ax.set_aspect("equal", "box")
        ax.grid(alpha=0.3)
    fig.suptitle("Test-set parity (predicted vs true)")
    fig.tight_layout()
    out = os.path.join(run_dir, "parity_plot.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else latest_run()
    print(f"Run folder: {run_dir}")
    plot_training_curve(run_dir)
    plot_parity(run_dir)


if __name__ == "__main__":
    main()
