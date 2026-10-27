"""
predict_surrogate.py
Use a model trained with initial_Cnn.py to predict kappa for every .nc file in a
dataset made of batch folders (batch_00_seed.., batch_01_seed.., ...).

It reads the training settings from the run folder, so the inputs are prepared
exactly as in training:
    runs/<run>/surrogate_model.pt   weights + y_mean / y_std
    runs/<run>/run.json             KAPPA_ATTR, DROP_NONCONVERGED

Usage:
    python predict_surrogate.py --run-dir runs/2026-10-18_14-02-11_scalar_20k \
                                --data-root //wsl.localhost/Ubuntu/home/evin/muTopOpt/new_dataset

Writes into <run-dir>/predict_<dataset-name>/ (or --out):
    predictions.csv   batch, file, converged, true kxx/kxy/kyy, predicted kxx/kxy/kyy, abs errors
    summary.txt       R2 / MAE / RMSE overall, MAE per batch
    parity.png        predicted vs true for kxx and kyy
"""

import argparse
import csv
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from netCDF4 import Dataset

LABEL_NAMES = ["kxx", "kxy", "kyy"]                                            # <<< CHANGED (same as initial_Cnn.py)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True,
                   help="training run folder containing surrogate_model.pt and run.json")      # <<< CHANGED
    p.add_argument("--data-root", required=True,
                   help="folder that contains the batch_XX_seedYY folders")
    p.add_argument("--cnn-script", default="initial_Cnn.py",
                   help="path to the training script that defines MicrostructureCNN")         # <<< ADDED
    p.add_argument("--kappa-attr", default=None, choices=["kappa_eff", "kappa_thresh"],
                   help="override the label used in training (default: read from run.json)")  # <<< ADDED
    p.add_argument("--keep-nonconverged", action="store_true",
                   help="also predict runs with converged=0")                                 # <<< ADDED
    p.add_argument("--out", default=None, help="output folder")
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model (imported from your training script, so the architecture always matches)
# ---------------------------------------------------------------------------
def import_cnn(script_path):                                                   # <<< ADDED
    if not os.path.exists(script_path):
        sys.exit(f"Cannot find {script_path}. Pass --cnn-script <path to initial_Cnn.py>")
    spec = importlib.util.spec_from_file_location("training_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # safe: training only runs under __main__
    return mod.MicrostructureCNN


def load_model(run_dir, cnn_script):
    ckpt = torch.load(os.path.join(run_dir, "surrogate_model.pt"),
                      map_location=DEVICE, weights_only=False)
    model = import_cnn(cnn_script)()
    model.load_state_dict(ckpt["state_dict"])                                  # <<< CHANGED (key used in initial_Cnn.py)
    model.to(DEVICE).eval()
    # scalar normalizer: y_mean = 0, y_std = one number
    # standard normalizer: y_mean, y_std = 3-vectors -> both work with broadcasting
    y_mean = torch.as_tensor(ckpt["y_mean"], dtype=torch.float32).to(DEVICE)   # <<< CHANGED
    y_std = torch.as_tensor(ckpt["y_std"], dtype=torch.float32).to(DEVICE)     # <<< CHANGED
    print(f"Model loaded ({ckpt.get('normalizer', '?')} normalizer, "
          f"y_mean={y_mean.cpu().numpy()}, y_std={y_std.cpu().numpy()})")
    return model, y_mean, y_std


def training_config(run_dir):                                                  # <<< ADDED
    path = os.path.join(run_dir, "run.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("config", {})


# ---------------------------------------------------------------------------
# Data (same preprocessing as load_dataset() in initial_Cnn.py)
# ---------------------------------------------------------------------------
def read_density(ds):
    arr = np.asarray(ds.variables["density"][:], dtype=np.float32)
    if arr.ndim == 3:                     # several frames (--dump-every): take the last one
        arr = arr[-1]
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise RuntimeError(f"density has shape {arr.shape}, expected 2D")
    return arr


def kappa3(flat):
    """flat [kxx, kxy, kyx, kyy] -> [kxx, 0.5*(kxy+kyx), kyy]  (same as _kappa3)."""
    a = np.asarray(flat, dtype=np.float64).reshape(-1)
    return np.array([a[0], 0.5 * (a[1] + a[2]), a[3]], dtype=np.float32)


def is_converged(ds):
    if "converged" not in ds.ncattrs():
        return None
    try:
        return bool(int(getattr(ds, "converged")))
    except Exception:
        return None


def load_batches(data_root, kappa_attr, drop_nonconverged):                    # <<< CHANGED
    batch_dirs = sorted(d for d in glob.glob(os.path.join(data_root, "**", "batch_*"), recursive=True)
                        if os.path.isdir(d))
    if not batch_dirs:
        sys.exit(f"No batch_* folders found under {data_root}")

    xs, ys, rows = [], [], []
    n_conv_skip = n_bad = 0
    for bdir in batch_dirs:
        batch = os.path.basename(bdir)
        files = sorted(glob.glob(os.path.join(bdir, "result2D_*.nc")))
        for f in files:
            try:
                with Dataset(f, "r") as ds:
                    conv = is_converged(ds)
                    if drop_nonconverged and conv is False:
                        n_conv_skip += 1
                        continue
                    rho = read_density(ds)
                    if kappa_attr == "kappa_thresh":          # trained on binarized rho
                        rho = (rho >= 0.5).astype(np.float32)
                    y = (kappa3(getattr(ds, kappa_attr)) if kappa_attr in ds.ncattrs()
                         else np.full(3, np.nan, dtype=np.float32))   # no label -> predict only
            except Exception as e:
                n_bad += 1
                print(f"  skip {f}: {e}")
                continue
            xs.append(rho[None, :, :])
            ys.append(y)
            rows.append({"batch": batch, "file": os.path.basename(f), "converged": conv})
        print(f"  {batch}: {len(files)} files")

    if not xs:
        sys.exit("Loaded 0 samples.")
    shapes = {x.shape for x in xs}
    if len(shapes) != 1:
        sys.exit(f"Mixed grid sizes {shapes} - point --data-root at one res_<grid> folder")

    print(f"Loaded {len(xs)} samples from {len(batch_dirs)} batches, rho shape {xs[0].shape}")
    print(f"  dropped: {n_conv_skip} non-converged, {n_bad} unreadable")
    return np.stack(xs), np.stack(ys), rows


# ---------------------------------------------------------------------------
# Metrics / plot
# ---------------------------------------------------------------------------
def metrics(t, p):
    err = p - t
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    r2 = 1 - np.sum(err ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-12)
    return r2, mae, rmse


def parity_plot(true, pred, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, i in zip(axes, [0, 2]):                    # kxx, kyy
        r2, mae, _ = metrics(true[:, i], pred[:, i])
        ax.scatter(true[:, i], pred[:, i], s=6, alpha=0.6)
        ax.plot([-0.05, 1.05], [-0.05, 1.05], "k--", lw=0.8)
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel(f"true {LABEL_NAMES[i]}"); ax.set_ylabel(f"predicted {LABEL_NAMES[i]}")
        ax.set_title(f"{LABEL_NAMES[i]}:  R2={r2:.3f}  MAE={mae:.4f}")
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    cfg = training_config(args.run_dir)                                        # <<< ADDED
    kappa_attr = args.kappa_attr or cfg.get("KAPPA_ATTR", "kappa_eff")
    drop_nonconv = (not args.keep_nonconverged) and cfg.get("DROP_NONCONVERGED", True)
    print(f"Label: {kappa_attr}  (binarize input: {kappa_attr == 'kappa_thresh'})  "
          f"drop non-converged: {drop_nonconv}  device: {DEVICE}")

    model, y_mean, y_std = load_model(args.run_dir, args.cnn_script)
    X, Y, rows = load_batches(args.data_root, kappa_attr, drop_nonconv)

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), args.batch_size):
            xb = torch.from_numpy(X[i:i + args.batch_size]).to(DEVICE)
            preds.append((model(xb) * y_std + y_mean).cpu().numpy())         # decode, as std.decode()
    pred = np.concatenate(preds)

    name = os.path.basename(os.path.normpath(args.data_root))
    out = args.out or os.path.join(args.run_dir, f"predict_{name}")
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(out, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "file", "converged"]
                   + [f"true_{n}" for n in LABEL_NAMES] + [f"pred_{n}" for n in LABEL_NAMES]
                   + [f"abs_err_{n}" for n in LABEL_NAMES])
        for r, t, p in zip(rows, Y, pred):
            w.writerow([r["batch"], r["file"], r["converged"]]
                       + [f"{v:.6g}" for v in t] + [f"{v:.6g}" for v in p]
                       + [f"{v:.6g}" for v in np.abs(p - t)])

    lines = [f"run: {args.run_dir}", f"data: {args.data_root}",
             f"label: {kappa_attr}   samples: {len(rows)}", ""]
    has = ~np.isnan(Y).any(axis=1)
    if has.any():
        lines.append("Overall")
        for i, n in enumerate(LABEL_NAMES):
            r2, mae, rmse = metrics(Y[has, i], pred[has, i])
            lines.append(f"  {n}: MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.3f}")
        lines += ["", "Per batch"]
        batches = np.array([r["batch"] for r in rows])
        for b in sorted(set(batches)):
            m = (batches == b) & has
            if m.any():
                lines.append(f"  {b}: n={m.sum():4d}  "
                             f"MAE kxx={np.mean(np.abs(pred[m, 0] - Y[m, 0])):.4f}  "
                             f"MAE kyy={np.mean(np.abs(pred[m, 2] - Y[m, 2])):.4f}")
        parity_plot(Y[has], pred[has], os.path.join(out, "parity.png"),
                    f"Prediction on {name} (predicted vs true)")
    else:
        lines.append(f"No '{kappa_attr}' in the .nc files - predictions only.")

    summary = "\n".join(lines)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print("\n" + summary + f"\n\nSaved to {out}")


if __name__ == "__main__":
    main()
