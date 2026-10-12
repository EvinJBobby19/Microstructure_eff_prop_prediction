"""
Usage:
  python initial_Cnn.py --data-glob "test_dataset_200/**/result2D_*.nc"
  python initial_Cnn.py --tag scalar_test --notes "first run with scalar normalization"

Every run creates its own folder:
  runs/<date>_<time>_<normalizer>[_<tag>]/
      run.json            config, dataset, environment, training summary, test metrics
      predictions.csv     test-set true vs predicted values per sample
      surrogate_model.pt  best model weights + normalizer values
      log.txt             everything printed to the console
"""

import argparse
import csv
import glob
import json
import os
import platform
import re
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from netCDF4 import Dataset

# CONFIG
DENSITY_VAR   = "density"
KAPPA_ATTR    = "kappa_eff"   # achieved kappa attr; switch to "kappa_thresh" for binarized

USE_SCALAR    = False          # False -> per-component Standardizer, True -> ScalarNormalizer
KAPPA_SOLID   = 1.0           # solid-phase conductivity (--kappa-solid in simulate_conduction.py)

DROP_NONCONVERGED = True

DEFAULT_GLOB  = "//wsl.localhost/Ubuntu/home/evin/muTopOpt/test_dataset_5000/**/result2D_*.nc"

VAL_FRAC      = 0.15
TEST_FRAC     = 0.15
BATCH_SIZE    = 64
LR            = 3e-4
MAX_EPOCHS    = 300
PATIENCE      = 300
SPLIT_SEED    = 0
TRAIN_SEED    = 0             # seeds weight init + batch order, so runs are reproducible
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

RUNS_DIR      = "runs"
LABEL_NAMES   = ["kxx", "kxy", "kyy"]


# RUN FOLDER + CONSOLE LOGGING

class Tee:
    """Writes everything printed to the console into log.txt as well."""
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write(s); self.file.write(s)
    def flush(self):
        self.stdout.flush(); self.file.flush()


def make_run_dir(tag=None):
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    norm = "scalar" if USE_SCALAR else "standard"
    name = f"{stamp}_{norm}" + (f"_{tag}" if tag else "")
    path = os.path.join(RUNS_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def to_jsonable(obj):
    """Convert numpy/torch values so json.dump can write them."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return to_jsonable(obj.detach().cpu().numpy())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2)


# DATA LOADING (.nc to rho images + kappa labels)

def _read_density(ds):
    """Read the density field and squeeze the singleton frame dim -> (H, W)."""
    if DENSITY_VAR is not None:
        arr = np.asarray(ds.variables[DENSITY_VAR][:], dtype=np.float32) #float 32 because saves memory avoids type errors (1,64,64) extracted )
    else:
        # fallback: prefer a var literally named 'density', else first that squeezes to 2D
        name = "density" if "density" in ds.variables else None
        if name is None:
            for n, v in ds.variables.items():
                if np.squeeze(np.asarray(v[:])).ndim == 2:
                    name = n; break
        if name is None:
            raise RuntimeError("No density-like variable found; set DENSITY_VAR.")
        arr = np.asarray(ds.variables[name][:], dtype=np.float32)
    arr = np.squeeze(arr)                             # (1,64,64) to (64,64)
    if arr.ndim != 2:  # check for 2d
        raise RuntimeError(f"density squeezed to {arr.shape}, expected 2D.")
    return arr


def _kappa3(flat):
    """flat [kxx,kxy,kyx,kyy] -> symmetric 3-vector [kxx, kxy, kyy]."""
    a = np.asarray(flat, dtype=np.float64).reshape(-1)
    kxx, kxy, kyx, kyy = a[0], a[1], a[2], a[3]
    return np.array([kxx, 0.5 * (kxy + kyx), kyy], dtype=np.float32)   # off diagonal is symmetrized , is usually zero or near to zero 


_TARGET_RE = re.compile(                              
    r"--target-kappa\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")

def _target3_from_cmd(ds):                            
    """Parse --target-kappa out of the command_line attr -> symmetric 3-vector, or NaNs."""
    cmd = getattr(ds, "command_line", "")
    m = _TARGET_RE.search(cmd)
    if not m:
        return np.array([np.nan, 0.0, np.nan], dtype=np.float32)
    g = [float(x) for x in m.groups()] # the 4 captured strings are converted to floats , can add additional print checks here if necessary 
    return np.array([g[0], 0.5 * (g[1] + g[2]), g[3]], dtype=np.float32)


def _is_converged(ds):                                
    """Read the .nc `converged` attr as bool; None if absent."""
    if "converged" not in ds.ncattrs():
        return None
    try:
        return bool(int(getattr(ds, "converged")))
    except Exception:
        return None


def load_dataset(data_glob):
    """Return X (N,1,H,W), Y_achieved (N,3), T_target (N,3)."""
    files = sorted(glob.glob(data_glob, recursive=True))   # returns a list of strings of the paths that are sorted 
    if not files:
        raise SystemExit(f"No files matched: {data_glob}")
    xs, ys, ts = [], [], []
    skipped_read = skipped_conv = 0                   

    for f in files:
        try:
            ds = Dataset(f, "r")
            conv = _is_converged(ds)                  # the check for the convergence 
            if DROP_NONCONVERGED and conv is False:
                skipped_conv += 1; ds.close(); continue
            if KAPPA_ATTR not in ds.ncattrs():  #kappa_eff this is the kappa atribute that we are searching for in the .nc file
                skipped_read += 1; ds.close(); continue
            rho = _read_density(ds)                   
            y = _kappa3(getattr(ds, KAPPA_ATTR))
            t = _target3_from_cmd(ds)                 
            ds.close()
        except Exception as e:
            skipped_read += 1
            print(f"  skip {f}: {e}")
            continue
        xs.append(rho[None, :, :])
        ys.append(y)
        ts.append(t)

    if not xs:
        raise SystemExit("Loaded 0 samples. Check DENSITY_VAR / KAPPA_ATTR / glob.")
    X = np.stack(xs).astype(np.float32)
    Y = np.stack(ys).astype(np.float32)
    T = np.stack(ts).astype(np.float32)
    print(f"Loaded {len(X)} samples, rho shape {X.shape[1:]}, target attr = {KAPPA_ATTR}")
    print(f"  dropped: {skipped_conv} non-converged, {skipped_read} unreadable/no-kappa")
    info = {
        "data_glob": data_glob,
        "data_glob_abspath": os.path.abspath(data_glob),
        "n_files_matched": len(files),
        "n_samples_used": len(X),
        "n_dropped_nonconverged": skipped_conv,
        "n_dropped_unreadable": skipped_read,
        "density_shape": list(X.shape[1:]),
    }
    return X, Y, T, info


def label_spread_report(Y):
    """Diversity check. With random diagonal targets this should look healthy."""
    print("\nLabel spread (achieved kappa):")
    stats = {}
    for i, n in enumerate(LABEL_NAMES):
        col = Y[:, i]
        stats[n] = {"mean": col.mean(), "std": col.std(), "min": col.min(), "max": col.max()}
        print(f"  {n}: mean={col.mean():.4f}  std={col.std():.4f}  "
              f"min={col.min():.4f}  max={col.max():.4f}")
    if Y[:, 0].std() < 0.02:
        print("  !! kxx std is tiny -> labels barely vary; check the generator.")
    print("  (kxy std ~0 is expected: targets are diagonal, so off-diagonal ~0.)")
    return stats


# SPLIT  --  random split is fine: each sample has an independent random target,
# so there are no near-duplicate structures to leak across the split.

def split_indices(n):
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(n)  # Produces the array [0, 1, …, 199] shuffled into random order, e.g. [137, 12, 199, 4, 88, …].
    n_test = int(round(TEST_FRAC * n)) # creates the number of test samples based on the TEST_FRAC and total number of samples n
    n_val  = int(round(VAL_FRAC * n)) # similary this also does the same 
    test = idx[:n_test]
    val  = idx[n_test:n_test + n_val]
    train = idx[n_test + n_val:]  # the splits are done here and the remaining inidces are assigned to the training set
    return train, val, test




#  TARGET STANDARDIZATION (fit on train, invert at inference) 

class Standardizer:
    def __init__(self, y_train):
        self.mean = torch.tensor(y_train.mean(0), dtype=torch.float32)
        self.std  = torch.tensor(y_train.std(0) + 1e-8, dtype=torch.float32)
    def encode(self, y): return (y - self.mean) / self.std
    def decode(self, y): return y * self.std + self.mean


class ScalarNormalizer:
    """
    Scales ALL label components (kxx, kxy, kyy) by ONE shared number.
    Drop-in replacement for Standardizer: same constructor and encode/decode.

    scale: conductivity of the solid phase (recommended).
           If None, falls back to the largest |label| in the training set.
    """
    def __init__(self, y_train, scale=None):
        if scale is None:
            scale = float(abs(y_train).max())
        self.mean = torch.tensor(0.0, dtype=torch.float32)
        self.std  = torch.tensor(float(scale), dtype=torch.float32)
    def encode(self, y): return (y - self.mean) / self.std
    def decode(self, y): return y * self.std + self.mean


# THE CNN

class ConvBlock(nn.Module):
    """conv (circular padding for periodic BCs) -> BN -> ReLU -> downsample."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1,
                              padding_mode="circular")    ### ************ maybe have to change the padding style 
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.ReLU(inplace=True)
        self.down = nn.MaxPool2d(2)
    def forward(self, x):
        return self.down(self.act(self.bn(self.conv(x))))


class MicrostructureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        head_in = 128
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )
    def forward(self, x):
        f = self.pool(self.features(x)).flatten(1)
        return self.head(f)


# LOSS + TRAINING LOOP (Adam, early stopping on val)

def batches(X, Y, idx, bs, shuffle):
    order = np.random.permutation(idx) if shuffle else idx
    for i in range(0, len(order), bs):
        sel = order[i:i + bs] 
        yield (torch.from_numpy(X[sel]).to(DEVICE),
               torch.from_numpy(Y[sel]).to(DEVICE))


def train(X, Y, tr, va, run_dir):
    model = MicrostructureCNN().to(DEVICE)

    if USE_SCALAR:
        std = ScalarNormalizer(Y[tr], scale=KAPPA_SOLID)
    else:
        std = Standardizer(Y[tr])
    std.mean = std.mean.to(DEVICE); std.std = std.std.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    history = {"epoch": [], "train_loss": [], "val_loss": []}      # <<< ADDED
    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0
    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()                                                                          # this is the training part of the code
        tloss, tnb = 0.0, 0 
        for xb, yb in batches(X, Y, tr, BATCH_SIZE, shuffle=True):  # shuffle is set to false because we want to keep the order of the data consistent across epochs
            opt.zero_grad()
            loss = loss_fn(model(xb), std.encode(yb))
            loss.backward(); opt.step()
            tloss += loss.item(); tnb += 1                         # <<< ADDED
        tloss /= max(tnb, 1)

        model.eval(); vloss, nb = 0.0, 0
        with torch.no_grad():
            for xb, yb in batches(X, Y, va, BATCH_SIZE, shuffle=True):
                vloss += loss_fn(model(xb), std.encode(yb)).item(); nb += 1
        vloss /= max(nb, 1)

        history["epoch"].append(epoch)                            # <<< ADDED
        history["train_loss"].append(tloss)                       # <<< ADDED
        history["val_loss"].append(vloss)                         # <<< ADDED


        if vloss < best_val - 1e-6:
            best_val, best_epoch, bad = vloss, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 10 == 0 or bad == 0:
            print(f"  epoch {epoch:3d}  train_mse(scaled)={tloss:.4f}  "   # <<< CHANGED
                  f"val_mse(scaled)={vloss:.4f}  best={best_val:.4f}")
        if bad >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:   # <<< ADDED block
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])
        for e, tl, vl in zip(history["epoch"], history["train_loss"], history["val_loss"]):
            w.writerow([e, f"{tl:.6g}", f"{vl:.6g}"])

    model_path = os.path.join(run_dir, "surrogate_model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "y_mean": std.mean.cpu(),
        "y_std":  std.std.cpu(),
        "normalizer": "scalar" if USE_SCALAR else "standard",
    }, model_path)
    print(f"saved -> {model_path}")

    summary = {
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "best_val_mse_scaled": best_val,
        "early_stopped": bad >= PATIENCE,
        "normalizer_mean": std.mean.cpu(),
        "normalizer_std": std.std.cpu(),
    }
    return model, std, summary


# EVALUATION

def evaluate(model, std, X, Y, te, run_dir):
    model.eval(); preds = []
    with torch.no_grad():
        for xb, _ in batches(X, Y, te, BATCH_SIZE, shuffle=False):
            preds.append(std.decode(model(xb)).cpu().numpy())
    pred = np.concatenate(preds)
    true = Y[te]

    metrics = {}
    print("\nCNN test performance (physical units):")
    for i, n in enumerate(LABEL_NAMES):
        t, p = true[:, i], pred[:, i]
        err = p - t
        mae  = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err**2))
        r2   = 1 - np.sum(err**2) / (np.sum((t - t.mean())**2) + 1e-12)   # 1e-12 prevents division by zero
        metrics[n] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"  {n}: MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.3f}")

    with open(os.path.join(run_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index"] + [f"true_{n}" for n in LABEL_NAMES]
                   + [f"pred_{n}" for n in LABEL_NAMES])
        for k, idx in enumerate(te):
            w.writerow([int(idx)] + [f"{v:.6g}" for v in true[k]] + [f"{v:.6g}" for v in pred[k]])

    return pred, true, metrics


def data_quality_report(Y, T):
    """How close did the optimizer get achieved kappa to the requested target?"""
    mask = ~np.isnan(T[:, 0])
    if mask.sum() == 0:
        print("  no target info parsed; skipping.")
        return {}
    out = {}
    for i, n in [(0, "kxx"), (2, "kyy")]:
        gap = np.abs(Y[mask, i] - T[mask, i])
        out[n] = {"mean_abs_gap": gap.mean(), "max_abs_gap": gap.max()}
        print(f"  target vs achieved {n}: mean|gap|={gap.mean():.4f}  max|gap|={gap.max():.4f}")
    return out


def config_dict():
    return {
        "DENSITY_VAR": DENSITY_VAR, "KAPPA_ATTR": KAPPA_ATTR,
        "normalizer": "ScalarNormalizer" if USE_SCALAR else "Standardizer",
        "KAPPA_SOLID": KAPPA_SOLID if USE_SCALAR else None,
        "DROP_NONCONVERGED": DROP_NONCONVERGED,
        "VAL_FRAC": VAL_FRAC, "TEST_FRAC": TEST_FRAC,
        "BATCH_SIZE": BATCH_SIZE, "LR": LR, "optimizer": "Adam", "loss": "MSELoss",
        "MAX_EPOCHS": MAX_EPOCHS, "PATIENCE": PATIENCE,
        "SPLIT_SEED": SPLIT_SEED, "TRAIN_SEED": TRAIN_SEED,
        "train_shuffle": False,
    }


def model_dict(model):
    return {
        "class": type(model).__name__,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "n_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "architecture": str(model),
    }


def environment_dict():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "command": " ".join(sys.argv),
        "working_dir": os.getcwd(),
        "script": os.path.abspath(__file__),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", default=DEFAULT_GLOB)
    ap.add_argument("--tag", default=None, help="short label added to the run folder name")
    ap.add_argument("--notes", default="", help="free-text note stored in run.json")
    args = ap.parse_args()

    run_dir = make_run_dir(args.tag)
    sys.stdout = Tee(os.path.join(run_dir, "log.txt"))
    started = datetime.now()

    torch.manual_seed(TRAIN_SEED); np.random.seed(TRAIN_SEED)

    overall_start = time.perf_counter()
    print(f"Run folder: {run_dir}")
    print(f"Using device: {DEVICE}")
    print("Loading data.......")
    X, Y, T, data_info = load_dataset(args.data_glob)
    label_stats = label_spread_report(Y)
    print("\nData quality (optimizer target vs achieved):")
    quality = data_quality_report(Y, T)
    tr, va, te = split_indices(len(X))
    print(f"\nSplit: train={len(tr)} val={len(va)} test={len(te)}  device={DEVICE}")

    print("\nTraining CNN...")
    train_start = time.perf_counter()
    model, std, train_summary = train(X, Y, tr, va, run_dir)
    train_summary["train_time_s"] = time.perf_counter() - train_start
    _, _, metrics = evaluate(model, std, X, Y, te, run_dir)
    overall_time = time.perf_counter() - overall_start
    print(f"\nOverall time: {overall_time:.2f} seconds or ({overall_time/60:.2f} minutes)")

    dataset_summary = dict(data_info)
    dataset_summary.update({
        "label_stats": label_stats,
        "target_vs_achieved": quality,
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "test_indices": te,
    })

    save_json(os.path.join(run_dir, "run.json"), {
        "run_dir": run_dir,
        "notes": args.notes,
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "total_time_s": overall_time,
        "config": config_dict(),
        "dataset": dataset_summary,
        "model": model_dict(model),
        "training": train_summary,
        "test_metrics": metrics,
        "environment": environment_dict(),
    })
    print(f"Run saved to {run_dir}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()