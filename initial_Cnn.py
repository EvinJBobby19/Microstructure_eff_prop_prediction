"""
Usage:
  python surrogate_cnn.py --data-glob "test_dataset_200/**/result2D_*.nc"
"""

import argparse
import glob
import re
import numpy as np
import torch
import torch.nn as nn
from netCDF4 import Dataset
import time

# CONFIG
DENSITY_VAR   = "density"     
KAPPA_ATTR    = "kappa_eff"   # achieved kappa attr; switch to "kappa_thresh" for binarized

DROP_NONCONVERGED = True      

# DEFAULT_GLOB  = "test_dataset_200/**/result2D_*.nc"  # <<< CHANGED: point at ONE resolution tree

DEFAULT_GLOB  = "//wsl.localhost/Ubuntu/home/evin/muTopOpt/test_dataset_200/**/result2D_*.nc"  # <<< CHANGED (was **/*.nc)

VAL_FRAC      = 0.15
TEST_FRAC     = 0.15
BATCH_SIZE    = 32
LR            = 1e-3
MAX_EPOCHS    = 300
PATIENCE      = 25
SPLIT_SEED    = 0
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


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
    return X, Y, T


def label_spread_report(Y):
    """Diversity check. With random diagonal targets this should look healthy."""
    names = ["kxx", "kxy", "kyy"]
    print("\nLabel spread (achieved kappa):")
    for i, n in enumerate(names):
        col = Y[:, i]
        print(f"  {n}: mean={col.mean():.4f}  std={col.std():.4f}  "
              f"min={col.min():.4f}  max={col.max():.4f}")
    if Y[:, 0].std() < 0.02:
        print("  !! kxx std is tiny -> labels barely vary; check the generator.")
    print("  (kxy std ~0 is expected: targets are diagonal, so off-diagonal ~0.)")


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


# THE CNN

class ConvBlock(nn.Module):
    """Part 2: conv (circular padding for periodic BCs) -> BN -> ReLU -> downsample."""
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
        self.pool = nn.AdaptiveAvgPool2d(1)          # Part 3 (resolution-agnostic)
        head_in = 128   # Part 4
        self.head = nn.Sequential(                   # Part 5 -> Part 6
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


def train(X, Y, tr, va):
    model = MicrostructureCNN().to(DEVICE)
    std = Standardizer(Y[tr]); std.mean = std.mean.to(DEVICE); std.std = std.std.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val, best_state, bad = float("inf"), None, 0                   # these variable are made use for early stopping and saving the best model state
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()                                                                          # this is the training part of the code 
        for xb, yb in batches(X, Y, tr, BATCH_SIZE, shuffle=False):  # shuffle is set to false because we want to keep the order of the data consistent across epochs
            opt.zero_grad()
            loss = loss_fn(model(xb), std.encode(yb))
            loss.backward(); opt.step()

        model.eval(); vloss, nb = 0.0, 0
        with torch.no_grad():
            for xb, yb in batches(X, Y, va, BATCH_SIZE, shuffle=False):
                vloss += loss_fn(model(xb), std.encode(yb)).item(); nb += 1
        vloss /= max(nb, 1)

        if vloss < best_val - 1e-6:
            best_val, best_state, bad = vloss, {k: v.cpu().clone()
                                                for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if epoch % 10 == 0 or bad == 0:
            print(f"  epoch {epoch:3d}  val_mse(std)={vloss:.4f}  best={best_val:.4f}")
        if bad >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)


    torch.save({                                        
        "state_dict": model.state_dict(),
        "y_mean": std.mean.cpu(),
        "y_std":  std.std.cpu(),
    }, "surrogate_model.pt")
    print("saved -> surrogate_model.pt")  # added


    return model, std


# EVALUATION  --  MAE 
def evaluate(model, std, X, Y, te):
    model.eval(); preds = []
    with torch.no_grad():
        for xb, _ in batches(X, Y, te, BATCH_SIZE, shuffle=False):
            preds.append(std.decode(model(xb)).cpu().numpy())
    pred = np.concatenate(preds)
    true = Y[te]
    print("\nCNN test performance (physical units):")
    for i, n in enumerate(["kxx", "kxy", "kyy"]):
        mae = np.mean(np.abs(pred[:, i] - true[:, i]))
        r2 = 1 - np.sum((pred[:, i] - true[:, i])**2) / (
            np.sum((true[:, i] - true[:, i].mean())**2) + 1e-12)    #### 1e-12 very small number added to prevent the dicison by zero error
        err = pred[:, i] - true[:, i]
        rmse = np.sqrt(np.mean(err**2))
        print(f"  {n}: MAE={mae:.4f}  R2={r2:.3f} RMSE={rmse:.4f} ")
    return pred, true


def data_quality_report(Y, T):
    """How close did the optimizer get achieved kappa to the requested target?
    A model-independent sanity check on the data. NaNs (no command_line) skipped."""
    mask = ~np.isnan(T[:, 0])
    if mask.sum() == 0:
        print("  no target info parsed; skipping.")
        return
    for i, n in [(0, "kxx"), (2, "kyy")]:
        gap = np.abs(Y[mask, i] - T[mask, i])
        print(f"  target vs achieved {n}: mean|gap|={gap.mean():.4f}  max|gap|={gap.max():.4f}")





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", default=DEFAULT_GLOB)
    args = ap.parse_args()
    overall_start = time.perf_counter()
    print(f"Using device: {DEVICE}")
    print(f"Loading data.......")
    X, Y, T = load_dataset(args.data_glob)
    label_spread_report(Y)
    print("\nData quality (optimizer target vs achieved):")
    data_quality_report(Y, T)
    tr, va, te = split_indices(len(X))
    print(f"\nSplit: train={len(tr)} val={len(va)} test={len(te)}  device={DEVICE}")

    print("\nTraining CNN...")
    model, std = train(X, Y, tr, va)
    evaluate(model, std, X, Y, te)
    overall_time = time.perf_counter() - overall_start
    print(f"\nOverall time: {overall_time:.2f} seconds or ({overall_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()