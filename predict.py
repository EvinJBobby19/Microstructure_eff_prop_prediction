"""
predict.py -- load the trained surrogate and predict kappa for a density field.

Usage:
  python predict.py --nc path/to/some.nc          # predict from a .nc file
  python predict.py --demo                         # predict on a synthetic field

The model file surrogate_model.pt is produced by training surrogate_cnn.py.
"""

import argparse
import numpy as np
import torch
from netCDF4 import Dataset

# reuse the exact same building blocks used at training time
from surrogate_cnn import MicrostructureCNN, _read_density, DEVICE

MODEL_PATH = "surrogate_model.pt"


def load_model(path=MODEL_PATH):
    """Rebuild the network and restore weights + the target scaler."""
    ckpt = torch.load(path, map_location=DEVICE)
    model = MicrostructureCNN(use_vf_scalar=ckpt["use_vf_scalar"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()                                  # inference mode (no dropout/BN updates)
    y_mean = ckpt["y_mean"].to(DEVICE)
    y_std  = ckpt["y_std"].to(DEVICE)
    return model, y_mean, y_std


def predict(model, y_mean, y_std, rho):
    """rho: a 2D (H,W) density array -> dict of predicted kappa components."""
    rho = np.asarray(rho, dtype=np.float32)
    if rho.ndim != 2:
        raise ValueError(f"expected a 2D density field, got shape {rho.shape}")
    # shape it the way the net expects: (batch=1, channel=1, H, W)
    x = torch.from_numpy(rho[None, None, :, :]).to(DEVICE)
    with torch.no_grad():
        out_std = model(x)                        # standardized output
        out = out_std * y_std + y_mean            # decode back to physical kappa
    kxx, kxy, kyy = out.cpu().numpy().ravel()
    return {"kxx": float(kxx), "kxy": float(kxy), "kyy": float(kyy)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", help="a .nc file to read the density field from")
    ap.add_argument("--demo", action="store_true",
                    help="predict on a synthetic 64x64 field instead")
    args = ap.parse_args()

    model, y_mean, y_std = load_model()

    if args.nc:
        ds = Dataset(args.nc, "r")
        rho = _read_density(ds)                   # same reader as training (squeezes frame)
        ds.close()
    elif args.demo:
        # a made-up field: a solid square in a void, ~50% volume fraction
        rho = np.zeros((64, 64), dtype=np.float32)
        rho[16:48, 16:48] = 1.0
    else:
        raise SystemExit("pass --nc <file> or --demo")

    pred = predict(model, y_mean, y_std, rho)
    print(f"volume fraction of input: {rho.mean():.4f}")
    print(f"predicted kappa: kxx={pred['kxx']:.4f}  "
          f"kxy={pred['kxy']:.4f}  kyy={pred['kyy']:.4f}")


if __name__ == "__main__":
    main()
