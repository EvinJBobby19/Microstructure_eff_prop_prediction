
import argparse
import numpy as np
import torch
from netCDF4 import Dataset
from initial_Cnn import MicrostructureCNN

def load_sample(path, binary_input=True):
    with Dataset(path, "r") as ds:
        rho = np.asarray(ds.variables["density"][:], dtype=np.float32).squeeze()
        K = np.asarray(ds.getncattr("kappa_thresh"), dtype=float).reshape(2, 2)

    if rho.ndim != 2:
        raise ValueError(f"Expected 2D density, got {rho.shape}")

    x = (rho >= 0.5).astype(np.float32) if binary_input else rho
    true = np.array([K[0, 0], (K[0, 1] + K[1, 0]) / 2, K[1, 1]])
    return x, true

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to surrogate_model.pt")
    p.add_argument("--a", required=True, help="First .nc file")
    p.add_argument("--b", required=True, help="Second .nc file")
    p.add_argument("--continuous-input", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)

    model = MicrostructureCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    binary_input = not args.continuous_input
    xa, ya = load_sample(args.a, binary_input)
    xb, yb = load_sample(args.b, binary_input)

    X = torch.from_numpy(np.stack([xa, xb])[:, None]).to(device)

    mean = torch.as_tensor(checkpoint["y_mean"], device=device)
    scale = torch.as_tensor(checkpoint["y_std"], device=device)

    with torch.no_grad():
        pred = (model(X) * scale + mean).cpu().numpy()

    truth = np.stack([ya, yb])
    names = ["kxx", "kxy", "kyy"]

    print(f"\nInput: {'binary' if binary_input else 'continuous'}")
    for j, name in enumerate(["Sample A", "Sample B"]):
        print(f"\n{name}")
        for i, comp in enumerate(names):
            error = pred[j, i] - truth[j, i]
            print(f"  {comp}: true={truth[j,i]:.6f} "
                  f"pred={pred[j,i]:.6f} error={error:+.6f}")

    print("\nDifference between the two samples:")
    for i, comp in enumerate(names):
        print(f"  {comp}: true difference={abs(truth[0,i]-truth[1,i]):.6f} "
              f"predicted difference={abs(pred[0,i]-pred[1,i]):.6f}")

    changed = np.count_nonzero(xa != xb)
    print(f"\nDifferent input pixels: {changed}/{xa.size}")
    print(f"Percentage different: {100*changed/xa.size:.2f}%")

if __name__ == "__main__":
    main()
