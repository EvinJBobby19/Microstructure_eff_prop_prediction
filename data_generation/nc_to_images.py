"""
Converts a folder of .nc simulation outputs into annotated PNG images
for easy visual navigation.

Usage:
    python nc_to_images.py /path/to/generated_datasets /path/to/output_images
"""

import sys
import glob
import os
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt


def extract_target_kappa(command_line):
    """Pulls the --target-kappa values out of the stored command_line string."""
    if not command_line or "--target-kappa" not in command_line:
        return None                                                          # <<< CHANGED: None instead of "?" so it can be used numerically
    parts = command_line.split("--target-kappa")[1].strip().split()
    # target-kappa is followed by 4 numeric values (a 2x2 tensor, flattened)
    values = []
    for p in parts:
        try:
            values.append(float(p))
        except ValueError:
            break
        if len(values) == 4:
            break
    return values if len(values) == 4 else None                              # <<< CHANGED


def fmt_tensor(values):
    """Format a flat kappa tensor attribute for the title, or '?' if absent."""
    if values is None:
        return "?"
    arr = np.asarray(values, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{v:.4f}" for v in arr) + "]"


def fmt_scalar(value, digits=3):                                             
    """Format a scalar attribute in scientific notation, or 'n/a' if absent."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}e}"
    except (TypeError, ValueError):
        return str(value)


def kappa_mse(target, achieved):                                             
    """Mean squared difference over all 4 components of the 2x2 kappa tensor."""
    if target is None or achieved is None:
        return None
    t = np.asarray(target, dtype=float).reshape(-1)
    a = np.asarray(achieved, dtype=float).reshape(-1)
    if t.shape != a.shape:
        return None
    return float(np.mean((a - t) ** 2))

def extract_cli_float(command_line, flag, default):                          
    """Pulls a single float following `flag` from command_line, else default."""
    if not command_line or flag not in command_line:
        return default
    parts = command_line.split(flag)[1].strip().split()
    try:
        return float(parts[0])
    except (IndexError, ValueError):
        return default


def render_one(nc_path, out_dir):
    ds = nc.Dataset(nc_path)

    # only variable present in these files
    density = np.array(ds.variables["density"][:]).squeeze()
    density = density.T                                                     # [x, y] -> [y, x] for correct axis orientation

    def get_attr(name, default="?"):
        return ds.getncattr(name) if name in ds.ncattrs() else default

    def getConvergedValue(value):
        if value:
            return "True"
        else:
            return "False"

    converged = getConvergedValue(get_attr("converged"))
    nb_iterations = get_attr("nb_iterations")
    final_objective = get_attr("final_objective", None)                     # <<< CHANGED: None default for fmt_scalar
    command_line = get_attr("command_line", "")
    gtol = extract_cli_float(command_line, "--bfgs-gtol", 2.5)
    target = extract_target_kappa(command_line)

    kappa_eff = get_attr("kappa_eff", None)
    kappa_eff_str = fmt_tensor(kappa_eff)

    # --- clean quality metrics (regularization-free) ---                    # <<< ADDED
    # prefer values written by simulate_conduction.py; fall back to local computation
    mse_cont = get_attr("mse_continuous", None)                              # <<< ADDED
    if mse_cont is None:                                                     # <<< ADDED
        mse_cont = kappa_mse(target, kappa_eff)                              # <<< ADDED

    fname = os.path.splitext(os.path.basename(nc_path))[0]

    fig, ax = plt.subplots(figsize=(4, 4.8))                                 # <<< CHANGED: slightly taller for extra title line
    im = ax.imshow(density, cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    title = (
        f"{fname}\n"
        f"target={fmt_tensor(target)}\n"                                     # <<< CHANGED
        f"achieved kappa_eff={kappa_eff_str}\n"
        f"mse={fmt_scalar(mse_cont)}\n"
        f"converged={converged}  iters={nb_iterations}  gtol={gtol:g}  "            # <<< CHANGED
        f"final_obj={fmt_scalar(final_objective)}"                           # <<< CHANGED: demoted to same line, optimizer-internal
    )
    ax.set_title(title, fontsize=7)

    out_path = os.path.join(out_dir, f"{fname}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ds.close()
    return out_path


def main():
    if len(sys.argv) != 3:
        print("Usage: python nc_to_images.py <nc_folder> <output_folder>")
        sys.exit(1)

    nc_folder, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(nc_folder, "*.nc")))
    print(f"Found {len(files)} .nc files")

    for i, f in enumerate(files):
        try:
            out_path = render_one(f, out_dir)
            print(f"[{i+1}/{len(files)}] {out_path}")
        except Exception as e:
            print(f"[{i+1}/{len(files)}] FAILED on {f}: {e}")


if __name__ == "__main__":
    main()