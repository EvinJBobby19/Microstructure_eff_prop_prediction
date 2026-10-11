"""
Converts a folder of .nc simulation outputs into annotated PNG images
for easy visual navigation.

For every <name>.nc two images are written:
    <name>.png         grey (continuous) density field as optimized
    <name>_thresh.png  binarized field (rho >= 0.5 -> 1, else 0)

Usage:
    python nc_to_images.py /path/to/generated_datasets /path/to/output_images
"""

import sys
import glob
import os
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt


THRESHOLD = 0.5          # rho >= THRESHOLD -> solid                            # <<< ADDED
GREY_LO, GREY_HI = 0.1, 0.9   # "intermediate phase" band for the greyness metric  # <<< ADDED


def extract_target_kappa(command_line):
    """Pulls the --target-kappa values out of the stored command_line string."""
    if not command_line or "--target-kappa" not in command_line:
        return None
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
    if len(values) == 4:
        return values
    if len(values) == 1:                                                       # <<< ADDED: isotropic shorthand
        k = values[0]
        return [k, 0.0, 0.0, k]
    return None


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

    # Always take the LAST frame: files written with --dump-every have several.  # <<< CHANGED
    density_var = ds.variables["density"]
    density = np.array(density_var[-1]) if density_var.ndim == 3 else np.array(density_var[:])
    density = density.squeeze().T                                              # [x, y] -> [y, x]

    def get_attr(name, default="?"):
        return ds.getncattr(name) if name in ds.ncattrs() else default

    def getConvergedValue(value):
        return "True" if value else "False"

    converged = getConvergedValue(get_attr("converged"))
    nb_iterations = get_attr("nb_iterations")
    final_objective = get_attr("final_objective", None)
    command_line = get_attr("command_line", "")
    gtol = extract_cli_float(command_line, "--bfgs-gtol", 2.5)
    reg_weight = extract_cli_float(command_line, "--reg-weight", None)          # <<< ADDED
    target = extract_target_kappa(command_line)

    # --- grey (continuous) field metrics ---
    kappa_eff = get_attr("kappa_eff", None)
    mse_cont = get_attr("mse_continuous", None)
    if mse_cont is None:
        mse_cont = kappa_mse(target, kappa_eff)
    greyness = float(np.mean((density > GREY_LO) & (density < GREY_HI)))       # <<< ADDED
    vf_grey = float(density.mean())                                            # <<< ADDED

    # --- thresholded field metrics ---                                        # <<< ADDED
    density_bin = (density >= THRESHOLD).astype(float)
    kappa_thresh = get_attr("kappa_thresh", None)   # written by simulate_conduction.py
    mse_thresh = kappa_mse(target, kappa_thresh)
    vf_bin = float(density_bin.mean())
    gap = None if (mse_cont is None or mse_thresh is None) else mse_thresh - mse_cont

    ds.close()

    fname = os.path.splitext(os.path.basename(nc_path))[0]
    rw_str = "?" if reg_weight is None else f"{reg_weight:g}"
    common = (
        f"converged={converged}  iters={nb_iterations}  gtol={gtol:g}  rw={rw_str}  "
        f"final_obj={fmt_scalar(final_objective)}"
    )

    def save(field, suffix, title):                                            # <<< ADDED
        fig, ax = plt.subplots(figsize=(4, 5))
        ax.imshow(field, cmap="viridis", vmin=0, vmax=1, origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title, fontsize=7)
        p = os.path.join(out_dir, f"{fname}{suffix}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return p

    out_grey = save(density, "", (
        f"{fname}  [grey]\n"
        f"target={fmt_tensor(target)}\n"
        f"achieved kappa_eff={fmt_tensor(kappa_eff)}\n"
        f"mse={fmt_scalar(mse_cont)}  greyness={greyness:.3f}  vf={vf_grey:.3f}\n"
        f"{common}"
    ))

    out_bin = save(density_bin, "_thresh", (
        f"{fname}  [thresholded, rho>={THRESHOLD}]\n"
        f"target={fmt_tensor(target)}\n"
        f"achieved kappa_eff={fmt_tensor(kappa_eff)}\n"
        f"kappa_thresh={fmt_tensor(kappa_thresh)}\n"
        f"mse_thresh={fmt_scalar(mse_thresh)}  gap={fmt_scalar(gap)}  vf={vf_bin:.3f}\n"
        f"{common}"
    ))

    return out_grey, out_bin


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
            out_grey, out_bin = render_one(f, out_dir)                          # <<< CHANGED
            print(f"[{i+1}/{len(files)}] {out_grey}  +  {os.path.basename(out_bin)}")
        except Exception as e:
            print(f"[{i+1}/{len(files)}] FAILED on {f}: {e}")


if __name__ == "__main__":
    main()