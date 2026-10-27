"""
Converts ALL batch folders of .nc simulation outputs into annotated PNG images,
both grey (continuous) and thresholded.

Expected input layout (as written by batch_parallelization_withtxt.py):
    <dataset_root>/batch_00_seed1000/result2D_000.nc
    <dataset_root>/batch_00_seed1000/result2D_001.nc
    <dataset_root>/batch_01_seed1001/...

Output layout (everything in ONE folder, named by batch + result number):
    <output_root>/batch00_result000.png          grey
    <output_root>/batch00_result000_thresh.png   thresholded

Usage:
    python batches_to_images.py <dataset_root> <output_root> [--workers N] [--skip-existing]
"""

import argparse                                                                # <<< ADDED
import glob
import os
import re                                                                      # <<< ADDED
from concurrent.futures import ProcessPoolExecutor, as_completed                # <<< ADDED

import matplotlib                                                              # <<< ADDED
matplotlib.use("Agg")                                                          # <<< ADDED: no display needed (cluster-safe)
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np


THRESHOLD = 0.5          # rho >= THRESHOLD -> solid
GREY_LO, GREY_HI = 0.1, 0.9   # "intermediate phase" band for the greyness metric


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
    if len(values) == 1:                                                       # isotropic shorthand
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


def image_name(nc_path):                                                       # <<< ADDED
    """batch_03_seed1003/result2D_017.nc -> 'batch03_result017'."""
    batch_dir = os.path.basename(os.path.dirname(os.path.abspath(nc_path)))
    b = re.search(r"batch_(\d+)", batch_dir)
    r = re.search(r"(\d+)$", os.path.splitext(os.path.basename(nc_path))[0])
    batch_str = f"batch{b.group(1)}" if b else batch_dir
    result_str = f"result{r.group(1)}" if r else os.path.splitext(os.path.basename(nc_path))[0]
    return f"{batch_str}_{result_str}"


def render_one(nc_path, out_dir, skip_existing=False):                         # <<< CHANGED: single output dir
    fname = image_name(nc_path)                                                # <<< CHANGED
    out_grey_path = os.path.join(out_dir, f"{fname}.png")                      # <<< CHANGED
    out_bin_path = os.path.join(out_dir, f"{fname}_thresh.png")                # <<< CHANGED
    if skip_existing and os.path.exists(out_grey_path) and os.path.exists(out_bin_path):  # <<< ADDED
        return out_grey_path, out_bin_path, True                               # <<< ADDED

    ds = nc.Dataset(nc_path)

    # Always take the LAST frame: files written with --dump-every have several.
    density_var = ds.variables["density"]
    density = np.array(density_var[-1]) if density_var.ndim == 3 else np.array(density_var[:])
    density = density.squeeze().T                                              # [x, y] -> [y, x]

    def get_attr(name, default="?"):
        return ds.getncattr(name) if name in ds.ncattrs() else default

    def getConvergedValue(value):
        if value is None:                                                      # <<< ADDED: missing attr no longer shows as True
            return "?"                                                         # <<< ADDED
        return "True" if value else "False"

    converged = getConvergedValue(get_attr("converged", None))                 # <<< CHANGED
    nb_iterations = get_attr("nb_iterations")
    final_objective = get_attr("final_objective", None)
    command_line = get_attr("command_line", "")
    gtol = extract_cli_float(command_line, "--bfgs-gtol", 2.5)
    reg_weight = extract_cli_float(command_line, "--reg-weight", None)
    target = extract_target_kappa(command_line)

    # --- grey (continuous) field metrics ---
    kappa_eff = get_attr("kappa_eff", None)
    mse_cont = get_attr("mse_continuous", None)
    if mse_cont is None:
        mse_cont = kappa_mse(target, kappa_eff)
    greyness = float(np.mean((density > GREY_LO) & (density < GREY_HI)))
    vf_grey = float(density.mean())

    # --- thresholded field metrics ---
    density_bin = (density >= THRESHOLD).astype(float)
    kappa_thresh = get_attr("kappa_thresh", None)   # written by simulate_conduction.py
    mse_thresh = kappa_mse(target, kappa_thresh)
    vf_bin = float(density_bin.mean())
    gap = None if (mse_cont is None or mse_thresh is None) else mse_thresh - mse_cont

    ds.close()

    rw_str = "?" if reg_weight is None else f"{reg_weight:g}"
    common = (
        f"converged={converged}  iters={nb_iterations}  gtol={gtol:g}  rw={rw_str}  "
        f"final_obj={fmt_scalar(final_objective)}"
    )

    def save(field, path, title):                                              # <<< CHANGED: takes full path
        fig, ax = plt.subplots(figsize=(4, 5))
        ax.imshow(field, cmap="viridis", vmin=0, vmax=1, origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title, fontsize=7)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    out_grey = save(density, out_grey_path, (                                  # <<< CHANGED
        f"{fname}  [grey]\n"
        f"target={fmt_tensor(target)}\n"
        f"achieved kappa_eff={fmt_tensor(kappa_eff)}\n"
        f"mse={fmt_scalar(mse_cont)}  greyness={greyness:.3f}  vf={vf_grey:.3f}\n"
        f"{common}"
    ))

    out_bin = save(density_bin, out_bin_path, (                                # <<< CHANGED
        f"{fname}  [thresholded, rho>={THRESHOLD}]\n"
        f"target={fmt_tensor(target)}\n"
        f"achieved kappa_eff={fmt_tensor(kappa_eff)}\n"
        f"kappa_thresh={fmt_tensor(kappa_thresh)}\n"
        f"mse_thresh={fmt_scalar(mse_thresh)}  gap={fmt_scalar(gap)}  vf={vf_bin:.3f}\n"
        f"{common}"
    ))

    return out_grey, out_bin, False                                            # <<< CHANGED


def find_batches(dataset_root):                                                # <<< ADDED
    """Every batch_* folder that contains .nc files; falls back to the root itself."""
    batches = sorted(d for d in glob.glob(os.path.join(dataset_root, "batch_*"))
                     if os.path.isdir(d))
    if not batches and glob.glob(os.path.join(dataset_root, "*.nc")):
        batches = [dataset_root]                                               # single flat folder also works
    return batches


def main():
    p = argparse.ArgumentParser(description=__doc__,                           # <<< CHANGED: argparse
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset_root", help="folder containing batch_* subfolders")
    p.add_argument("output_root", help="where the PNGs are written")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1,         # <<< ADDED
                   help="parallel processes (default: all cores)")
    p.add_argument("--skip-existing", action="store_true",                     # <<< ADDED
                   help="don't re-render files whose two PNGs already exist")
    args = p.parse_args()

    batches = find_batches(args.dataset_root)                                  # <<< ADDED
    if not batches:
        print(f"No batch_* folders or .nc files found in {args.dataset_root}")
        return

    # Build the full job list over all batches                                 # <<< ADDED
    os.makedirs(args.output_root, exist_ok=True)                               # <<< CHANGED: one folder
    jobs = []
    for batch_dir in batches:
        batch_name = os.path.basename(os.path.normpath(batch_dir))
        files = sorted(glob.glob(os.path.join(batch_dir, "*.nc")))
        print(f"{batch_name}: {len(files)} .nc files")
        jobs += files                                                          # <<< CHANGED

    total = len(jobs)
    print(f"Total: {total} files in {len(batches)} batch(es), {args.workers} worker(s)\n")

    done = failed = skipped = 0                                                # <<< ADDED
    with ProcessPoolExecutor(max_workers=args.workers) as pool:                 # <<< ADDED
        futures = {pool.submit(render_one, f, args.output_root, args.skip_existing): f for f in jobs}  # <<< CHANGED
        for fut in as_completed(futures):
            f = futures[fut]
            done += 1
            try:
                out_grey, out_bin, was_skipped = fut.result()
                if was_skipped:
                    skipped += 1
                    continue
                print(f"[{done}/{total}] {out_grey}  +  {os.path.basename(out_bin)}")
            except Exception as e:
                failed += 1
                print(f"[{done}/{total}] FAILED on {f}: {e}")

    print(f"\nFinished: {total - failed - skipped} rendered, "                  # <<< ADDED
          f"{skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()