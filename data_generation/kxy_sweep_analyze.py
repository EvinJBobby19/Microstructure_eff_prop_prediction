"""
Analyse a kxy_sweep.py output folder (kxx, kyy, kxy all swept).

Run AFTER the sweep and after nc_to_images_updated.py:

    python nc_to_images_updated.py kxy_sweep/res_64_seed1000  kxy_images
    python kxy_sweep_analyze.py    kxy_sweep/res_64_seed1000  kxy_images  kxy_analysis

Writes into <out_dir>:
  summary.csv               one row per run: target, achieved (grey + thresholded),
                            errors, vf, greyness, converged
  achieved_kxy_grey.png     one panel per kyy; in each, achieved kxy vs target kxy,
  achieved_kxy_thresh.png   one line per kxx                                      # <<< CHANGED
  max_kxy_grey.png          heatmap over (kxx, kyy): largest kxy actually achieved
  max_kxy_thresh.png        (grey / thresholded structure)                         # <<< CHANGED
  grids/grid_kyy0.50.png    microstructure tiles per kyy: kxx on x, kxy on y      # <<< CHANGED
  grids/grid_kyy0.50_thresh.png
"""

import csv
import glob
import math
import os
import re
import sys

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

NAME_RE = re.compile(r"result2D_kxx([\d.]+)_kyy([\d.]+)_kxy([\d.]+)\.nc$")      # <<< CHANGED
GREY_LO, GREY_HI = 0.1, 0.9


def load_run(path):
    m = NAME_RE.search(os.path.basename(path))
    kxx_t, kyy_t, kxy_t = (float(g) for g in m.groups())                       # <<< CHANGED
    ds = nc.Dataset(path)
    d = ds.variables["density"]
    rho = np.array(d[-1]) if d.ndim == 3 else np.array(d[:])
    attrs = ds.ncattrs()
    k = np.array(ds.getncattr("kappa_eff"), float).reshape(2, 2) if "kappa_eff" in attrs else None
    kt = np.array(ds.getncattr("kappa_thresh"), float).reshape(2, 2) if "kappa_thresh" in attrs else None
    conv = bool(np.ravel(ds.getncattr("converged"))[0]) if "converged" in attrs else None
    iters = int(np.ravel(ds.getncattr("nb_iterations"))[0]) if "nb_iterations" in attrs else None
    ds.close()
    return dict(kxx_t=kxx_t, kxy_t=kxy_t, kyy_t=kyy_t, k=k, kt=kt,
                converged=conv, iters=iters, vf=float(rho.mean()),
                greyness=float(np.mean((rho > GREY_LO) & (rho < GREY_HI))))


def row(r):
    T = np.array([[r["kxx_t"], r["kxy_t"]], [r["kxy_t"], r["kyy_t"]]])
    out = dict(target_kxx=r["kxx_t"], target_kyy=r["kyy_t"], target_kxy=r["kxy_t"],
               converged=r["converged"], iters=r["iters"],
               vf=round(r["vf"], 4), greyness=round(r["greyness"], 4))
    for tag, K in (("grey", r["k"]), ("thresh", r["kt"])):
        if K is None:
            continue
        out.update({
            f"{tag}_kxx": round(K[0, 0], 4), f"{tag}_kxy": round(K[0, 1], 4),
            f"{tag}_kyx": round(K[1, 0], 4), f"{tag}_kyy": round(K[1, 1], 4),
            f"{tag}_kxy_err": round(K[0, 1] - r["kxy_t"], 4),
            f"{tag}_mse": float(np.mean((K - T) ** 2)),
        })
    return out


def line_panels(rows, tag, path):                                               # <<< CHANGED: one panel per kyy
    kyys = sorted({r["target_kyy"] for r in rows})
    kxxs = sorted({r["target_kxx"] for r in rows})
    ncol = 5
    nrow = math.ceil(len(kyys) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    cmap = plt.get_cmap("viridis")
    lim = max(r["target_kxy"] for r in rows)
    for idx, kyy in enumerate(kyys):
        ax = axes[idx // ncol][idx % ncol]
        for i, kxx in enumerate(kxxs):
            rs = sorted([r for r in rows if r["target_kyy"] == kyy and r["target_kxx"] == kxx
                         and f"{tag}_kxy" in r], key=lambda r: r["target_kxy"])
            if rs:
                ax.plot([r["target_kxy"] for r in rs], [r[f"{tag}_kxy"] for r in rs],
                        "-o", ms=2, lw=1, color=cmap(i / max(len(kxxs) - 1, 1)),
                        label=f"kxx={kxx:.2f}")
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        ax.set_title(f"kyy={kyy:.2f}", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(kyys):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("target kxy", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"achieved kxy ({tag})", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7)
    fig.suptitle(f"Achieved vs target kxy [{tag}]  (dashed: achieved = target)")
    fig.tight_layout(rect=(0, 0, 0.92, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def max_kxy_map(rows, tag, path):                                               # <<< CHANGED: eigenvalue panel removed
    """Largest |kxy| achieved anywhere in each (kxx, kyy) cell."""
    kxxs = sorted({r["target_kxx"] for r in rows})
    kyys = sorted({r["target_kyy"] for r in rows})
    ach = np.full((len(kyys), len(kxxs)), np.nan)
    for i, kyy in enumerate(kyys):
        for j, kxx in enumerate(kxxs):
            vals = [abs(r[f"{tag}_kxy"]) for r in rows
                    if r["target_kxx"] == kxx and r["target_kyy"] == kyy and f"{tag}_kxy" in r]
            if vals:
                ach[i, j] = max(vals)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(ach, origin="lower", cmap="viridis", vmin=0)
    ax.set_xticks(range(len(kxxs)), [f"{v:.2f}" for v in kxxs], rotation=90, fontsize=7)
    ax.set_yticks(range(len(kyys)), [f"{v:.2f}" for v in kyys], fontsize=7)
    ax.set_xlabel("kxx")
    ax.set_ylabel("kyy")
    ax.set_title(f"max achieved |kxy| [{tag}]", fontsize=9)
    for i in range(ach.shape[0]):
        for j in range(ach.shape[1]):
            if not np.isnan(ach[i, j]):
                ax.text(j, i, f"{ach[i, j]:.2f}", ha="center", va="center", fontsize=5, color="w")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tile_grid(img_dir, rows, kyy, suffix, path):
    sub = [r for r in rows if r["target_kyy"] == kyy]                           # <<< CHANGED: one grid per kyy
    kxxs = sorted({r["target_kxx"] for r in sub})
    kxys = sorted({r["target_kxy"] for r in sub})
    fig, axes = plt.subplots(len(kxys), len(kxxs),
                             figsize=(1.6 * len(kxxs), 2.0 * len(kxys)), squeeze=False)
    for i, kxy in enumerate(reversed(kxys)):          # kxy increases upwards
        for j, kxx in enumerate(kxxs):
            ax = axes[i][j]
            ax.axis("off")
            p = os.path.join(img_dir,
                             f"result2D_kxx{kxx:.2f}_kyy{kyy:.2f}_kxy{kxy:.2f}{suffix}.png")
            if os.path.exists(p):
                ax.imshow(plt.imread(p))
            if i == len(kxys) - 1:
                ax.set_title(f"kxx={kxx:.2f}", fontsize=7, y=-0.15)
            if j == 0:
                ax.text(-0.1, 0.5, f"kxy={kxy:.2f}", transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=7)
    fig.suptitle(f"kyy = {kyy:.2f} {'[thresholded]' if suffix else '[grey]'}", fontsize=12)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    if len(sys.argv) != 4:
        print("Usage: python kxy_sweep_analyze.py <nc_folder> <image_folder> <out_dir>")
        sys.exit(1)
    nc_dir, img_dir, out_dir = sys.argv[1:]
    grid_dir = os.path.join(out_dir, "grids")
    os.makedirs(grid_dir, exist_ok=True)

    files = sorted(f for f in glob.glob(os.path.join(nc_dir, "*.nc")) if NAME_RE.search(f))
    rows = []
    for f in files:
        try:
            rows.append(row(load_run(f)))
        except Exception as e:
            print(f"skip {f}: {e}")
    print(f"{len(rows)} runs loaded")
    if not rows:
        sys.exit(1)

    keys = sorted({k for r in rows for k in r}, key=lambda k: (not k.startswith("target"), k))
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    for tag in ("grey", "thresh"):
        line_panels(rows, tag, os.path.join(out_dir, f"achieved_kxy_{tag}.png"))
        max_kxy_map(rows, tag, os.path.join(out_dir, f"max_kxy_{tag}.png"))

    for kyy in sorted({r["target_kyy"] for r in rows}):
        tile_grid(img_dir, rows, kyy, "", os.path.join(grid_dir, f"grid_kyy{kyy:.2f}.png"))
        tile_grid(img_dir, rows, kyy, "_thresh",
                  os.path.join(grid_dir, f"grid_kyy{kyy:.2f}_thresh.png"))
    print(f"wrote summary.csv, line panels, max-kxy maps and per-kyy grids -> {out_dir}")


if __name__ == "__main__":
    main()