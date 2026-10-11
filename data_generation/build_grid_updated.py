"""
Assembles the annotated PNGs produced by nc_to_images.py into 2D
contact-sheet charts:

    - x-axis (across the TOP) = seed values
    - y-axis (down the LEFT)  = reg-weight values
    - each cell               = the annotated PNG for that (seed, reg_weight)

nc_to_images.py writes TWO PNGs per .nc file:                                  # <<< CHANGED
    result2D_seed0001_rw2.3000.png          grey (continuous) density
    result2D_seed0001_rw2.3000_thresh.png   thresholded (rho >= 0.5) density

This script builds a separate grid for each kind so the two can be compared
with the same seed / reg-weight layout. Each tile keeps the per-image title
annotations baked in by nc_to_images.py; this script only reads the seed /
reg-weight back out of the FILENAME to place each tile -- it does not
re-render or re-annotate anything.

Usage:
    python build_grid.py <png_folder> <output_chart.png> [mode]

    mode = both   (default) writes <output_chart>.png        grey grid
                                   <output_chart>_thresh.png thresholded grid
    mode = grey   only the grey grid
    mode = thresh only the thresholded grid
    mode = pair   one grid where every cell shows grey | thresholded side by side
"""

import sys
import glob
import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


SEED_RE = re.compile(r"seed(\d+)")
RW_RE = re.compile(r"rw([0-9]+\.?[0-9]*)")
THRESH_SUFFIX = "_thresh"                                                    # <<< ADDED


def parse_seed_rw(path):
    """Pull (seed:int, reg_weight:float) out of a filename, or (None, None)."""
    name = os.path.basename(path)
    s = SEED_RE.search(name)
    r = RW_RE.search(name)
    if not s or not r:
        return None, None
    return int(s.group(1)), float(r.group(1))


def is_thresh(path):                                                         # <<< ADDED
    return os.path.splitext(os.path.basename(path))[0].endswith(THRESH_SUFFIX)


def collect(png_folder):                                                     # <<< ADDED
    """Return (grey_cells, thresh_cells, seeds, rws).

    Each *_cells is a dict (seed, rw) -> filepath. The axis value sets are the
    union over both kinds so the two grids always share the same layout.
    """
    files = sorted(glob.glob(os.path.join(png_folder, "*.png")))
    if not files:
        print(f"No PNGs found in {png_folder}")
        sys.exit(1)

    grey, thresh = {}, {}
    seeds, rws = set(), set()
    skipped = []
    for f in files:
        seed, rw = parse_seed_rw(f)
        if seed is None:
            skipped.append(os.path.basename(f))
            continue
        (thresh if is_thresh(f) else grey)[(seed, rw)] = f
        seeds.add(seed)
        rws.add(rw)

    if skipped:
        print(f"Skipped {len(skipped)} file(s) without seed/rw in name: {skipped[:5]}"
              + (" ..." if len(skipped) > 5 else ""))

    seeds = sorted(seeds)            # x-axis (top), left -> right
    rws = sorted(rws)                # y-axis (left), top -> bottom
    print(f"Grid: {len(rws)} reg-weights x {len(seeds)} seeds  "
          f"(grey tiles: {len(grey)}, thresholded tiles: {len(thresh)})")
    return grey, thresh, seeds, rws


def _place_tile(ax, path):                                                   # <<< ADDED
    ax.set_xticks([])
    ax.set_yticks([])
    if path is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center",
                fontsize=9, color="red", transform=ax.transAxes)
    else:
        ax.imshow(mpimg.imread(path))


def _label_axes(ax, i, j, rw, seed):                                         # <<< ADDED
    # y-axis label (reg-weight) on the leftmost column only
    if j == 0:
        ax.set_ylabel(f"rw={rw:g}", fontsize=11, rotation=90,
                      labelpad=8, fontweight="bold")
    # x-axis label (seed) on the TOP row only
    if i == 0:
        ax.set_title(f"seed={seed}", fontsize=11, fontweight="bold", pad=6)


def build_grid(cells, seeds, rws, out_path, label):
    """One grid from one kind of tile (grey or thresholded)."""            # <<< CHANGED
    n_cols, n_rows = len(seeds), len(rws)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )
    for i, rw in enumerate(rws):          # rows
        for j, seed in enumerate(seeds):  # cols
            ax = axes[i][j]
            _place_tile(ax, cells.get((seed, rw)))
            _label_axes(ax, i, j, rw, seed)

    fig.suptitle(label, fontsize=14, fontweight="bold", y=1.0)               # <<< ADDED
    fig.supxlabel("seed  ->", fontsize=13, fontweight="bold", y=0.985)
    fig.supylabel("reg-weight  ->", fontsize=13, fontweight="bold", x=0.005)
    fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.975])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def build_pair_grid(grey, thresh, seeds, rws, out_path):                     # <<< ADDED
    """One grid where each cell is [grey | thresholded] for the same run."""
    n_cols, n_rows = len(seeds), len(rws)
    fig, axes = plt.subplots(
        n_rows, 2 * n_cols,
        figsize=(3.2 * 2 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )
    for i, rw in enumerate(rws):
        for j, seed in enumerate(seeds):
            ax_g, ax_t = axes[i][2 * j], axes[i][2 * j + 1]
            _place_tile(ax_g, grey.get((seed, rw)))
            _place_tile(ax_t, thresh.get((seed, rw)))
            _label_axes(ax_g, i, j, rw, seed)
            if i == 0:
                ax_g.set_title(f"seed={seed}  grey", fontsize=11,
                               fontweight="bold", pad=6)
                ax_t.set_title(f"seed={seed}  thresh", fontsize=11,
                               fontweight="bold", pad=6)

    fig.suptitle("grey | thresholded", fontsize=14, fontweight="bold", y=1.0)
    fig.supxlabel("seed  ->", fontsize=13, fontweight="bold", y=0.985)
    fig.supylabel("reg-weight  ->", fontsize=13, fontweight="bold", x=0.005)
    fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.975])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python build_grid.py <png_folder> <output_chart.png> [both|grey|thresh|pair]")
        sys.exit(1)

    png_folder, out_path = sys.argv[1], sys.argv[2]
    mode = sys.argv[3].lower() if len(sys.argv) == 4 else "both"           # <<< ADDED
    if mode not in ("both", "grey", "thresh", "pair"):
        print(f"Unknown mode '{mode}'. Use both, grey, thresh or pair.")
        sys.exit(1)

    grey, thresh, seeds, rws = collect(png_folder)
    root, ext = os.path.splitext(out_path)
    ext = ext or ".png"

    if mode in ("both", "grey"):
        build_grid(grey, seeds, rws, root + ext, "grey (continuous) density")
    if mode in ("both", "thresh"):
        if not thresh:
            print("No *_thresh.png tiles found -- run the updated nc_to_images.py first.")
        else:
            build_grid(thresh, seeds, rws, root + THRESH_SUFFIX + ext,
                       "thresholded density (rho >= 0.5)")
    if mode == "pair":
        build_pair_grid(grey, thresh, seeds, rws, root + "_pair" + ext)


if __name__ == "__main__":
    main()