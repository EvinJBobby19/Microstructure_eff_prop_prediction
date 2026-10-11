"""
Assembles the annotated PNGs produced by nc_to_images.py into a single
2D contact-sheet chart:

    - x-axis (across the TOP) = seed values
    - y-axis (down the LEFT)  = reg-weight values
    - each cell               = the annotated PNG for that (seed, reg_weight)

Each tile keeps the per-image title annotations baked in by nc_to_images.py
(target, achieved kappa_eff, converged, iters, final_obj). This script only
reads the seed / reg-weight back out of the FILENAME to place each tile on
the grid -- it does not re-render or re-annotate anything.

Filenames are expected to look like:
    result2D_seed0001_rw2.3000.png

Usage:
    python build_grid.py <png_folder> <output_chart.png>
"""

import sys
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


SEED_RE = re.compile(r"seed(\d+)")
RW_RE = re.compile(r"rw([0-9]+\.?[0-9]*)")


def parse_seed_rw(path):
    """Pull (seed:int, reg_weight:float) out of a filename, or (None, None)."""
    name = os.path.basename(path)
    s = SEED_RE.search(name)
    r = RW_RE.search(name)
    if not s or not r:
        return None, None
    return int(s.group(1)), float(r.group(1))


def build_grid(png_folder, out_path):
    files = sorted(glob.glob(os.path.join(png_folder, "*.png")))
    if not files:
        print(f"No PNGs found in {png_folder}")
        sys.exit(1)

    # Map (seed, rw) -> filepath, and collect the unique axis values.
    cell = {}
    seeds, rws = set(), set()
    skipped = []
    for f in files:
        seed, rw = parse_seed_rw(f)
        if seed is None:
            skipped.append(os.path.basename(f))
            continue
        cell[(seed, rw)] = f
        seeds.add(seed)
        rws.add(rw)

    if skipped:
        print(f"Skipped {len(skipped)} file(s) without seed/rw in name: {skipped[:5]}"
              + (" ..." if len(skipped) > 5 else ""))

    seeds = sorted(seeds)            # x-axis (top), left -> right
    rws = sorted(rws)                # y-axis (left), top -> bottom
    n_cols = len(seeds)
    n_rows = len(rws)
    print(f"Grid: {n_rows} reg-weights x {n_cols} seeds "
          f"({len(cell)} tiles filled, {n_rows * n_cols} cells total)")

    # Size the figure so each tile has room; tiles already contain their titles.
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )

    for i, rw in enumerate(rws):          # rows
        for j, seed in enumerate(seeds):  # cols
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            path = cell.get((seed, rw))
            if path is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        fontsize=9, color="red", transform=ax.transAxes)
            else:
                ax.imshow(mpimg.imread(path))

            # y-axis label (reg-weight) on the leftmost column only
            if j == 0:
                ax.set_ylabel(f"rw={rw:g}", fontsize=11, rotation=90,
                              labelpad=8, fontweight="bold")
            # x-axis label (seed) on the TOP row only, axis on top
            if i == 0:
                ax.set_title(f"seed={seed}", fontsize=11, fontweight="bold", pad=6)

    # Overall axis captions
    fig.supxlabel("seed  ->", fontsize=13, fontweight="bold", y=0.995)
    fig.supylabel("reg-weight  ->", fontsize=13, fontweight="bold", x=0.005)

    fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.985])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python build_grid.py <png_folder> <output_chart.png>")
        sys.exit(1)
    build_grid(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()