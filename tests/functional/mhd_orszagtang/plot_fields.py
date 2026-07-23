#!/usr/bin/env python3
"""
Render the Orszag-Tang pressure pattern at t=1 across resolutions and schemes,
as a 2 (scheme) x 3 (resolution) panel of imshow plots.

Reads the run dirs produced by grid_convergence.py. Uses raw dataset extraction
(all_primal=False) to avoid the broken _compute_to_primal plotting path.

Usage:
  python plot_fields.py <o2_out_root> <o4_out_root> <png_out>
where *_out_root is the dir containing phare_outputs/orszag_convergence/run_f{1,2,4}
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyphare.pharesee.run import Run

FINAL_TIME = 1.0
FACTORS = [1, 2, 4]
BASE_CELLS = 64


def pressure(out_root, f):
    cells = BASE_CELLS * f
    diag_dir = str(Path(out_root) / "phare_outputs" / "orszag_convergence" / f"run_f{f}")
    sf = Run(diag_dir).GetMHDP(FINAL_TIME, all_primal=False)
    level = sf.levels(FINAL_TIME)[0]
    pd = level.patches[0].patch_datas
    arr = pd[next(iter(pd))].dataset[:]
    gx = (arr.shape[0] - cells) // 2
    gy = (arr.shape[1] - cells) // 2
    return arr[gx:gx + cells, gy:gy + cells]


def main(o2_root, o4_root, png_out):
    rows = [("o2", o2_root), ("o4", o4_root)]
    fig, axes = plt.subplots(2, len(FACTORS), figsize=(4 * len(FACTORS), 8),
                             constrained_layout=True)
    # shared color scale from the finest field
    ref = pressure(o4_root, FACTORS[-1])
    vmin, vmax = float(ref.min()), float(ref.max())

    for r, (label, root) in enumerate(rows):
        for c, f in enumerate(FACTORS):
            ax = axes[r, c]
            P = pressure(root, f)
            im = ax.imshow(P.T, origin="lower", extent=[0, 1, 0, 1],
                           vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(f"{label}  {BASE_CELLS * f}$^2$")
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.7, label="pressure @ t=1")
    fig.suptitle("Orszag-Tang pressure: 2nd vs 4th order, by resolution", fontsize=15)
    fig.savefig(png_out, dpi=150)
    print(f"wrote {png_out}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: plot_fields.py <o2_out_root> <o4_out_root> <png_out>")
    main(*sys.argv[1:])
