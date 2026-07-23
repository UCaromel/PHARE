#!/usr/bin/env python3
"""Render OT pressure fields from the npz dumps: o2-vs-o4 ladder + 1024 truth."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load(npz_dir, label, nx):
    return np.load(Path(npz_dir) / f"fields_{label}_{nx}.npz")["P"]


def main(npz_dir):
    cols = [128, 256, 512]
    o2 = [load(npz_dir, "o2", n) for n in cols]
    o4 = [load(npz_dir, "o4", n) for n in cols]
    truth = load(npz_dir, "o4", 1024)

    vmax = max(a.max() for a in o2 + o4)        # shared scale for the ladder
    vmin = min(a.min() for a in o2 + o4)

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1.15], figure=fig)

    for r, (lbl, row) in enumerate([("o2", o2), ("o4", o4)]):
        for c, (n, P) in enumerate(zip(cols, row)):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(P.T, origin="lower", extent=[0, 1, 0, 1],
                           vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(f"{lbl}  {n}$^2$", fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=fig.axes, shrink=0.5, label="pressure @ t=1 (ladder scale)")

    axt = fig.add_subplot(gs[:, 3])
    imt = axt.imshow(truth.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    axt.set_title("o4 1024$^2$ (reference)", fontsize=13)
    axt.set_xticks([]); axt.set_yticks([])
    fig.colorbar(imt, ax=axt, shrink=0.6)

    fig.suptitle("Orszag-Tang pressure @ t=1: 2nd vs 4th order convergence",
                 fontsize=16)
    out = "ot_fields_highres.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "npz")
