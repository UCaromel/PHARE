#!/usr/bin/env python3
"""
spacetime_plot.py — 1-D space-time colourmap for PHARE diagnostic output.

Reads a single .h5 or .vtkhdf diagnostic file produced by PHARE, assembles a
2-D (space × time) image, and saves or displays it.

  x-axis  : spatial coordinate
  y-axis  : simulation time
  colour  : field magnitude (or selected component)

Usage
-----
  python tools/spacetime_plot.py <file.h5|file.vtkhdf> [options]

Examples
--------
  # scalar field — quantity is detected automatically
  python tools/spacetime_plot.py phare_outputs/shock/rho.h5

  # vector field — pick a component
  python tools/spacetime_plot.py phare_outputs/shock/B.h5 --qty Bx

  # vector field — plot L2 magnitude of all components
  python tools/spacetime_plot.py phare_outputs/shock/B.h5 --magnitude

  # save to file, custom colourmap, log scale
  python tools/spacetime_plot.py phare_outputs/shock/rho.h5 \\
      --output rho_spacetime.png --cmap inferno --log
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# matplotlib style
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    """Apply publication-quality rcParams (LaTeX fonts, high DPI)."""
    plt.rcParams.update(
        {
            "axes.formatter.limits": [-3, 5],
            "axes.labelsize": "large",
            "figure.autolayout": True,
            "figure.dpi": 300,
            "figure.figsize": (6, 4),
            "font.size": 12.0,
            "text.usetex": True,
            "text.latex.preamble": " ".join(
                [
                    r"\usepackage{amsmath}",
                    r"\usepackage{amssymb}",
                ]
            ),
            "xtick.labelsize": "large",
            "ytick.labelsize": "large",
        }
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_phare():
    """Import PHARE modules, with a helpful error if PYTHONPATH is not set."""
    try:
        from pyphare.pharesee.hierarchy import hierarchy_from, all_times_from
        from pyphare.pharesee.hierarchy.hierarchy_utils import flat_finest_field_1d
        return hierarchy_from, all_times_from, flat_finest_field_1d
    except ImportError as exc:
        sys.exit(
            f"Cannot import pyphare: {exc}\n"
            "Make sure PYTHONPATH includes the repo root and the build directory, e.g.:\n"
            "  PYTHONPATH=<build>:<repo>/pyphare:<repo> python tools/spacetime_plot.py ..."
        )


def _detect_quantities(h5_file: str, all_times_from, hierarchy_from) -> list[str]:
    """Return sorted list of pdata quantity names present in the file."""
    times = all_times_from(h5_file)
    if len(times) == 0:
        sys.exit(f"No timestamps found in {h5_file}")
    hier = hierarchy_from(h5_filename=h5_file, times=[times[0]])
    lvl0_patches = hier.level(0, times[0]).patches
    if not lvl0_patches:
        sys.exit("No patches found at level 0")
    return sorted(lvl0_patches[0].patch_datas.keys())


def _build_spacetime(
    h5_file: str,
    qtys: list[str],
    all_times_from,
    hierarchy_from,
    flat_finest_field_1d,
    npoints: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 2-D (time × space) data array for the given list of quantities.

    When *qtys* contains more than one name the L2 magnitude is returned.
    All profiles are linearly interpolated onto a uniform x-grid of *npoints*
    points spanning the full domain at each time step.

    Returns
    -------
    x_uniform   : 1-D array, shape (npoints,)  — spatial coordinate
    times_arr   : 1-D array, shape (Nt,)        — simulation times
    data_2d     : 2-D array, shape (Nt, npoints) — field values
    """
    times = all_times_from(h5_file)
    if len(times) == 0:
        sys.exit(f"No timestamps found in {h5_file}")

    # Determine the common x-range from the first snapshot
    hier0 = hierarchy_from(h5_filename=h5_file, times=[times[0]])
    # Use the first quantity to probe the domain extent
    ref_data, ref_x = flat_finest_field_1d(hier0, qtys[0], times[0])
    idx = np.argsort(ref_x)
    x_uniform = np.linspace(ref_x[idx[0]], ref_x[idx[-1]], npoints)

    data_2d = np.full((len(times), npoints), np.nan)

    for it, t in enumerate(times):
        hier = hierarchy_from(h5_filename=h5_file, times=[t]) if it > 0 else hier0

        if len(qtys) == 1:
            vals, xs = flat_finest_field_1d(hier, qtys[0], t)
            idx = np.argsort(xs)
            data_2d[it] = np.interp(x_uniform, xs[idx], vals[idx])
        else:
            # L2 magnitude over all components
            acc = np.zeros(npoints)
            for qty in qtys:
                vals, xs = flat_finest_field_1d(hier, qty, t)
                idx = np.argsort(xs)
                acc += np.interp(x_uniform, xs[idx], vals[idx]) ** 2
            data_2d[it] = np.sqrt(acc)

    return x_uniform, np.asarray(times, dtype=float), data_2d


# ---------------------------------------------------------------------------
# quantity → LaTeX label mapping
# ---------------------------------------------------------------------------

_LATEX_LABELS: dict[str, str] = {
    # MHD scalars
    "mhdRho":  r"$\rho$",
    "mhdP":    r"$P$",
    "mhdEtot": r"$E_\mathrm{tot}$",
    # MHD velocity components
    "mhdVx": r"$V_x$",
    "mhdVy": r"$V_y$",
    "mhdVz": r"$V_z$",
    # Magnetic field components
    "Bx": r"$B_x$",
    "By": r"$B_y$",
    "Bz": r"$B_z$",
    # Hybrid scalars
    "rho":      r"$\rho$",
    "pressure": r"$P$",
    # Hybrid velocity components
    "Vx": r"$V_x$",
    "Vy": r"$V_y$",
    "Vz": r"$V_z$",
    # Electric field components
    "Ex": r"$E_x$",
    "Ey": r"$E_y$",
    "Ez": r"$E_z$",
}


def _latex_label(qty: str, is_magnitude: bool = False, stem: str = "") -> str:
    """Return a LaTeX string for a pdata quantity name."""
    if is_magnitude:
        base = _LATEX_LABELS.get(stem, rf"$\mathrm{{{stem}}}$")
        # Strip outer $…$ to insert |…|
        inner = base[1:-1] if base.startswith("$") and base.endswith("$") else base
        return rf"$|{inner}|$"
    return _LATEX_LABELS.get(qty, rf"$\mathrm{{{qty}}}$")

def spacetime_plot(
    h5_file: str,
    qty: str | None,
    magnitude: bool,
    cmap: str | None,
    output: str | None,
    npoints: int,
    log: bool,
) -> None:
    hierarchy_from, all_times_from, flat_finest_field_1d = _load_phare()

    available = _detect_quantities(h5_file, all_times_from, hierarchy_from)

    if not available:
        sys.exit("No field quantities found in the file.")

    # --- resolve which pdata quantity/ies to plot ---
    if magnitude:
        qtys = available  # L2 magnitude over all components in the file
        label = _latex_label("", is_magnitude=True, stem=Path(h5_file).stem)
    elif qty is not None:
        if qty not in available:
            sys.exit(
                f"Quantity '{qty}' not found. Available: {', '.join(available)}"
            )
        qtys = [qty]
        label = _latex_label(qty)
    elif len(available) == 1:
        qtys = available
        label = _latex_label(available[0])
    else:
        # Multiple quantities present, no selection — list them and exit
        print(
            f"File contains multiple quantities: {', '.join(available)}\n"
            "Use --qty NAME to select one, or --magnitude for the L2 norm."
        )
        return

    print(f"Plotting {label} from {h5_file} …")
    x, times, data = _build_spacetime(
        h5_file, qtys, all_times_from, hierarchy_from, flat_finest_field_1d, npoints
    )

    # --- colour normalisation ---
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    if log:
        # clip negative/zero values for log scale
        vmin = max(vmin, np.nanmax(np.abs(data)) * 1e-6)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        # use symmetric norm for signed data (zero-centred colourmap)
        if vmin < 0 < vmax:
            absmax = max(abs(vmin), abs(vmax))
            norm = mcolors.Normalize(vmin=-absmax, vmax=absmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # --- colourmap default ---
    if cmap is None:
        cmap = "RdBu_r" if (vmin < 0 < vmax) else "inferno"

    # --- figure ---
    _apply_style()
    fig, ax = plt.subplots()

    # Use no interpolation: one data pixel maps directly to one display pixel,
    # which keeps sharp features (shocks, discontinuities) crisp.
    # Smooth interpolations (bilinear, spline*) spread each pixel over neighbours,
    # blurring sharp gradients — avoid them here.
    extent = [x[0], x[-1], times[0], times[-1]]
    mesh = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="spline16",
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(label)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")

    fig.tight_layout()

    if output:
        fig.savefig(output, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("file", help=".h5 or .vtkhdf diagnostic file")
    p.add_argument(
        "--qty",
        default=None,
        help="pdata quantity name to plot (e.g. Bx, mhdRho). "
             "Omit to auto-detect for scalar files.",
    )
    p.add_argument(
        "--magnitude",
        action="store_true",
        help="Plot L2 magnitude of all components in the file.",
    )
    p.add_argument("--cmap", default=None, help="Matplotlib colormap name.")
    p.add_argument(
        "--output", "-o", default=None,
        help="Save figure to this path instead of displaying. "
             "Format is inferred from the extension (.png, .pdf, .svg, …).",
    )
    p.add_argument(
        "--npoints",
        type=int,
        default=1024,
        help="Number of uniform spatial interpolation points (default: 1024).",
    )
    p.add_argument("--log", action="store_true", help="Use logarithmic colour scale.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    spacetime_plot(
        h5_file=args.file,
        qty=args.qty,
        magnitude=args.magnitude,
        cmap=args.cmap,
        output=args.output,
        npoints=args.npoints,
        log=args.log,
    )


if __name__ == "__main__":
    main()
