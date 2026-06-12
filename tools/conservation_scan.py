#!/usr/bin/env python3
"""Conservation scan for coupled MHD(L0)/Hybrid(L1) runs — phase 8 evidence base.

Reads an existing diag dir (no simulation) and produces time series of
domain integrals, split so coarse-level non-conservation (missing reflux at
the MHD/Hybrid CF boundary) is visible:

  - mass:      total = ∫L0-uncovered mhd_rho + ∫L1 ions_mass_density,
               plus full-L0 and L1-only series
  - momentum:  same split, rho*V per component (L0 ddd product,
               L1 ppp node product then corner-average)
  - energy:    magnetic energy per level (shared EM_B), L0 total energy
               P/(gamma-1) + 0.5*rho*V^2 + 0.5*B^2 (mhd_Etot not dumped)

All centerings are detected data-driven (dataset shape vs patch box + ghosts),
so MHD ddd and hybrid ppp/Yee data go through the same cell-value extractor.
Regrid times = dumps where the L1 patch-box set changed (from EM_B).

Usage: python conservation_scan.py <diag_dir> <out_dir> [--gamma 1.6667]
"""

import argparse
import json
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyphare.core import box as boxm
from pyphare.pharesee.hierarchy.fromh5 import get_times_from_h5
from pyphare.pharesee.run import Run

RATIO = 2  # refinement ratio


# ---------------------------------------------------------------- extraction


def cell_values(pdata, patch_box, ubox, values=None):
    """Cell-centered values of pdata over ubox (subset of patch_box, AMR space).

    Per-dim centering is detected from the dataset shape: interior extent equal
    to the box cell count is dual (taken as-is), cell count + 1 is primal
    (adjacent nodes averaged). Works for ddd, ppp and Yee-staggered data.
    """
    arr = np.asarray(pdata.dataset[:]) if values is None else values
    ng = np.asarray(pdata.ghosts_nbr)
    ncells = patch_box.shape
    lo = ubox.lower - patch_box.lower
    n = ubox.shape

    for d in range(patch_box.ndim):
        interior = arr.shape[d] - 2 * int(ng[d])
        start = int(ng[d] + lo[d])
        if interior == ncells[d]:  # dual: cell-centered already
            arr = np.take(arr, range(start, start + int(n[d])), axis=d)
        elif interior == ncells[d] + 1:  # primal: average adjacent nodes
            a = np.take(arr, range(start, start + int(n[d])), axis=d)
            b = np.take(arr, range(start + 1, start + 1 + int(n[d])), axis=d)
            arr = 0.5 * (a + b)
        else:
            raise ValueError(
                f"cannot infer centering: axis {d} interior {interior} "
                f"vs {ncells[d]} cells (ghosts {ng[d]})"
            )
    return arr


def uncovered_boxes(patch_box, covered):
    boxes = [patch_box]
    for cbox in covered:
        boxes = [piece for b in boxes for piece in (b - cbox)]
    return boxes


def level_integral(level, key, covered=(), values_of=None):
    """∫ field dV over the level, skipping cells covered by `covered` boxes.

    values_of(patch) may supply a derived node/cell array (e.g. rho*V, B^2)
    instead of the raw dataset; it must have the dataset's shape.
    """
    total = 0.0
    for patch in level.patches:
        if not patch.patch_datas:
            continue
        pdata = patch.patch_datas[key]
        cell_vol = float(np.prod(pdata.dl))
        values = values_of(patch) if values_of is not None else None
        for ubox in uncovered_boxes(patch.box, covered):
            total += float(
                np.sum(cell_values(pdata, patch.box, ubox, values=values)) * cell_vol
            )
    return total


def single_key(patch_datas, *preferred):
    for cand in preferred:
        if cand in patch_datas:
            return cand
    keys = list(patch_datas.keys())
    if len(keys) == 1:
        return keys[0]
    raise KeyError(f"ambiguous keys {keys}, none of {preferred}")


def vector_keys(patch_datas):
    out = {}
    for axis in "xyz":
        matches = [k for k in patch_datas if k.lower().endswith(axis)]
        if len(matches) != 1:
            raise KeyError(f"no unique '{axis}' key in {list(patch_datas)}")
        out[axis] = matches[0]
    return out


def first_patch_datas(hier, ilvl, time):
    for patch in hier.level(ilvl, time).patches:
        if patch.patch_datas:
            return patch.patch_datas
    raise RuntimeError(f"level {ilvl} has no patch data")


# ---------------------------------------------------------------- per-time


def l1_coverage(b_hier, time):
    """L1 patch boxes coarsened onto L0 index space (empty if no L1)."""
    lvls = b_hier.levels(time)
    if 1 not in lvls:
        return [], []
    boxes = [p.box for p in lvls[1].patches]
    return boxes, [boxm.coarsen(b, RATIO) for b in boxes]


def mass_at_hybrid(run, time, covered):
    """Hybrid-only run: ions_mass_density on every level."""
    rho = run._get_hier_for(time, "ions_mass_density")
    key = single_key(first_patch_datas(rho, 0, time), "rho", "mass_density")
    l0_full = level_integral(rho.level(0, time), key)
    l0_unc = level_integral(rho.level(0, time), key, covered=covered)
    l1 = (
        level_integral(rho.level(1, time), key)
        if 1 in rho.levels(time)
        else 0.0
    )
    return {"L0_full": l0_full, "L0_uncovered": l0_unc, "L1": l1, "total": l0_unc + l1}


def momentum_at_hybrid(run, time, covered):
    rho = run._get_hier_for(time, "ions_mass_density")
    v = run._get_hier_for(time, "ions_bulkVelocity")
    key = single_key(first_patch_datas(rho, 0, time), "rho", "mass_density")
    vk = vector_keys(first_patch_datas(v, 0, time))

    def product(ilvl, v_key):
        rho_by_id = {p.id: p for p in rho.level(ilvl, time).patches if p.patch_datas}

        def values_of(patch):
            rho_pd = rho_by_id[patch.id].patch_datas[key]
            return np.asarray(patch.patch_datas[v_key].dataset[:]) * np.asarray(
                rho_pd.dataset[:]
            )

        return values_of

    out = {}
    for axis in "xyz":
        l0_full = level_integral(v.level(0, time), vk[axis], values_of=product(0, vk[axis]))
        l0_unc = level_integral(
            v.level(0, time), vk[axis], covered=covered, values_of=product(0, vk[axis])
        )
        l1 = (
            level_integral(v.level(1, time), vk[axis], values_of=product(1, vk[axis]))
            if 1 in v.levels(time)
            else 0.0
        )
        out[axis] = {
            "L0_full": l0_full,
            "L0_uncovered": l0_unc,
            "L1": l1,
            "total": l0_unc + l1,
        }
    return out


def mass_at(run, time, covered):
    rho0 = run._get_hier_for(time, "mhd_rho")
    key0 = single_key(first_patch_datas(rho0, 0, time), "mhdRho", "rho")
    l0_full = level_integral(rho0.level(0, time), key0)
    l0_unc = level_integral(rho0.level(0, time), key0, covered=covered)

    rho1 = run._get_hier_for(time, "ions_mass_density")
    key1 = single_key(first_patch_datas(rho1, 1, time), "rho", "mass_density")
    l1 = level_integral(rho1.level(1, time), key1)
    return {"L0_full": l0_full, "L0_uncovered": l0_unc, "L1": l1, "total": l0_unc + l1}


def momentum_at(run, time, covered):
    rho0 = run._get_hier_for(time, "mhd_rho")
    v0 = run._get_hier_for(time, "mhd_V")
    key0 = single_key(first_patch_datas(rho0, 0, time), "mhdRho", "rho")
    vk0 = vector_keys(first_patch_datas(v0, 0, time))

    rho1 = run._get_hier_for(time, "ions_mass_density")
    v1 = run._get_hier_for(time, "ions_bulkVelocity")
    key1 = single_key(first_patch_datas(rho1, 1, time), "rho", "mass_density")
    vk1 = vector_keys(first_patch_datas(v1, 1, time))

    def product(rho_hier, v_hier, ilvl, rho_key, v_key):
        rho_by_id = {
            p.id: p for p in rho_hier.level(ilvl, time).patches if p.patch_datas
        }

        def values_of(patch):
            rho_pd = rho_by_id[patch.id].patch_datas[rho_key]
            return np.asarray(patch.patch_datas[v_key].dataset[:]) * np.asarray(
                rho_pd.dataset[:]
            )

        return values_of

    out = {}
    for axis in "xyz":
        l0_full = level_integral(
            v0.level(0, time),
            vk0[axis],
            values_of=product(rho0, v0, 0, key0, vk0[axis]),
        )
        l0_unc = level_integral(
            v0.level(0, time),
            vk0[axis],
            covered=covered,
            values_of=product(rho0, v0, 0, key0, vk0[axis]),
        )
        l1 = level_integral(
            v1.level(1, time),
            vk1[axis],
            values_of=product(rho1, v1, 1, key1, vk1[axis]),
        )
        out[axis] = {
            "L0_full": l0_full,
            "L0_uncovered": l0_unc,
            "L1": l1,
            "total": l0_unc + l1,
        }
    return out


def magnetic_energy_at(b_hier, time, covered):
    out = {}
    for ilvl, cov in ((0, covered), (1, [])):
        if ilvl not in b_hier.levels(time):
            continue
        level = b_hier.level(ilvl, time)
        bk = vector_keys(first_patch_datas(b_hier, ilvl, time))
        nrj_full, nrj_unc = 0.0, 0.0
        for axis in "xyz":
            sq = lambda patch, k=bk[axis]: (
                np.asarray(patch.patch_datas[k].dataset[:]) ** 2
            )
            nrj_full += 0.5 * level_integral(level, bk[axis], values_of=sq)
            nrj_unc += 0.5 * level_integral(level, bk[axis], covered=cov, values_of=sq)
        out[f"L{ilvl}_full"] = nrj_full
        out[f"L{ilvl}_uncovered"] = nrj_unc
    return out


def l0_etot_at(run, b_hier, time, covered, gamma):
    """L0 total energy density P/(g-1) + 0.5 rho V^2 (ddd) + 0.5 B^2 (Yee).

    Hydro part and magnetic part integrate separately (different centerings);
    both go through the same cell-value extractor so the sum is per-cell exact.
    """
    rho0 = run._get_hier_for(time, "mhd_rho")
    v0 = run._get_hier_for(time, "mhd_V")
    p0 = run._get_hier_for(time, "mhd_P")
    key_rho = single_key(first_patch_datas(rho0, 0, time), "mhdRho", "rho")
    key_p = single_key(first_patch_datas(p0, 0, time), "mhdP", "P")
    vk = vector_keys(first_patch_datas(v0, 0, time))

    rho_by_id = {p.id: p for p in rho0.level(0, time).patches if p.patch_datas}
    v_by_id = {p.id: p for p in v0.level(0, time).patches if p.patch_datas}

    def hydro_values(patch):
        rho = np.asarray(rho_by_id[patch.id].patch_datas[key_rho].dataset[:])
        vpds = v_by_id[patch.id].patch_datas
        v2 = sum(np.asarray(vpds[vk[a]].dataset[:]) ** 2 for a in "xyz")
        press = np.asarray(patch.patch_datas[key_p].dataset[:])
        return press / (gamma - 1.0) + 0.5 * rho * v2

    out = {}
    for tag, cov in (("full", []), ("uncovered", covered)):
        hydro = level_integral(p0.level(0, time), key_p, covered=cov, values_of=hydro_values)
        mag = magnetic_energy_at(b_hier, time, covered)[
            "L0_full" if tag == "full" else "L0_uncovered"
        ]
        out[tag] = hydro + mag
    return out


# ---------------------------------------------------------------- driver


def l1_box_set(b_hier, time):
    lvls = b_hier.levels(time)
    if 1 not in lvls:
        return ()
    return tuple(
        sorted(
            (tuple(int(i) for i in p.box.lower), tuple(int(i) for i in p.box.upper))
            for p in lvls[1].patches
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("diag_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--gamma", type=float, default=5.0 / 3.0)
    ap.add_argument(
        "--mode",
        choices=("coupled", "hybrid"),
        default="coupled",
        help="coupled: MHD L0 + hybrid L1; hybrid: ions moments on all levels",
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    run = Run(args.diag_dir)
    t_of = lambda f: [
        float(t) for t in get_times_from_h5(os.path.join(args.diag_dir, f))
    ]
    times_b = t_of("EM_B.h5")
    if args.mode == "hybrid":
        times_moments = sorted(
            set(t_of("ions_mass_density.h5")) & set(t_of("ions_bulkVelocity.h5"))
        )
    else:
        times_moments = sorted(
            set(t_of("mhd_rho.h5"))
            & set(t_of("ions_mass_density.h5"))
            & set(t_of("mhd_V.h5"))
            & set(t_of("ions_bulkVelocity.h5"))
            & set(t_of("mhd_P.h5"))
        )
    print(f"B dumps: {len(times_b)}  moment dumps: {len(times_moments)}", flush=True)

    # regrid markers from B (densest series)
    regrids = []
    prev = None
    for t in times_b:
        cur = l1_box_set(run.GetB(t, all_primal=False), t)
        if prev is not None and cur != prev:
            regrids.append(t)
        prev = cur
    print(f"L1 layout changes at: {[f'{t:.3f}' for t in regrids]}", flush=True)

    series = {"times": times_moments, "regrids": regrids, "gamma": args.gamma}
    mass, mom, mag, etot = [], [], [], []
    for t in times_moments:
        b_hier = run.GetB(t, all_primal=False)
        _, covered = l1_coverage(b_hier, t)
        if args.mode == "hybrid":
            mass.append(mass_at_hybrid(run, t, covered))
            mom.append(momentum_at_hybrid(run, t, covered))
            mag.append(magnetic_energy_at(b_hier, t, covered))
        else:
            mass.append(mass_at(run, t, covered))
            mom.append(momentum_at(run, t, covered))
            mag.append(magnetic_energy_at(b_hier, t, covered))
            etot.append(l0_etot_at(run, b_hier, t, covered, args.gamma))
        print(
            f"t={t:7.3f} mass total={mass[-1]['total']:.6f} "
            f"L0full={mass[-1]['L0_full']:.6f} L1={mass[-1]['L1']:.6f}",
            flush=True,
        )
    series.update(mass=mass, momentum=mom, magnetic=mag, l0_etot=etot)

    with open(os.path.join(args.out_dir, "conservation.json"), "w") as f:
        json.dump(series, f, indent=1)

    # ------------------------------------------------------------- plots
    ts = np.asarray(times_moments)

    def mark_regrids(ax):
        for t in regrids:
            ax.axvline(t, color="0.8", lw=0.6, zorder=0)

    def drift(vals):
        v = np.asarray(vals, dtype=float)
        return (v - v[0]) / abs(v[0]) if v[0] != 0 else v - v[0]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for key, label in (
        ("total", "L0-uncovered + L1 (total)"),
        ("L0_full", "L0 full"),
        ("L0_uncovered", "L0 uncovered"),
        ("L1", "L1"),
    ):
        axes[0].plot(ts, [m[key] for m in mass], marker=".", label=label)
        axes[1].plot(ts, drift([m[key] for m in mass]), marker=".", label=label)
    axes[0].set_ylabel("mass")
    axes[1].set_ylabel("relative drift")
    axes[1].set_xlabel("t")
    for ax in axes:
        mark_regrids(ax)
        ax.legend(fontsize=8)
    axes[0].set_title(f"mass conservation — {os.path.basename(args.diag_dir.rstrip('/'))} ({args.mode})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "mass.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, axis in zip(axes, "xyz"):
        for key in ("total", "L0_full", "L1"):
            ax.plot(ts, [m[axis][key] for m in mom], marker=".", label=key)
        ax.set_ylabel(f"momentum {axis}")
        mark_regrids(ax)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("t")
    axes[0].set_title("momentum (rho V) — coupled harris")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "momentum.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for key in sorted(mag[0]):
        axes[0].plot(ts, [m[key] for m in mag], marker=".", label=f"B nrj {key}")
    if etot:
        for key in ("full", "uncovered"):
            axes[1].plot(ts, [e[key] for e in etot], marker=".", label=f"L0 Etot {key}")
    for ax in axes:
        mark_regrids(ax)
        ax.legend(fontsize=8)
    axes[1].set_xlabel("t")
    axes[0].set_title("magnetic energy per level / L0 total energy")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "energy.png"), dpi=200)
    plt.close(fig)

    # headline numbers
    m0, mf = mass[0], mass[-1]
    print("\n=== headline ===")
    print(
        f"total mass drift: {(mf['total'] - m0['total']) / m0['total'] * 100:+.4f}% "
        f"over t=[{ts[0]:.3f}, {ts[-1]:.3f}]"
    )
    print(
        f"L0-full mass drift: {(mf['L0_full'] - m0['L0_full']) / m0['L0_full'] * 100:+.4f}%"
    )
    if etot:
        e0, ef = etot[0], etot[-1]
        print(
            f"L0 Etot (full) drift: {(ef['full'] - e0['full']) / e0['full'] * 100:+.4f}%"
        )
    g0, gf = mag[0], mag[-1]
    bkey = "L0_full"
    print(f"B energy {bkey}: {g0[bkey]:.6f} -> {gf[bkey]:.6f}")


if __name__ == "__main__":
    sys.exit(main())
