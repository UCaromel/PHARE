#!/usr/bin/env python3
# divB e2e for the higher-order field-refinement operators (refinement_order 0/2/4).
#
# Idea: in the discrete Yee scheme Faraday preserves divB exactly, so on a fine level
# created by B prolongation, max|divB| is set *purely* by the B refinement at the
# coarse-fine boundary. We init B from an analytically divergence-free, periodic vector
# potential Az = A0 cos(kx x) cos(ky y)  ->  Bx = dAz/dy, By = -dAz/dx, div B = 0 exactly.
# A divB-preserving refinement keeps fine-level max|divB| at the smooth coarse-truncation
# floor for every order; a misclassified / overwritten shared face spikes it to O(B/dx).
#
# Two modes, both 2D (divB is identically 0 in 1D):
#   boxes   - deterministic fine level via refinement_boxes (static C-F boundary)
#   tagging - tagger-created level that regrids over time (exercises the regrid path)
#
# Run: mpirun -n 12 python -u divb_refinement.py [boxes|tagging] [orders...]

import sys
import numpy as np

import pyphare.pharein as ph
from pyphare import cpp
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator, startMPI

ph.NO_GUI()
startMPI()

cells = (40, 40)
dl = (0.25, 0.25)  # domain 10 x 10
Lx, Ly = cells[0] * dl[0], cells[1] * dl[1]
kx, ky = 2 * np.pi / Lx, 2 * np.pi / Ly
A0 = 0.05

time_step = 0.002
NSTEPS = {"boxes": 2, "tagging": 20}  # tagging needs steps for regrids to fire

FINE_BOX = [[10, 10], [29, 29]]  # interior: C-F boundaries sit inside the smooth field

# A divB spike from a misclassified/overwritten shared face is O(B/dx) ~ 0.1. The refinement
# is divB-preserving iff the fine level does not AMPLIFY divB beyond the floor it inherits.
# That floor is mode-dependent: for the smooth (resolved) sinusoid the discrete init is
# essentially div-free so the coarse floor is ~1e-8 and fine sits at the ~1e-5 truncation
# level; for the sharp island the discrete point-sample carries a common-mode divB (~3e-3)
# present identically on coarse. So the contract is: fine <= REL_TOL * max(coarse-floor, ABS_TOL)
# AND fine within REL_TOL of the order-0 baseline (isolates the operator from the init floor).
ABS_TOL = 1e-4  # floor below which divB is "already zero" (truncation, not a bug)
REL_TOL = 5.0   # fine must not exceed REL_TOL x (coarse floor) nor REL_TOL x (order-0 fine)


# boxes mode: gentle periodic div-free sinusoid B = curl(Az ẑ), Az = A0 cos(kx x)cos(ky y).
def bx_sin(x, y):
    return -A0 * ky * np.cos(kx * x) * np.sin(ky * y)


def by_sin(x, y):
    return A0 * kx * np.sin(kx * x) * np.cos(ky * y)


# tagging mode: localized div-free island B = curl(Az ẑ), Az = AI exp(-r^2/w^2).
# curl of a z-potential is divergence-free for ANY Az, so divB = 0 exactly; the bump is
# sharp enough (and decays to ~0 at the boundary) to trigger the |ΔBy| tagger and regrid.
AI, WI = 0.4, 0.9
XC, YC = 0.5 * Lx, 0.5 * Ly


def _gauss(x, y):
    return np.exp(-((x - XC) ** 2 + (y - YC) ** 2) / WI**2)


def bx_isl(x, y):
    return AI * (-2.0 * (y - YC) / WI**2) * _gauss(x, y)


def by_isl(x, y):
    return AI * (2.0 * (x - XC) / WI**2) * _gauss(x, y)


def config(mode, order, diag_dir):
    nsteps = NSTEPS[mode]
    check_time = nsteps * time_step
    common = dict(
        time_step=time_step,
        time_step_nbr=nsteps,
        cells=cells,
        dl=dl,
        boundary_types=["periodic", "periodic"],
        refinement_order=order,  # <-- path under test
        strict=True,
        nesting_buffer=0,
        diag_options={
            "format": "phareh5",
            "options": {"dir": diag_dir, "mode": "overwrite"},
        },
    )
    if mode == "boxes":
        bx, by = bx_sin, by_sin
        sim = ph.Simulation(
            refinement="boxes",
            refinement_boxes={"L0": {"B0": FINE_BOX}},
            smallest_patch_size=10,
            largest_patch_size=40,
            **common,
        )
    else:
        bx, by = bx_isl, by_isl
        sim = ph.Simulation(
            refinement="tagging",
            max_nbr_levels=2,
            tag_buffer=1,
            **common,
        )

    def bz(x, y):
        return 0.2 + 0.0 * x

    def density(x, y):
        return 1.0 + 0.0 * x

    def thermal(x, y):
        return 0.1 + 0.0 * x

    def zero(x, y):
        return 0.0 * x

    ph.MaxwellianFluidModel(
        bx=bx,
        by=by,
        bz=bz,
        protons={
            "charge": 1,
            "density": density,
            "vbulkx": zero,
            "vbulky": zero,
            "vbulkz": zero,
            "vthx": thermal,
            "vthy": thermal,
            "vthz": thermal,
            "nbr_part_per_cell": 30,
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.0)
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=np.array([check_time]))
    return sim, check_time


def max_divb_per_level(diag_dir, check_time):
    # IMPORTANT: measure the DOMAIN INTERIOR only. _compute_divB builds divB from the full
    # B datasets (ghosts included) and drops ghosts_nbr, so dataset[:] still spans the ghost
    # band. The fine-level coarse-fine ghost fill is a known divB hot-spot (order-independent,
    # pre-existing) that is NOT a physical interior violation -> strip a ghost margin.
    run = Run(diag_dir)
    B = run.GetB(check_time, all_primal=False)
    blvls = B.levels(check_time)
    ng = int(blvls[min(blvls)].patches[0].patch_datas["Bx"].ghosts_nbr[0])
    divb = run.GetDivB(check_time)
    lvls = divb.levels(check_time)
    out = {}
    for lvl, level in lvls.items():
        m = 0.0
        for patch in level.patches:
            arr = np.abs(patch.patch_datas["value"].dataset[:])
            if all(s > 2 * ng for s in arr.shape):
                arr = arr[tuple(slice(ng, -ng) for _ in arr.shape)]
            if arr.size:
                m = max(m, float(np.nanmax(arr)))
        out[lvl] = m
    return out


def run_order(mode, order):
    diag_dir = f"divb_{mode}_o{order}"
    sim, check_time = config(mode, order, diag_dir)
    if cpp.mpi_rank() == 0:
        print(f"=== divB {mode} refinement_order={order} ===", flush=True)
    Simulator(sim).run().reset()
    ph.global_vars.sim = None
    if cpp.mpi_rank() == 0:
        per = max_divb_per_level(diag_dir, check_time)
        finest = max(per.keys())
        coarse = per[min(per.keys())]
        print(
            f"  order={order}: levels={sorted(per)}  "
            f"max|divB|_fine={per[finest]:.3e}  max|divB|_coarse={coarse:.3e}",
            flush=True,
        )
        return per
    return None


if __name__ == "__main__":
    argv = sys.argv[1:]
    mode = argv[0] if argv and argv[0] in ("boxes", "tagging") else "boxes"
    orders = [int(a) for a in argv if a.isdigit()] or [0, 2, 4]

    res = {o: run_order(mode, o) for o in orders}
    if cpp.mpi_rank() != 0:
        sys.exit(0)

    print(f"=== divB {mode} summary ===", flush=True)
    base_per = res[orders[0]]
    base = base_per[max(base_per.keys())]  # order-0 fine-level divB
    ok = True
    for o in orders:
        per = res[o]
        finest = max(per.keys())
        m = per[finest]
        if finest == min(per.keys()):
            ok = False
            print(f"  order={o}: NO FINE LEVEL FORMED  [FAIL]", flush=True)
            continue
        coarse = per[min(per.keys())]
        ceiling = REL_TOL * max(coarse, ABS_TOL)  # not amplified beyond the inherited floor
        floor_ok = m <= ceiling
        rel_ok = m <= REL_TOL * base if base else True  # operator vs legacy baseline
        status = "OK" if (floor_ok and rel_ok) else "FAIL"
        if not (floor_ok and rel_ok):
            ok = False
        print(
            f"  order={o}: fine={m:.3e}  coarse={coarse:.3e}  base={base:.3e}  "
            f"fine/coarse={m / coarse if coarse else float('inf'):.2f}  "
            f"fine/base={m / base if base else float('inf'):.2f}  [{status}]",
            flush=True,
        )
    print(f"DIVB_{mode.upper()}_OK" if ok else f"DIVB_{mode.upper()}_FAIL", flush=True)
    sys.exit(0 if ok else 1)
