#!/usr/bin/env python3
"""
Grid-convergence study on Orszag-Tang at t=1 (the canonical MHD picture).

Question: how many grid points does OT@t=1 need to be "converged" with the
scheme compiled into the current binary?

Order (2nd vs 4th FV) is a build/branch property, NOT a runtime knob, so this
SAME script runs on each branch and writes a JSON labelled by PHARE_ORDER_LABEL;
compare.py merges two such JSONs into the showing.

Method (self-convergence, no analytic reference):
  - periodic uniform grid (AMR off), domain [0,1]^2
  - sweep cells by powers of two, FINEST run = truth (256^2)
  - dt scales with dx at fixed CFL ratio (explicit scheme)
  - block-average the finest pressure down to each coarse grid, relative L1
  - converged Nx = smallest Nx with error < TOL

Caveat: OT@t=1 has shocks -> L1 degrades toward 1st order at discontinuities for
BOTH schemes, so the o2-vs-o4 separation is smaller than on a smooth problem.

Serial only (single-patch gather), matching mhd_convergence/convergence.py.
"""
import os
import json
import subprocess
from pathlib import Path

import numpy as np

import pyphare.pharein as ph
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator
from pyphare.pharesee.hierarchy.hierarchy_utils import single_patch_for_LO

ph.NO_GUI()
os.environ["PHARE_SCOPE_TIMING"] = "1"

# ----------------------------------------------------------------------------- knobs
FACTORS = [1, 2, 4]      # base 64 -> 64,128,256 ; truth = 256 (canonical OT res)
BASE_CELLS = 64
FINAL_TIME = 1.0
CFL = 256.0 / 1430.0     # dt/dx from the reference OT test (256^2, dt=1/1430)
QTY = "P"                # cell-centered -> exact block-restriction; classic OT panel
TOL = 1e-2               # relative-L1 "converged" threshold (tune to taste)
OUT = Path("phare_outputs/orszag_convergence")

# Held fixed across branches: high-order reconstruction + high-order time
# integrator so the o4 FV scheme can express its order; identical on both
# branches so the only difference is the branch's FV spatial order.
RECONSTRUCTION = "WENOZ"
LIMITER = "None"
TIMESTEPPER = "SSPRK4_5"


def order_label():
    if "PHARE_ORDER_LABEL" in os.environ:
        return os.environ["PHARE_ORDER_LABEL"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def config(cells, diag_dir):
    dl = 1.0 / cells
    n_steps = max(5, round(FINAL_TIME / (CFL * dl)))
    n_steps += (-n_steps) % 5          # divisible by 5 so dumps land on steps
    time_step = FINAL_TIME / n_steps

    sim = ph.Simulation(
        time_step=time_step,
        final_time=FINAL_TIME,
        cells=(cells, cells),
        dl=(dl, dl),
        refinement="tagging",
        max_mhd_level=1,
        max_nbr_levels=1,
        hyper_resistivity=0.0,
        resistivity=0.0,
        diag_options={
            "format": "phareh5",
            "options": {"dir": diag_dir, "mode": "overwrite"},
        },
        strict=True,
        nesting_buffer=1,
        hyper_mode="spatial",
        eta=0.0,
        nu=0.0,
        gamma=5.0 / 3.0,
        reconstruction=RECONSTRUCTION,
        limiter=LIMITER,
        riemann="Rusanov",
        mhd_timestepper=TIMESTEPPER,
        hall=False,
        res=False,
        hyper_res=False,
        model_options=["MHDModel"],
    )

    B0 = 1.0 / np.sqrt(4.0 * np.pi)

    def density(x, y):
        return 25.0 / (36.0 * np.pi)

    def vx(x, y):
        return -np.sin(2.0 * np.pi * y / sim.simulation_domain()[1])

    def vy(x, y):
        return np.sin(2.0 * np.pi * x / sim.simulation_domain()[0])

    def vz(x, y):
        return 0.0

    def bx(x, y):
        return -B0 * np.sin(2.0 * np.pi * y / sim.simulation_domain()[1])

    def by(x, y):
        return B0 * np.sin(4.0 * np.pi * x / sim.simulation_domain()[0])

    def bz(x, y):
        return 0.0

    def p(x, y):
        return 5.0 / (12.0 * np.pi)

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)
    ph.MHDDiagnostics(quantity="P", write_timestamps=[0.0, FINAL_TIME])
    return sim


def domain_field(diag_dir, cells):
    """Single-patch, ghost-stripped domain pressure at FINAL_TIME."""
    run = Run(diag_dir)
    # Use all_primal=False to bypass _compute_to_primal (broken on older branches).
    # This returns the raw PatchHierarchy or ScalarField depending on the branch.
    # For pressure (cell-centered), the raw data is already on the primal grid.
    sf = run.GetMHDP(FINAL_TIME, all_primal=False)
    # Both ScalarField and PatchHierarchy support .levels(time) on current branches.
    level = sf.levels(FINAL_TIME)[0]
    pd = level.patches[0].patch_datas
    arr = pd[next(iter(pd))].dataset[:]
    gx = (arr.shape[0] - cells) // 2
    gy = (arr.shape[1] - cells) // 2
    return arr[gx:gx + cells, gy:gy + cells]


def restrict(fine, ratio):
    nx, ny = fine.shape[0] // ratio, fine.shape[1] // ratio
    return fine.reshape(nx, ratio, ny, ratio).mean(axis=(1, 3))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    label = order_label()

    fields = {}
    for f in FACTORS:
        cells = BASE_CELLS * f
        diag_dir = str(OUT / f"run_f{f}")
        ph.global_vars.sim = None
        Simulator(config(cells, diag_dir)).run().reset()
        fields[f] = domain_field(diag_dir, cells)

    finest = FACTORS[-1]
    truth = fields[finest]
    nxs, errors = [], []
    for f in FACTORS[:-1]:
        ref = restrict(truth, finest // f)
        e = float(np.sum(np.abs(fields[f] - ref)) / np.sum(np.abs(ref)))
        nxs.append(BASE_CELLS * f)
        errors.append(e)

    converged_nx = next((n for n, e in zip(nxs, errors) if e < TOL), None)

    result = {
        "label": label, "problem": "orszag_tang", "qty": QTY, "tol": TOL,
        "final_time": FINAL_TIME, "factors": FACTORS,
        "nx": nxs, "rel_l1_err": errors,
        "converged_nx": converged_nx, "truth_nx": BASE_CELLS * finest,
        "reconstruction": RECONSTRUCTION, "limiter": LIMITER,
        "timestepper": TIMESTEPPER,
    }
    path = OUT / f"convergence_{label.replace('/', '_')}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
