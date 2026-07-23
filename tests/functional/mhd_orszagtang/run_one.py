#!/usr/bin/env python3
"""
Run ONE Orszag-Tang resolution and dump the t=1 domain pressure to .npz.

MPI-safe: the sim runs on all ranks (launch with mpirun for speed); field
extraction + npz dump happen on rank 0 only (h5 already holds every rank's
patches), matching the rank-0 pattern in mhd_harris/harris.py.

Usage (per resolution):
  mpirun -n 30 python3 run_one.py <cells>
Label comes from env PHARE_ORDER_LABEL; output -> npz/fields_<label>_<cells>.npz
"""
import os
import sys
from pathlib import Path

import numpy as np

import pyphare.pharein as ph
from pyphare import cpp
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator, startMPI

ph.NO_GUI()
os.environ["PHARE_SCOPE_TIMING"] = "1"

FINAL_TIME = 1.0
CFL = 256.0 / 1430.0
RECONSTRUCTION = "WENOZ"
LIMITER = "None"
TIMESTEPPER = "SSPRK4_5"
NPZ_DIR = Path("npz")


def config(cells, diag_dir):
    dl = 1.0 / cells
    n_steps = max(5, round(FINAL_TIME / (CFL * dl)))
    n_steps += (-n_steps) % 5
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


def domain_pressure(diag_dir, cells):
    """Stitch all (possibly MPI-tiled) patches into the full domain array.

    Under mpirun the level holds many tiled patches; patches[0] is just one
    tile (with NaN ghosts). Place each patch's ghost-stripped interior by its
    box bounds. Bypasses single_patch_for_LO's broken _compute_to_primal path.
    """
    sf = Run(diag_dir).GetMHDP(FINAL_TIME, all_primal=False)
    level = sf.levels(FINAL_TIME)[0]
    full = np.full((cells, cells), np.nan)
    for p in level.patches:
        pd = p.patch_datas[next(iter(p.patch_datas))]
        gx, gy = int(pd.ghosts_nbr[0]), int(pd.ghosts_nbr[1])
        lo, up = p.box.lower, p.box.upper
        inner = pd.dataset[gx:-gx if gx else None, gy:-gy if gy else None]
        full[lo[0]:up[0] + 1, lo[1]:up[1] + 1] = inner
    assert np.isfinite(full).all(), "domain not fully covered by patches"
    return full


def main():
    cells = int(sys.argv[1])
    label = os.environ.get("PHARE_ORDER_LABEL", "unknown")
    diag_dir = f"phare_outputs/ot_{label}_{cells}"

    ph.global_vars.sim = None
    Simulator(config(cells, diag_dir)).run().reset()

    if cpp.mpi_rank() == 0:
        NPZ_DIR.mkdir(parents=True, exist_ok=True)
        P = domain_pressure(diag_dir, cells)
        out = NPZ_DIR / f"fields_{label}_{cells}.npz"
        np.savez(out, P=P, nx=cells, label=label,
                 reconstruction=RECONSTRUCTION, timestepper=TIMESTEPPER)
        print(f"wrote {out}  shape={P.shape}")
    cpp.mpi_barrier()


if __name__ == "__main__":
    startMPI()
    main()
