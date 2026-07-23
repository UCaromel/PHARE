#!/usr/bin/env python3
# Tagging-driven counterpart of divb_5level_crash.py. Same MHD physics/numerics and Harris
# Bx(y)-only init, but levels are created by the tagger (refinement="tagging") with
# nesting_buffer=1 instead of explicit nested boxes. We write a t=0 diagnostic to VERIFY all
# NLEVELS are already present at init, then run a single advance and watch for the crash.
#
# Run: mpirun --oversubscribe -n 4 python3 -u divb_5level_tagging.py [nlevels] [order]

import sys

import numpy as np

import pyphare.pharein as ph
from pyphare import cpp
from pyphare.simulator.simulator import Simulator, startMPI
from pyphare.pharesee.run import Run

ph.NO_GUI()
startMPI()

cells = (20, 80)
dl = (0.5, 0.5)
Lx, Ly = cells[0] * dl[0], cells[1] * dl[1]
L = 0.5
V1, V2 = -1.0, 1.0
K = 0.7

time_step = 0.002
NSTEPS = 1

NLEVELS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ORDER = int(sys.argv[2]) if len(sys.argv) > 2 else 4
LIMITER = "none"


def _S(y, y0, l):
    return 0.5 * (1.0 + np.tanh((y - y0) / l))


def bx_harris(x, y):
    return V1 + (V2 - V1) * (_S(y, 0.3 * Ly, L) - _S(y, 0.7 * Ly, L)) + 0.0 * x


def by_harris(x, y):
    return 0.0 * x + 0.0 * y


def config(diag_dir):
    common = dict(
        time_step=time_step,
        time_step_nbr=NSTEPS,
        cells=cells,
        dl=dl,
        boundary_types=["periodic", "periodic"],
        interp_order=1,
        max_mhd_level=NLEVELS,
        refinement_order=ORDER,
        strict=True,
        nesting_buffer=1,
        gamma=5.0 / 3.0,
        eta=0.0,
        nu=0.0,
        resistivity=0.0,
        hyper_resistivity=0.0,
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="SSPRK4_5",
        hall=True,
        hyper_res=False,
        model_options=["MHDModel"],
        diag_options={
            "format": "phareh5",
            "options": {"dir": diag_dir, "mode": "overwrite"},
        },
    )
    if ORDER != 0:
        common["refinement_limiter"] = LIMITER
    sim = ph.Simulation(
        refinement="tagging",
        max_nbr_levels=NLEVELS,
        tag_buffer=1,
        **common,
    )

    def bz(x, y):
        return 0.0 * x

    def density(x, y):
        return (
            0.4
            + 1.0 / np.cosh((y - 0.3 * Ly) / L) ** 2
            + 1.0 / np.cosh((y - 0.7 * Ly) / L) ** 2
        )

    def pressure(x, y):
        b2 = bx_harris(x, y) ** 2 + by_harris(x, y) ** 2 + bz(x, y) ** 2
        p = K - 0.5 * b2
        assert np.all(p > 0)
        return p

    def zero(x, y):
        return 0.0 * x

    ph.MHDModel(
        density=density,
        vx=zero,
        vy=zero,
        vz=zero,
        bx=bx_harris,
        by=by_harris,
        bz=bz,
        p=pressure,
    )
    # write at BOTH t=0 (verify init levels) and after the single advance
    ph.ElectromagDiagnostics(
        quantity="B", write_timestamps=np.array([0.0, NSTEPS * time_step])
    )
    return sim


if __name__ == "__main__":
    diag_dir = "divb_5level_tagging"
    if cpp.mpi_rank() == 0:
        print(
            f"=== tagging single-advance: order={ORDER} levels(max)={NLEVELS} "
            f"nesting_buffer=1 ===",
            flush=True,
        )
    sim = config(diag_dir)
    Simulator(sim).run().reset()
    if cpp.mpi_rank() == 0:
        run = Run(diag_dir)
        B = run.GetB(0.0, all_primal=False)
        lvls = sorted(B.levels(0.0).keys())
        print(f"INIT levels present (t=0): {lvls}", flush=True)
        print("SURVIVED single advance", flush=True)
