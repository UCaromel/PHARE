#!/usr/bin/env python3
# Minimal reproduction: high-order (order-4, no limiting) MHD refiner crash on the FIRST
# advance at high level count. 5 nested static boxes -> levels 0..4 exist at init; we run a
# SINGLE advance and watch for the crash. Same MHD solver / numerics as divb_refinement.py
# (WENOZ / Rusanov / SSPRK4_5 / Hall, interp_order=1), same Harris Bx(y)-only div-free init.
#
# Run: mpirun --oversubscribe -n 4 python3 -u divb_5level_crash.py

import sys

import numpy as np

import pyphare.pharein as ph
from pyphare import cpp
from pyphare.simulator.simulator import Simulator, startMPI

ph.NO_GUI()
startMPI()

# Coarse grid + Harris sheet (identical physics to divb_refinement.py).
cells = (20, 80)
dl = (0.5, 0.5)
Lx, Ly = cells[0] * dl[0], cells[1] * dl[1]
L = 0.5
V1, V2 = -1.0, 1.0
K = 0.7

time_step = 0.002
NSTEPS = 1  # single advance -- the crash is reported on the first advance

# CLI: divb_5level_crash.py [nlevels] [order]   (defaults 5 4)
NLEVELS = int(sys.argv[1]) if len(sys.argv) > 1 else 5  # levels 0..NLEVELS-1
ORDER = int(sys.argv[2]) if len(sys.argv) > 2 else 4
HALF = int(sys.argv[3]) if len(sys.argv) > 3 else 4  # box half-width (coarse cells) = nesting margin
LIMITER = "none"


def _S(y, y0, l):
    return 0.5 * (1.0 + np.tanh((y - y0) / l))


def bx_harris(x, y):
    return V1 + (V2 - V1) * (_S(y, 0.3 * Ly, L) - _S(y, 0.7 * Ly, L)) + 0.0 * x


def by_harris(x, y):
    return 0.0 * x + 0.0 * y


# Nested boxes centered on the lower sheet (L0 cell y=24, x=10). Half-widths are constant in
# cells, so box_{n+1} (index 2x) sits comfortably inside 2*box_n -> nesting is automatic.
def nested_boxes(nlevels, hx=4, hy=6, cx0=10, cy0=24):
    boxes = {}
    for n in range(nlevels - 1):  # boxes on L0..L(n-2) create L1..L(n-1)
        f = 2 ** n
        cx, cy = cx0 * f, cy0 * f
        boxes[f"L{n}"] = {"B0": [[cx - hx, cy - hy], [cx + hx, cy + hy]]}
    return boxes


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
        nesting_buffer=0,
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
        refinement="boxes",
        refinement_boxes=nested_boxes(NLEVELS, hx=HALF, hy=HALF),
        smallest_patch_size=6,
        largest_patch_size=20,
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
    ph.ElectromagDiagnostics(
        quantity="B", write_timestamps=np.array([NSTEPS * time_step])
    )
    return sim


if __name__ == "__main__":
    diag_dir = "divb_5level_crash"
    if cpp.mpi_rank() == 0:
        print(
            f"=== 5-level single-advance: order={ORDER} limiter={LIMITER} "
            f"levels={NLEVELS} ===",
            flush=True,
        )
    sim = config(diag_dir)
    if cpp.mpi_rank() == 0:
        print(f"refinement_boxes={nested_boxes(NLEVELS, hx=HALF, hy=HALF)}", flush=True)
    Simulator(sim).run().reset()
    if cpp.mpi_rank() == 0:
        print("SURVIVED single advance with 5 levels", flush=True)
