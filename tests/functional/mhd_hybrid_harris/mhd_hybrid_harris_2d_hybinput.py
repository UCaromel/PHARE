#!/usr/bin/env python3
"""
2D MHD-Hybrid coupled Harris run with the SAME physical inputs as the
default hybrid Harris case (tests/functional/harris/harris_2d.py):
same domain (200x100, dl=0.4), same density profile, same B (dB=0.1 island
perturbation), same ion pressure (Pi = 0.7 - b^2/2, i.e. T = (K - b2/2)/n
with K=0.7), Te=0, same nbrPPC=100 and seed, same final_time=50.

Coupled numerics kept from mhd_hybrid_harris_2d.py (MHD L0, hybrid L1,
WENOZ/Rusanov/TVDRK3, hall + spatial hyper-resistivity nu=0.02).
interp_order=1 matches both the hybrid script default and the kaa build
permutation (2,1,4,TVDRK3,WENOZ,None,Rusanov,true,false,true).

No divB gate: the point-sampled island perturbation carries an
O(dx^2 * dB) divB floor (~1.4e-3) that CT preserves — the absolute 5e-6
gate of the smoke test only applies to the dB=0 init.

Plain script (no SimulatorTest — its tearDown deletes the diag dir).
"""

import numpy as np

import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator, startMPI

ph.NO_GUI()

cells = (200, 100)
time_step = 0.005
final_time = 50

heavy_timestamps = [0, final_time / 2, final_time]
b_timestamps = [round(i * 1.0, 10) for i in range(int(final_time) + 1)]
diag_dir = "phare_outputs/mhd_hybrid_harris_2d_hybinput"


def config():
    L = 0.5

    sim = ph.Simulation(
        time_step=time_step,
        final_time=final_time,
        cells=cells,
        dl=(0.40, 0.40),
        refinement="tagging",
        max_mhd_level=1,
        max_nbr_levels=2,
        # WENOZ MHD reconstruction carries 6 ghosts -> smallest_patch_size > 6
        smallest_patch_size=10,
        interp_order=1,
        hyper_resistivity=0.002,
        resistivity=0.001,
        diag_options={
            "format": "phareh5",
            "options": {"dir": diag_dir, "mode": "overwrite"},
        },
        strict=True,
        nesting_buffer=1,
        hyper_mode="spatial",
        eta=0.0,
        nu=0.02,
        gamma=5.0 / 3.0,
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="TVDRK3",
        hall=True,
        res=False,
        hyper_res=True,
        model_options=["MHDModel", "HybridModel"],
    )

    def S(y, y0, l):
        return 0.5 * (1.0 + np.tanh((y - y0) / l))

    def density(x, y):
        Ly = sim.simulation_domain()[1]
        return (
            0.4
            + 1.0 / np.cosh((y - Ly * 0.3) / L) ** 2
            + 1.0 / np.cosh((y - Ly * 0.7) / L) ** 2
        )

    def vx(x, y):
        return 0.0

    def vy(x, y):
        return 0.0

    def vz(x, y):
        return 0.0

    def bx(x, y):
        Lx = sim.simulation_domain()[0]
        Ly = sim.simulation_domain()[1]
        sigma = 1.0
        dB = 0.1

        x0 = x - 0.5 * Lx
        y1 = y - 0.3 * Ly
        y2 = y - 0.7 * Ly

        dBx1 = -2 * dB * y1 * np.exp(-(x0**2 + y1**2) / sigma**2)
        dBx2 = 2 * dB * y2 * np.exp(-(x0**2 + y2**2) / sigma**2)

        v1 = -1.0
        v2 = 1.0
        return v1 + (v2 - v1) * (S(y, Ly * 0.3, L) - S(y, Ly * 0.7, L)) + dBx1 + dBx2

    def by(x, y):
        Lx = sim.simulation_domain()[0]
        Ly = sim.simulation_domain()[1]
        sigma = 1.0
        dB = 0.1

        x0 = x - 0.5 * Lx
        y1 = y - 0.3 * Ly
        y2 = y - 0.7 * Ly

        dBy1 = 2 * dB * x0 * np.exp(-(x0**2 + y1**2) / sigma**2)
        dBy2 = -2 * dB * x0 * np.exp(-(x0**2 + y2**2) / sigma**2)

        return dBy1 + dBy2

    def bz(x, y):
        return 0.0

    def p(x, y):
        # ion pressure of the hybrid case: n*T with T = (K - b2/2)/n, K=0.7;
        # Te=0 so MHD total P == Pi
        b2 = bx(x, y) ** 2 + by(x, y) ** 2 + bz(x, y) ** 2
        return 0.7 - b2 * 0.5

    ph.MHDModel(
        density=density,
        vx=vx,
        vy=vy,
        vz=vz,
        bx=bx,
        by=by,
        bz=bz,
        p=p,
        protons={
            "charge": 1,
            "mass": 1,
            "nbr_part_per_cell": 100,
            "init": {"seed": 12334},
        },
    )

    ph.ElectronModel(closure="isothermal", Te=0.0)

    ph.ElectromagDiagnostics(quantity="B", write_timestamps=b_timestamps)
    ph.ElectromagDiagnostics(quantity="E", write_timestamps=heavy_timestamps)

    for quantity in ["rho", "V", "P"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=heavy_timestamps)

    ph.FluidDiagnostics(quantity="mass_density", write_timestamps=heavy_timestamps)
    ph.FluidDiagnostics(quantity="bulkVelocity", write_timestamps=heavy_timestamps)

    return sim


if __name__ == "__main__":
    startMPI()
    Simulator(config()).run()
