#!/usr/bin/env python3
# investigation-only: upstream harris config, env-tunable
import os
import numpy as np
import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator, startMPI

ph.NO_GUI()
cells = (150, 75)
time_step = 0.005
LEVELS = int(os.environ.get("MM_LEVELS", "2"))
NSTEPS = int(os.environ.get("MM_STEPS", "50"))
PATCH = int(os.environ.get("MM_PATCH", "40"))
hall = os.environ.get("MM_HALL", "1") == "1"


def config():
    L = 0.5

    sim = ph.Simulation(
        time_step=time_step,
        final_time=NSTEPS * time_step,
        cells=cells,
        dl=(0.40, 0.40),
        refinement="tagging",
        max_mhd_level=LEVELS,
        max_nbr_levels=LEVELS,
        largest_patch_size=PATCH,
        smallest_patch_size=int(os.environ.get("MM_SMALLEST", "8")),
        hyper_resistivity=0.0,
        resistivity=0.0,
        strict=True,
        nesting_buffer=int(os.environ.get("MM_NEST", "1")),
        hyper_mode="spatial",
        eta=0.0,
        nu=0.02,
        gamma=5.0 / 3.0,
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="TVDRK3",
        hall=hall,
        model_options=["MHDModel"],
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

        dBx1 = -2 * dB * y1 * np.exp(-(x0**2 + y1**2) / (sigma) ** 2)
        dBx2 = 2 * dB * y2 * np.exp(-(x0**2 + y2**2) / (sigma) ** 2)

        v1 = -1
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

        dBy1 = 2 * dB * x0 * np.exp(-(x0**2 + y1**2) / (sigma) ** 2)
        dBy2 = -2 * dB * x0 * np.exp(-(x0**2 + y2**2) / (sigma) ** 2)

        return dBy1 + dBy2

    def bz(x, y):
        return 0.0

    def p(x, y):
        return 1.0 - (bx(x, y) ** 2 + by(x, y) ** 2) / 2.0

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    return sim


if __name__ == "__main__":
    startMPI()
    Simulator(config()).run().reset()
