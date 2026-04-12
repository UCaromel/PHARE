#!/usr/bin/env python3
"""
Smoke test: 2D MHD-Hybrid coupling, Harris current sheet.

MHD on level 0, Hybrid (PIC) on level 1.
Verifies initialization + a handful of timesteps run without crash or NaN.
"""

import numpy as np
from pathlib import Path

from pyphare import cpp
import pyphare.pharein as ph
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator, startMPI

from tests.simulator import SimulatorTest

ph.NO_GUI()

cells = (80, 40)
time_step = 0.005
final_time = 0.05
timestamps = [0, final_time]
diag_dir = "phare_outputs/mhd_hybrid_harris_2d"

hall = True
res = False
hyper_res = True

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
        interp_order=2,
        hyper_resistivity=0.002,
        resistivity=0.0,
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
        hall=hall,
        res=res,
        hyper_res=hyper_res,
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
        return 1.0 - (bx(x, y) ** 2 + by(x, y) ** 2) / 2.0

    # MHDModel carries the proton population spec for the Hybrid level.
    # Particle density/velocities/temperatures are placeholders here;
    # actual injection uses MHD state at runtime (postprocessRefine).
    ph.MHDModel(
        density=density,
        vx=vx,
        vy=vy,
        vz=vz,
        bx=bx,
        by=by,
        bz=bz,
        p=p,
        protons={"charge": 1, "mass": 1, "nbr_part_per_cell": 10},
    )

    ph.ElectronModel(closure="isothermal", Te=0.0)

    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    ph.ElectromagDiagnostics(quantity="E", write_timestamps=timestamps)

    for quantity in ["rho", "V", "P"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)

    ph.FluidDiagnostics(quantity="mass_density", write_timestamps=timestamps)

    return sim


class MHDHybridHarrisTest(SimulatorTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulator = None

    def tearDown(self):
        super().tearDown()
        if self.simulator is not None:
            self.simulator.reset()
        self.simulator = None
        ph.global_vars.sim = None

    def test_run(self):
        self.register_diag_dir_for_cleanup(diag_dir)
        Simulator(config()).run().reset()
        if cpp.mpi_rank() == 0:
            # Verify the Hybrid level was actually created and has B data.
            # If tagging never fires, coupling code is untested — fail loudly.
            b_hier = Run(diag_dir).GetB(final_time)
            levels = b_hier.levels(final_time)
            assert 1 in levels, (
                "Hybrid level 1 not found in B diagnostics — "
                "MHD tagger may not have fired on the initial Harris profile"
            )
        cpp.mpi_barrier()
        return self


if __name__ == "__main__":
    startMPI()
    MHDHybridHarrisTest().test_run().tearDown()
