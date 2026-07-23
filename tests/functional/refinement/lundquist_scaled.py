#!/usr/bin/env python3
# Scaled-down reproduction of lundquist_run166.py (adastra 5-level NaN-on-first-advance).
# IDENTICAL config/init to the production run, only the Lundquist number SL is reduced so the
# L0 grid is tiny (nx~32) and it runs locally. Everything else faithful:
#   refinement="tagging", 5 levels, nesting_buffer=1, interp_order=2, legacy refinement
#   (no refinement_order set), WENOZ/Rusanov/SSPRK4_5, hall=False, res=True.
# Run only a few steps -- the crash is reported on the FIRST advance.
import os
import sys

import numpy as np

import pyphare.pharein as ph
from pyphare import cpp
from pyphare.simulator.simulator import Simulator, startMPI

from tests.simulator import SimulatorTest

ph.NO_GUI()

# SL controls grid size: nx = round(round(10*sqrt(SL))/16). SL=2621 -> nx=32.
SL = float(sys.argv[1]) if len(sys.argv) > 1 else 2621.0
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

L_sys = 1.0
va = 1.0
n_levels = 5
refine_ratio = 2
refine_factor = refine_ratio ** (n_levels - 1)

eta_val = (L_sys * va) / SL
eps_SP = 1.0 / np.sqrt(SL)
delta_SP = L_sys * eps_SP
dl_target = 0.1 * delta_SP
nx_fine = int(round(L_sys / dl_target))
nx = int(round(nx_fine / refine_factor))
dl_val = L_sys / nx
dt_val = 0.1 * dl_val
L_shear = 1.5 * delta_SP

cells = (nx, nx)
time_step = dt_val
final_time = NSTEPS * dt_val

diag_dir = "phare_outputs/lundquist_scaled"


def config():
    L = L_shear
    sim = ph.Simulation(
        time_step=time_step,
        final_time=final_time,
        cells=cells,
        dl=(dl_val, dl_val),
        interp_order=2,
        refinement="tagging",
        max_mhd_level=n_levels,
        max_nbr_levels=n_levels,
        hyper_resistivity=0.0,
        resistivity=0.0,
        strict=True,
        nesting_buffer=1,
        hyper_mode="spatial",
        eta=eta_val,
        nu=0.0,
        gamma=5.0 / 3.0,
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="SSPRK4_5",
        hall=False,
        res=True,
        hyper_res=False,
        model_options=["MHDModel"],
    )
    sim = ph.global_vars.sim

    N_modes = int(1.0 / (10 * sim.dl[1]))
    np.random.seed(0)
    phases = np.random.uniform(0, 2 * np.pi, N_modes)
    modes = np.arange(1, N_modes + 1)

    def S(y, y0, l):
        return 0.5 * (1.0 + np.tanh((y - y0) / l))

    def vx(x, y):
        return 0.0

    def vy(x, y):
        return 0.0

    def vz(x, y):
        return 0.0

    def bx(x, y):
        Lx = sim.simulation_domain()[0]
        Ly = sim.simulation_domain()[1]
        psi0 = 0.001
        y1 = 0.25 * Ly
        y2 = 0.75 * Ly
        kx = 2 * np.pi / Lx
        ky = 2 * np.pi / Ly
        term1 = ky * np.sin(ky * (y - y1))
        term2 = -ky * np.sin(ky * (y - y2))
        dBx = psi0 * np.cos(kx * x) * (term1 + term2)
        v1 = -1.0
        v2 = 1.0
        return v1 + (v2 - v1) * (S(y, y1, L) - S(y, y2, L)) + dBx

    def by(x, y):
        Lx = sim.simulation_domain()[0]
        Ly = sim.simulation_domain()[1]
        psi0 = 0.001
        y1 = 0.25 * Ly
        y2 = 0.75 * Ly
        kx = 2 * np.pi / Lx
        ky = 2 * np.pi / Ly
        term1 = np.cos(ky * (y - y1))
        term2 = -np.cos(ky * (y - y2))
        dBy = -psi0 * kx * np.sin(kx * x) * (term1 + term2)
        return dBy

    def bz(x, y):
        return 0.0

    def p(x, y):
        return 6.0 - (bx(x, y) ** 2 + by(x, y) ** 2) / 2.0

    def density(x, y):
        Lx = sim.simulation_domain()[0]
        kx = 2 * np.pi / Lx
        return p(x, y) / 6.0 + sum(
            np.sin(kx * x * m + phi) * 1e-8 * kx * m
            for m, phi in zip(modes, phases)
        )

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)
    return sim


class HarrisTest(SimulatorTest):
    def __init__(self, *args, **kwargs):
        super(HarrisTest, self).__init__(*args, **kwargs)
        self.simulator = None

    def tearDown(self):
        super(HarrisTest, self).tearDown()
        if self.simulator is not None:
            self.simulator.reset()
        self.simulator = None
        ph.global_vars.sim = None

    def test_run(self):
        if cpp.mpi_rank() == 0:
            print(
                f"=== lundquist_scaled SL={SL} nx={nx} dl={dl_val:.5f} dt={dt_val:.6f} "
                f"L_shear={L_shear:.5f} eta={eta_val:.3e} steps={NSTEPS} ===",
                flush=True,
            )
        Simulator(config()).run().reset()
        cpp.mpi_barrier()
        if cpp.mpi_rank() == 0:
            print("SURVIVED", flush=True)
        return self


if __name__ == "__main__":
    startMPI()
    HarrisTest().test_run().tearDown()
