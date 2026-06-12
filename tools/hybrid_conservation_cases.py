#!/usr/bin/env python3
"""Hybrid-only conservation baseline cases — phase 8 follow-up.

Two cases, selected by argv[1]:
  uniform — periodic uniform plasma, single level. Deposited mass should be
            exactly conserved (shape-function partition of unity).
  amr     — harris-like, tagging, 2 levels; same grid/dt as the coupled
            mhd_hybrid_harris_2d gate. Quantifies drift from particle
            splitting + level dynamics alone (no MHD coupling).

Dumps ions_mass_density / ions_bulkVelocity (moment cadence) and EM_B
(every step, for regrid detection). Analyse with:
  python tools/conservation_scan.py <diag_dir> <out> --mode hybrid

Usage: python tools/hybrid_conservation_cases.py {uniform|amr}
"""

import sys

import numpy as np

import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator, startMPI

ph.NO_GUI()


def uniform_config():
    time_step = 0.01
    time_step_nbr = 200
    sim = ph.Simulation(
        smallest_patch_size=20,
        largest_patch_size=20,
        time_step=time_step,
        time_step_nbr=time_step_nbr,
        boundary_types=("periodic", "periodic"),
        cells=(40, 40),
        dl=(0.2, 0.2),
        interp_order=1,  # match kaa coupled validation run
        diag_options={
            "format": "phareh5",
            "options": {"dir": "phare_outputs/hybrid_conserv_uniform", "mode": "overwrite"},
        },
        strict=True,
    )

    def density(x, y):
        return 1.0

    def bx(x, y):
        return 1.0

    def by(x, y):
        return 0.0

    def bz(x, y):
        return 0.0

    def v0(x, y):
        return 0.0

    def vth(x, y):
        return 0.3

    ph.MaxwellianFluidModel(
        bx=bx,
        by=by,
        bz=bz,
        protons={
            "charge": 1,
            "density": density,
            "vbulkx": v0,
            "vbulky": v0,
            "vbulkz": v0,
            "vthx": vth,
            "vthy": vth,
            "vthz": vth,
            "nbr_part_per_cell": 100,
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.1)

    final_time = time_step * time_step_nbr
    moments = [round(i * 10 * time_step, 10) for i in range(time_step_nbr // 10 + 1)]
    b_times = [round(i * 10 * time_step, 10) for i in range(time_step_nbr // 10 + 1)]
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=b_times)
    ph.FluidDiagnostics(quantity="mass_density", write_timestamps=moments)
    ph.FluidDiagnostics(quantity="bulkVelocity", write_timestamps=moments)
    return sim


def amr_config():
    # hybrid analog of tests/simulator/mhd/harris_2d_amr/case.py — same grid,
    # dt, level count and duration, so the pure-MHD conservation bar
    # (mass 1e-5 / energy 1e-4) is directly comparable, and a coupled rerun
    # on this setup later closes the triangle.
    time_step = 0.005
    nsteps = 6
    sim = ph.Simulation(
        smallest_patch_size=8,
        largest_patch_size=16,
        time_step=time_step,
        time_step_nbr=nsteps,
        cells=(40, 40),
        dl=(0.40, 0.40),
        interp_order=1,  # match kaa coupled validation run (interp 1)
        refinement="tagging",
        max_nbr_levels=2,
        hyper_resistivity=0.002,
        resistivity=0.0,
        nesting_buffer=1,
        diag_options={
            "format": "phareh5",
            "options": {"dir": "phare_outputs/hybrid_conserv_amr", "mode": "overwrite"},
        },
        strict=False,
    )

    L = 0.5

    def S(y, y0, l):
        return 0.5 * (1.0 + np.tanh((y - y0) / l))

    def density(x, y):
        Ly = sim.simulation_domain()[1]
        return (
            0.4
            + 1.0 / np.cosh((y - Ly * 0.3) / L) ** 2
            + 1.0 / np.cosh((y - Ly * 0.7) / L) ** 2
        )

    def bx(x, y):
        Lx, Ly = sim.simulation_domain()
        sigma, dB = 1.0, 0.1
        x0 = x - 0.5 * Lx
        y1 = y - 0.3 * Ly
        y2 = y - 0.7 * Ly
        dBx1 = -2 * dB * y1 * np.exp(-(x0**2 + y1**2) / sigma**2)
        dBx2 = 2 * dB * y2 * np.exp(-(x0**2 + y2**2) / sigma**2)
        v1, v2 = -1.0, 1.0
        return v1 + (v2 - v1) * (S(y, Ly * 0.3, L) - S(y, Ly * 0.7, L)) + dBx1 + dBx2

    def by(x, y):
        Lx, Ly = sim.simulation_domain()
        sigma, dB = 1.0, 0.1
        x0 = x - 0.5 * Lx
        y1 = y - 0.3 * Ly
        y2 = y - 0.7 * Ly
        dBy1 = 2 * dB * x0 * np.exp(-(x0**2 + y1**2) / sigma**2)
        dBy2 = -2 * dB * x0 * np.exp(-(x0**2 + y2**2) / sigma**2)
        return dBy1 + dBy2

    def bz(x, y):
        return 0.0

    def b2(x, y):
        return bx(x, y) ** 2 + by(x, y) ** 2 + bz(x, y) ** 2

    def T(x, y):
        K = 0.7
        temp = 1.0 / density(x, y) * (K - b2(x, y) * 0.5)
        assert np.all(temp > 0)
        return temp

    def v0(x, y):
        return 0.0

    def vth(x, y):
        return np.sqrt(T(x, y))

    ph.MaxwellianFluidModel(
        bx=bx,
        by=by,
        bz=bz,
        protons={
            "charge": 1,
            "density": density,
            "vbulkx": v0,
            "vbulky": v0,
            "vbulkz": v0,
            "vthx": vth,
            "vthy": vth,
            "vthz": vth,
            "nbr_part_per_cell": 100,
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.1)

    moments = [round(i * time_step, 10) for i in range(nsteps + 1)]
    b_times = moments  # every step: regrid detection
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=b_times)
    ph.FluidDiagnostics(quantity="mass_density", write_timestamps=moments)
    ph.FluidDiagnostics(quantity="bulkVelocity", write_timestamps=moments)
    return sim


def coupled_config():
    # MHD L0 + hybrid L1 on the EXACT amr_config() setup (grid, dt, steps,
    # interp, B field, particle count) so the 3-way comparison is fair:
    # MHD bar (1e-5) / hybrid AMR baseline / coupled.
    time_step = 0.005
    nsteps = 6
    sim = ph.Simulation(
        smallest_patch_size=8,
        largest_patch_size=16,
        time_step=time_step,
        time_step_nbr=nsteps,
        cells=(40, 40),
        dl=(0.40, 0.40),
        interp_order=1,
        refinement="tagging",
        max_mhd_level=1,
        max_nbr_levels=2,
        hyper_resistivity=0.002,
        resistivity=0.0,
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
        diag_options={
            "format": "phareh5",
            "options": {"dir": "phare_outputs/hybrid_conserv_coupled", "mode": "overwrite"},
        },
        strict=False,
    )

    L = 0.5

    def S(y, y0, l):
        return 0.5 * (1.0 + np.tanh((y - y0) / l))

    def density(x, y):
        Ly = sim.simulation_domain()[1]
        return (
            0.4
            + 1.0 / np.cosh((y - Ly * 0.3) / L) ** 2
            + 1.0 / np.cosh((y - Ly * 0.7) / L) ** 2
        )

    def bx(x, y):
        Lx, Ly = sim.simulation_domain()
        sigma, dB = 1.0, 0.1
        x0 = x - 0.5 * Lx
        y1 = y - 0.3 * Ly
        y2 = y - 0.7 * Ly
        dBx1 = -2 * dB * y1 * np.exp(-(x0**2 + y1**2) / sigma**2)
        dBx2 = 2 * dB * y2 * np.exp(-(x0**2 + y2**2) / sigma**2)
        v1, v2 = -1.0, 1.0
        return v1 + (v2 - v1) * (S(y, Ly * 0.3, L) - S(y, Ly * 0.7, L)) + dBx1 + dBx2

    def by(x, y):
        Lx, Ly = sim.simulation_domain()
        sigma, dB = 1.0, 0.1
        x0 = x - 0.5 * Lx
        y1 = y - 0.3 * Ly
        y2 = y - 0.7 * Ly
        dBy1 = 2 * dB * x0 * np.exp(-(x0**2 + y1**2) / sigma**2)
        dBy2 = -2 * dB * x0 * np.exp(-(x0**2 + y2**2) / sigma**2)
        return dBy1 + dBy2

    def bz(x, y):
        return 0.0

    def v0(x, y):
        return 0.0

    Te = 0.1

    def p(x, y):
        # total thermal pressure matching amr_config(): ion Pi = K - b2/2
        # (K=0.7) plus isothermal electron Pe = n*Te
        b2 = bx(x, y) ** 2 + by(x, y) ** 2 + bz(x, y) ** 2
        return 0.7 - b2 * 0.5 + Te * density(x, y)

    ph.MHDModel(
        density=density,
        vx=v0,
        vy=v0,
        vz=v0,
        bx=bx,
        by=by,
        bz=bz,
        p=p,
        protons={
            "charge": 1,
            "mass": 1,
            "nbr_part_per_cell": 100,
            "init": {"seed": 1333},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=Te)

    moments = [round(i * time_step, 10) for i in range(nsteps + 1)]
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=moments)
    for quantity in ["rho", "V", "P"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=moments)
    ph.FluidDiagnostics(quantity="mass_density", write_timestamps=moments)
    ph.FluidDiagnostics(quantity="bulkVelocity", write_timestamps=moments)
    return sim


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "uniform"
    startMPI()
    config = {"uniform": uniform_config, "amr": amr_config, "coupled": coupled_config}[case]
    Simulator(config()).run()


if __name__ == "__main__":
    main()
