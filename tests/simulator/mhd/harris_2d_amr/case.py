#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np

import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator, startMPI

from tests.simulator import SimulatorTest
from tests.simulator.mhd.test_mhd_tools import (
    check_mhd_conservation,
    combination_name,
)


os.environ["PHARE_SCOPE_TIMING"] = "1"

ph.NO_GUI()


case_dir = Path(__file__).resolve().parent
case_name = case_dir.name
time_step = 0.005
time_step_nbr = 6
final_time = time_step * time_step_nbr
timestamps = [0.0, final_time]

HARRIS_2D_AMR_COMBINATION = {
    "mhd_timestepper": "TVDRK3",
    "reconstruction": "WENOZ",
    "limiter": "None",
    "riemann": "Rusanov",
    "hall": True,
    "res": False,
    "hyper_res": True,
}
COMBINATIONS = (HARRIS_2D_AMR_COMBINATION,)


def config(
    combination=HARRIS_2D_AMR_COMBINATION,
    diag_dir=f"phare_outputs/simulator/mhd/{case_name}",
):
    cells = (20, 20)
    dl = (0.4, 0.4)

    sim = ph.Simulation(
        smallest_patch_size=8,
        largest_patch_size=16,
        time_step_nbr=time_step_nbr,
        time_step=time_step,
        cells=cells,
        dl=dl,
        interp_order=2,
        refinement="tagging",
        max_nbr_levels=3,
        max_mhd_level=3,
        diag_options={"format": "phareh5", "options": {"dir": diag_dir, "mode": "overwrite"}},
        strict=True,
        nesting_buffer=1,
        hyper_mode="spatial",
        eta=0.0,
        nu=0.02,
        gamma=5.0 / 3.0,
        reconstruction=combination["reconstruction"],
        limiter=combination["limiter"],
        riemann=combination["riemann"],
        mhd_timestepper=combination["mhd_timestepper"],
        hall=combination["hall"],
        res=combination["res"],
        hyper_res=combination["hyper_res"],
        model_options=["MHDModel"],
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
        dBy1 = 2 * dB * x0 * np.exp(-(x0**2 + y1**2) / sigma**2)
        dBy2 = -2 * dB * x0 * np.exp(-(x0**2 + y2**2) / sigma**2)
        return dBy1 + dBy2

    def bz(x, y):
        return 0.0

    def p(x, y):
        return 1.0 - (bx(x, y) ** 2 + by(x, y) ** 2) / 2.0

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    for quantity in ["rho", "Etot"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)

    return sim


class Harris2DAMRTest(SimulatorTest):
    def test_conservation(self):
        for combination in COMBINATIONS:
            with self.subTest(combination=combination_name(combination)):
                check_mhd_conservation(
                    self,
                    case_name=case_name,
                    config=config,
                    combination=combination,
                    final_time=final_time,
                    mass_rtol=1e-5,
                    energy_rtol=1e-4,
                    check_divB=False,
                )
        return self


def main():
    Simulator(config()).run()


if __name__ == "__main__":
    startMPI()
    Harris2DAMRTest().test_conservation().tearDown()
