#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np

import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator, startMPI

from tests.simulator import SimulatorTest
from tests.simulator.mhd.test_mhd_tools import (
    DEFAULT_COMBINATION,
    MHD_COMBINATIONS,
    check_mhd_conservation,
    combination_name,
)


os.environ["PHARE_SCOPE_TIMING"] = "1"

ph.NO_GUI()


case_dir = Path(__file__).resolve().parent
case_name = case_dir.name
time_step = 0.00025
time_step_nbr = 6
final_time = time_step * time_step_nbr
timestamps = [0.0, final_time]


def config(
    combination=DEFAULT_COMBINATION,
    diag_dir=f"phare_outputs/simulator/mhd/{case_name}",
):
    cells = (32, 32)
    dl = (1.0 / cells[0], 1.0 / cells[1])

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

    B0 = 5.0 / np.sqrt(4.0 * np.pi)
    v0 = 2.0
    r0 = 0.1
    r1 = 0.115

    def _r(x, y):
        return np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)

    def _f(r):
        return (r1 - r) / (r1 - r0)

    def density(x, y):
        r_ = _r(x, y)
        return np.where(r_ <= r0, 10.0, np.where(r_ < r1, 1.0 + 9.0 * _f(r_), 1.0))

    def vx(x, y):
        r_ = _r(x, y)
        f_ = _f(r_)
        return np.where(
            r_ <= r0,
            -v0 * (y - 0.5) / r0,
            np.where(r_ < r1, -f_ * v0 * (y - 0.5) / r_, 0.0),
        )

    def vy(x, y):
        r_ = _r(x, y)
        f_ = _f(r_)
        return np.where(
            r_ <= r0,
            v0 * (x - 0.5) / r0,
            np.where(r_ < r1, f_ * v0 * (x - 0.5) / r_, 0.0),
        )

    def vz(x, y):
        return 0.0

    def bx(x, y):
        return B0

    def by(x, y):
        return 0.0

    def bz(x, y):
        return 0.0

    def p(x, y):
        return 1.0

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    for quantity in ["rho", "Etot"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)

    return sim


class Rotor2DAMRTest(SimulatorTest):
    def test_conservation(self):
        for combination in MHD_COMBINATIONS:
            with self.subTest(combination=combination_name(combination)):
                check_mhd_conservation(
                    self,
                    case_name=case_name,
                    config=config,
                    combination=combination,
                    final_time=final_time,
                )
        return self


def main():
    Simulator(config()).run()


if __name__ == "__main__":
    startMPI()
    Rotor2DAMRTest().test_conservation().tearDown()
