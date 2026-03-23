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
    combination_name,
    compare_case_to_reference,
)


os.environ["PHARE_SCOPE_TIMING"] = "1"

ph.NO_GUI()


case_dir = Path(__file__).resolve().parent
case_name = case_dir.name
reference_root = case_dir / "golden_data"
time_step = 0.0005
time_step_nbr = 6
final_time = time_step * time_step_nbr
timestamps = [0.0, final_time]
atol = 1e-11
rtol = 1e-8


def config(
    combination=DEFAULT_COMBINATION,
    diag_dir=f"phare_outputs/simulator/mhd/{case_name}",
):
    cells = (32, 32)
    dl = (0.2, 0.2)

    sim = ph.Simulation(
        smallest_patch_size=20,
        largest_patch_size=32,
        time_step_nbr=time_step_nbr,
        time_step=time_step,
        cells=cells,
        dl=dl,
        interp_order=2,
        refinement_boxes={},
        diag_options={"format": "phareh5", "options": {"dir": diag_dir, "mode": "overwrite"}},
        strict=True,
        nesting_buffer=0,
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

    b0 = 1.0 / np.sqrt(4.0 * np.pi)
    domain = sim.simulation_domain()

    def density(x, y):
        return 25.0 / (36.0 * np.pi)

    def vx(x, y):
        return -np.sin(2.0 * np.pi * y / domain[1])

    def vy(x, y):
        return np.sin(2.0 * np.pi * x / domain[0])

    def vz(x, y):
        return 0.0

    def bx(x, y):
        return -b0 * np.sin(2.0 * np.pi * y / domain[1])

    def by(x, y):
        return b0 * np.sin(4.0 * np.pi * x / domain[0])

    def bz(x, y):
        return 0.0

    def p(x, y):
        return 5.0 / (12.0 * np.pi)

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    for quantity in ["rho", "V", "P"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)

    return sim


class OrszagTang2DTest(SimulatorTest):
    def test_matches_reference(self):
        for combination in MHD_COMBINATIONS:
            with self.subTest(combination=combination_name(combination)):
                compare_case_to_reference(
                    self,
                    case_name=case_name,
                    reference_root=reference_root,
                    config=config,
                    combination=combination,
                    final_time=final_time,
                    atol=atol,
                    rtol=rtol,
                )
        return self


def main():
    Simulator(config()).run()


if __name__ == "__main__":
    startMPI()
    OrszagTang2DTest().test_matches_reference().tearDown()
