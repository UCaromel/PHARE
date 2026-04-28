import unittest
from copy import deepcopy

from pyphare.simulator.simulator import Simulator, startMPI
from pyphare.pharesee.run import Run
from pyphare.pharesee.hierarchy import hierarchy_utils as hootils

from tests.simulator import SimulatorTest
from tests.simulator.test_vtk_diagnostics import mhd_config


class VTKDiagnosticsMHDTest(SimulatorTest):
    def _run_mhd(self, simInput):
        simulation = mhd_config(self.simulation(**simInput))
        self.assertEqual(len(simulation.cells), 2)
        Simulator(simulation).run().reset()
        return simulation.diag_options["options"]["dir"]

    def test_compare_mhd_b_to_phareh5(self):
        sim_input = {
            "time_step": 0.001,
            "final_time": 0.001,
            "interp_order": 2,
            "boundary_types": ["periodic", "periodic"],
            "cells": [20, 20],
            "dl": [0.3, 0.3],
            "diag_options": {
                "format": "pharevtkhdf",
                "options": {"mode": "overwrite", "dir": "phare_outputs/vtk_diagnostic_test"},
            },
            "strict": True,
            "nesting_buffer": 1,
            "hyper_mode": "spatial",
            "eta": 0.0,
            "nu": 0.02,
            "gamma": 5.0 / 3.0,
            "reconstruction": "WENOZ",
            "limiter": "None",
            "riemann": "Rusanov",
            "mhd_timestepper": "TVDRK3",
            "hall": True,
            "res": False,
            "hyper_res": True,
            "model_options": ["MHDModel"],
        }

        vtk_diags = self._run_mhd(deepcopy(sim_input))

        sim_input["diag_options"]["format"] = "phareh5"
        phareh5_diags = self._run_mhd(deepcopy(sim_input))

        eqr = hootils.hierarchy_compare(Run(vtk_diags).GetB(0.001), Run(phareh5_diags).GetB(0.001))
        if not eqr:
            print(eqr)
        self.assertTrue(eqr)


if __name__ == "__main__":
    startMPI()
    unittest.main()
