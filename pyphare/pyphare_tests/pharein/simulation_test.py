import unittest
import numpy as np
from unittest.mock import patch


import pyphare.pharein.global_vars as global_vars

from pyphare.core import phare_utilities
from pyphare.pharein import simulation
from pyphare import cpp


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.cells_array = [80, (80, 40), (80, 40, 12)]
        self.dl_array = [0.1, (0.1, 0.2), (0.1, 0.2, 0.3)]
        self.domain_size_array = [100.0, (100.0, 80.0), (100.0, 80.0, 20.0)]
        self.ndim = [1, 2]  # TODO https://github.com/PHAREHUB/PHARE/issues/232
        self.bcs = [
            "periodic",
            ("periodic", "periodic"),
            ("periodic", "periodic", "periodic"),
        ]
        self.layout = "yee"
        self.time_step = 0.001
        self.time_step_nbr = 1000
        self.final_time = 1.0
        global_vars.sim = None

    def mhd_kwargs(self, **overrides):
        kwargs = {
            "cells": 80,
            "dl": 0.1,
            "time_step_nbr": 1,
            "time_step": 0.01,
            "model_options": "MHDModel",
            "reconstruction": "WENOZ",
            "limiter": "None",
            "riemann": "Rusanov",
            "mhd_timestepper": "TVDRK3",
        }
        kwargs.update(overrides)
        return kwargs

    def assert_invalid_before_import(self, expected, **overrides):
        with patch("pyphare.cpp.importlib.import_module") as import_module:
            with self.assertRaisesRegex(ValueError, expected):
                simulation.Simulation(**self.mhd_kwargs(**overrides))
            import_module.assert_not_called()

    def test_dl(self):
        for cells, domain_size, dim, bc in zip(
            self.cells_array, self.domain_size_array, self.ndim, self.bcs
        ):
            j = simulation.Simulation(
                time_step_nbr=self.time_step_nbr,
                boundary_types=bc,
                cells=cells,
                domain_size=domain_size,
                final_time=self.final_time,
            )

            if phare_utilities.none_iterable(domain_size, cells):
                domain_size = phare_utilities.listify(domain_size)
                cells = phare_utilities.listify(cells)

            for d in np.arange(dim):
                self.assertEqual(j.dl[d], domain_size[d] / float(cells[d]))

            global_vars.sim = None

    def test_boundary_conditions(self):
        j = simulation.Simulation(
            time_step_nbr=1000,
            boundary_types="periodic",
            cells=80,
            domain_size=10,
            final_time=1.0,
        )

        for d in np.arange(j.ndim):
            self.assertEqual("periodic", j.boundary_types[d])

    def test_assert_boundary_condition(self):
        simulation.Simulation(
            time_step_nbr=1000,
            boundary_types="periodic",
            cells=80,
            domain_size=10,
            final_time=1000,
        )

    def test_time_step(self):
        s = simulation.Simulation(
            time_step_nbr=1000,
            boundary_types="periodic",
            cells=80,
            domain_size=10,
            final_time=10,
        )
        self.assertEqual(0.01, s.time_step)


    def test_mhd_order_defaults_to_o2_module_identity(self):
        sim = simulation.Simulation(**self.mhd_kwargs())
        self.assertEqual(sim.mhd_order, 2)
        self.assertEqual(
            cpp.simulator_id(sim),
            "1_O2_TVDRK3_WENOZ_None_Rusanov_false_false_false",
        )

    def test_mhd_module_identity_ignores_explicit_nonzero_interp_order(self):
        # check_interp_order only rejects an explicit interp_order against
        # valid_interp_orders; it does not require HybridModel to be present,
        # so a pure-MHD sim can legitimately end up with a nonzero
        # sim.interp_order. simulator_id() must still key off model_options,
        # not interp_order's truthiness, so the module identity is unaffected.
        sim = simulation.Simulation(**self.mhd_kwargs(interp_order=2))
        self.assertEqual(sim.model_options, ["MHDModel"])
        self.assertEqual(sim.interp_order, 2)
        self.assertEqual(
            cpp.simulator_id(sim),
            "1_O2_TVDRK3_WENOZ_None_Rusanov_false_false_false",
        )

    def test_hybrid_module_identity_unaffected_by_mhd_gate(self):
        sim = simulation.Simulation(
            time_step_nbr=self.time_step_nbr,
            boundary_types="periodic",
            cells=80,
            domain_size=10,
            final_time=1.0,
        )
        self.assertEqual(sim.model_options, ["HybridModel"])
        self.assertEqual(
            cpp.simulator_id(sim),
            f"{sim.ndim}_{sim.interp_order}_{sim.refined_particle_nbr}",
        )

    def test_mhd_order_accepts_o2_and_o4(self):
        for order, timestepper in ((2, "TVDRK3"), (4, "SSPRK4_5")):
            with self.subTest(order=order):
                global_vars.sim = None
                sim = simulation.Simulation(
                    **self.mhd_kwargs(mhd_order=order, mhd_timestepper=timestepper)
                )
                self.assertEqual(sim.mhd_order, order)

    def test_mhd_requires_reconstruction_and_timestepper(self):
        for key, expected in (
            ("reconstruction", "non-empty reconstruction"),
            ("mhd_timestepper", "non-empty mhd_timestepper"),
        ):
            with self.subTest(key=key):
                self.assert_invalid_before_import(expected, **{key: ""})

    def test_mixed_models_are_rejected(self):
        self.assert_invalid_before_import(
            "mixed MHDModel/HybridModel simulations are unsupported",
            model_options=["MHDModel", "HybridModel"],
        )

    def test_explicit_mhd_refinement_order_is_rejected(self):
        self.assert_invalid_before_import(
            "refinement_order is derived from mhd_order for MHD; omit it",
            refinement_order=2,
        )

    def test_bad_o4_reconstruction_is_rejected(self):
        self.assert_invalid_before_import(
            "MHD4 reconstruction must be WENOZ or MP5",
            mhd_order=4,
            reconstruction="Linear",
        )

    def test_bad_o4_timestepper_is_rejected(self):
        self.assert_invalid_before_import(
            "MHD4 timestepper must be TVDRK3 or SSPRK4_5",
            mhd_order=4,
            mhd_timestepper="TVDRK2",
        )

    def test_hierarchical_o4_tvdrk3_is_rejected(self):
        self.assert_invalid_before_import(
            "MHD4\\+TVDRK3 does not support coarse-fine hierarchies",
            mhd_order=4,
            refinement="tagging",
            max_nbr_levels=2,
        )

    def test_hierarchical_o4_ssprk4_5_is_accepted(self):
        sim = simulation.Simulation(
            **self.mhd_kwargs(
                mhd_order=4,
                mhd_timestepper="SSPRK4_5",
                refinement="tagging",
                max_nbr_levels=2,
            )
        )
        self.assertEqual(sim.mhd_order, 4)

if __name__ == "__main__":
    unittest.main()
