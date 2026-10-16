import unittest
import numpy as np


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

    def assert_invalid(self, expected, **overrides):
        global_vars.sim = None
        with self.assertRaisesRegex(ValueError, expected):
            simulation.Simulation(**self.mhd_kwargs(**overrides))

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

    def test_hall_hlld_requires_hyper_resistivity(self):
        def make(**kwargs):
            global_vars.sim = None
            return simulation.Simulation(
                time_step_nbr=1000,
                boundary_types="periodic",
                cells=80,
                domain_size=10,
                final_time=1.0,
                hall=True,
                **kwargs,
            )

        with self.assertRaises(ValueError):
            make(riemann="HLLD")
        make(riemann="HLLD", nu=0.01)
        make(riemann="Rusanov")
        make(riemann="HLL")

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
        self.assertEqual(0.01, s.time_stepper.time_step)

    def test_mhd_order_defaults_to_o2_module_identity(self):
        sim = simulation.Simulation(**self.mhd_kwargs())
        self.assertEqual(sim.mhd_order, 2)
        self.assertEqual(cpp.simulator_id(sim), "1_O2_WENOZ_None_Rusanov_false")

    def test_mhd_order_accepts_o2_and_o4(self):
        for order, timestepper in ((2, "TVDRK3"), (4, "TVDRK3"), (4, "SSPRK4_5")):
            with self.subTest(order=order, timestepper=timestepper):
                global_vars.sim = None
                sim = simulation.Simulation(
                    **self.mhd_kwargs(mhd_order=order, mhd_timestepper=timestepper)
                )
                self.assertEqual(sim.mhd_order, order)

    def test_mhd_order_must_be_2_or_4(self):
        for order in (3, True, "4"):
            with self.subTest(order=order):
                self.assert_invalid("mhd_order must be 2 or 4", mhd_order=order)

    def test_fourth_order_mixed_models_are_rejected(self):
        self.assert_invalid(
            "mixed MHDModel/HybridModel simulations are second order only",
            model_options=["MHDModel", "HybridModel"],
            interp_order=1,
            mhd_order=4,
        )

    def test_fourth_order_rejects_hyper_resistivity(self):
        self.assert_invalid(
            "does not accept hyper-resistivity", mhd_order=4, hall=True, nu=0.01
        )

    def test_bad_o4_reconstruction_is_rejected(self):
        self.assert_invalid(
            "reconstruction must be WENOZ or MP5", mhd_order=4, reconstruction="Linear"
        )

    def test_bad_o4_timestepper_is_rejected(self):
        self.assert_invalid(
            "timestepper must be TVDRK3 or SSPRK4_5", mhd_order=4, mhd_timestepper="TVDRK2"
        )

    def test_fourth_order_tvdrk3_rejects_refined_levels(self):
        self.assert_invalid(
            "needs SSPRK4_5", mhd_order=4, refinement="tagging", max_nbr_levels=2
        )

    def test_fourth_order_ssprk4_5_accepts_refined_levels(self):
        sim = simulation.Simulation(
            **self.mhd_kwargs(
                mhd_order=4, mhd_timestepper="SSPRK4_5", refinement="tagging", max_nbr_levels=2
            )
        )
        self.assertEqual(sim.mhd_order, 4)

    def test_refinement_order_must_match_the_compiled_order(self):
        self.assert_invalid("must match the compiled order 4", mhd_order=4, refinement_order=2)


if __name__ == "__main__":
    unittest.main()
