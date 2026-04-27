import unittest
import numpy as np


import pyphare.pharein.global_vars as global_vars

from pyphare.core import phare_utilities
from pyphare.pharein import MHDModel, simulation


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

    def test_mhd_model_accepts_external_magnetic_field(self):
        simulation.Simulation(
            time_step_nbr=1000,
            boundary_types="periodic",
            cells=80,
            domain_size=10,
            final_time=10,
            model_options=["MHDModel"],
        )

        model = MHDModel(
            bx=lambda x: 1.0 + 0.0 * x,
            by=lambda x: 0.0 * x,
            bz=lambda x: 0.0 * x,
            b0x=lambda x: 0.5 + 0.0 * x,
            b0y=lambda x: 0.25 * x,
            b0z=lambda x: -0.25 * x,
        )

        self.assertIn("b0x", model.model_dict)
        self.assertIn("b0y", model.model_dict)
        self.assertIn("b0z", model.model_dict)
        self.assertEqual(model.model_dict["b0x"](1.0), 0.5)
        self.assertEqual(model.model_dict["b0y"](2.0), 0.5)
        self.assertEqual(model.model_dict["b0z"](2.0), -0.5)

    def test_mhd_model_derives_b1_from_total_b_and_b0(self):
        simulation.Simulation(
            time_step_nbr=1000,
            boundary_types="periodic",
            cells=80,
            domain_size=10,
            final_time=10,
            model_options=["MHDModel"],
        )

        model = MHDModel(
            bx=lambda x: 1.0 + 0.0 * x,
            by=lambda x: 0.2 + 0.0 * x,
            bz=lambda x: -0.1 + 0.0 * x,
            b0x=lambda x: 0.3 + 0.0 * x,
            b0y=lambda x: -0.15 + 0.0 * x,
            b0z=lambda x: 0.05 + 0.0 * x,
        )

        self.assertAlmostEqual(model.model_dict["b1x"](1.0), 0.7)
        self.assertAlmostEqual(model.model_dict["b1y"](1.0), 0.35)
        self.assertAlmostEqual(model.model_dict["b1z"](1.0), -0.15)

    def test_mhd_model_derives_total_b_from_b1_and_b0(self):
        simulation.Simulation(
            time_step_nbr=1000,
            boundary_types="periodic",
            cells=80,
            domain_size=10,
            final_time=10,
            model_options=["MHDModel"],
        )

        model = MHDModel(
            b1x=lambda x: 0.7 + 0.0 * x,
            b1y=lambda x: 0.35 + 0.0 * x,
            b1z=lambda x: -0.15 + 0.0 * x,
            b0x=lambda x: 0.3 + 0.0 * x,
            b0y=lambda x: -0.15 + 0.0 * x,
            b0z=lambda x: 0.05 + 0.0 * x,
        )

        self.assertAlmostEqual(model.model_dict["bx"](1.0), 1.0)
        self.assertAlmostEqual(model.model_dict["by"](1.0), 0.2)
        self.assertAlmostEqual(model.model_dict["bz"](1.0), -0.1)

    def test_mhd_model_defaults_total_b_to_b0_when_only_external_magnetic_is_given(self):
        simulation.Simulation(
            time_step_nbr=1000,
            boundary_types="periodic",
            cells=80,
            domain_size=10,
            final_time=10,
            model_options=["MHDModel"],
        )

        model = MHDModel(
            b0x=lambda x: 0.3 + 0.0 * x,
            b0y=lambda x: -0.15 + 0.0 * x,
            b0z=lambda x: 0.05 + 0.0 * x,
        )

        self.assertAlmostEqual(model.model_dict["bx"](1.0), 0.3)
        self.assertAlmostEqual(model.model_dict["by"](1.0), -0.15)
        self.assertAlmostEqual(model.model_dict["bz"](1.0), 0.05)
        self.assertAlmostEqual(model.model_dict["b1x"](1.0), 0.0)
        self.assertAlmostEqual(model.model_dict["b1y"](1.0), 0.0)
        self.assertAlmostEqual(model.model_dict["b1z"](1.0), 0.0)

    def test_mhd_model_rejects_total_b_and_b1_together(self):
        simulation.Simulation(
            time_step_nbr=1000,
            boundary_types="periodic",
            cells=80,
            domain_size=10,
            final_time=10,
            model_options=["MHDModel"],
        )

        with self.assertRaisesRegex(
            ValueError, "either total magnetic field B or perturbation B1"
        ):
            MHDModel(
                bx=lambda x: 1.0 + 0.0 * x,
                b1x=lambda x: 0.7 + 0.0 * x,
                b0x=lambda x: 0.3 + 0.0 * x,
            )


if __name__ == "__main__":
    unittest.main()
