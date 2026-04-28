import unittest

from ddt import data, ddt, unpack

import pyphare.pharein as ph
from pyphare.core import phare_utilities as phut
from pyphare.pharein.simulation import check_patch_size

from tests.simulator.initialize.test_init_mhd import MHDInitializationTest

ph.NO_GUI()
ndim = 1


def permute_mhd():
    return [dict(super_class=MHDInitializationTest, interp_order=2)]


@ddt
class InitializationMHD1DTest(MHDInitializationTest):
    @data(*permute_mhd())
    @unpack
    def test_B_is_as_provided_by_user_with_external_magnetic(self, super_class, **kwargs):
        print(f"{self._testMethodName}_{ndim}d")
        self.__class__ = super_class
        self._test_B_is_as_provided_by_user(
            ndim,
            **kwargs,
            initialize_only=True,
            time_step_nbr=1,
            timestamps=[0.0],
            density=lambda x: 1.0 + 0.0 * x,
            vx=lambda x: 0.0 * x,
            vy=lambda x: 0.0 * x,
            vz=lambda x: 0.0 * x,
            bx=lambda x: 1.0 + 0.0 * x,
            by=lambda x: 0.2 + 0.0 * x,
            bz=lambda x: -0.1 + 0.0 * x,
            p=lambda x: 1.0 + 0.0 * x,
            b0x=lambda x: 0.3 + 0.0 * x,
            b0y=lambda x: -0.15 + 0.0 * x,
            b0z=lambda x: 0.05 + 0.0 * x,
        )

    @data(*permute_mhd())
    @unpack
    def test_B_is_b0_plus_b1_when_user_provides_perturbation(self, super_class, **kwargs):
        print(f"{self._testMethodName}_{ndim}d")
        self.__class__ = super_class
        self._test_B_is_as_provided_by_user(
            ndim,
            **kwargs,
            initialize_only=True,
            time_step_nbr=1,
            timestamps=[0.0],
            density=lambda x: 1.0 + 0.0 * x,
            vx=lambda x: 0.0 * x,
            vy=lambda x: 0.0 * x,
            vz=lambda x: 0.0 * x,
            b1x=lambda x: 0.7 + 0.0 * x,
            b1y=lambda x: 0.35 + 0.0 * x,
            b1z=lambda x: -0.15 + 0.0 * x,
            p=lambda x: 1.0 + 0.0 * x,
            b0x=lambda x: 0.3 + 0.0 * x,
            b0y=lambda x: -0.15 + 0.0 * x,
            b0z=lambda x: 0.05 + 0.0 * x,
        )

    @data(*permute_mhd())
    @unpack
    def test_B_defaults_to_b0_when_only_external_magnetic_is_provided(
        self, super_class, **kwargs
    ):
        print(f"{self._testMethodName}_{ndim}d")
        self.__class__ = super_class
        self._test_B_is_as_provided_by_user(
            ndim,
            **kwargs,
            initialize_only=True,
            time_step_nbr=1,
            timestamps=[0.0],
            density=lambda x: 1.0 + 0.0 * x,
            vx=lambda x: 0.0 * x,
            vy=lambda x: 0.0 * x,
            vz=lambda x: 0.0 * x,
            p=lambda x: 1.0 + 0.0 * x,
            b0x=lambda x: 0.3 + 0.0 * x,
            b0y=lambda x: -0.15 + 0.0 * x,
            b0z=lambda x: 0.05 + 0.0 * x,
        )

    @data(*permute_mhd())
    @unpack
    def test_user_cannot_specify_b_and_b1_together(self, super_class, **kwargs):
        print(f"{self._testMethodName}_{ndim}d")
        self.__class__ = super_class

        _, smallest_patch_size = check_patch_size(
            ndim, interp_order=kwargs["interp_order"], cells=120
        )
        self.simulation(
            smallest_patch_size=smallest_patch_size,
            largest_patch_size=10,
            time_step_nbr=1,
            time_step=0.001,
            boundary_types=["periodic"] * ndim,
            cells=phut.np_array_ify(120, ndim),
            dl=phut.np_array_ify(0.1, ndim),
            interp_order=kwargs["interp_order"],
            refinement_boxes={},
            diag_options={"format": "phareh5", "options": {"dir": "phare_outputs/init"}},
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
            hall=True,
            res=False,
            hyper_res=True,
            model_options=["MHDModel"],
        )

        with self.assertRaisesRegex(
            ValueError, "either total magnetic field B or perturbation B1"
        ):
            ph.MHDModel(
                density=lambda x: 1.0 + 0.0 * x,
                vx=lambda x: 0.0 * x,
                vy=lambda x: 0.0 * x,
                vz=lambda x: 0.0 * x,
                bx=lambda x: 1.0 + 0.0 * x,
                b1x=lambda x: 0.7 + 0.0 * x,
                p=lambda x: 1.0 + 0.0 * x,
                b0x=lambda x: 0.3 + 0.0 * x,
            )


if __name__ == "__main__":
    unittest.main()
