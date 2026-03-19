import unittest

from ddt import data, ddt, unpack

import pyphare.pharein as ph

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


if __name__ == "__main__":
    unittest.main()
