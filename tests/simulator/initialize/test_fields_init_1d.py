"""
This file exists independently from test_initialization.py to isolate dimension
  test cases and allow each to be overridden in some way if required.
"""

import unittest
from ddt import data, ddt, unpack

import pyphare.pharein as ph

from tests.simulator.initialize.test_init_mhd import MHDInitializationTest
from tests.simulator.initialize.test_init_hybrid import HybridInitializationTest

ph.NO_GUI()

ndim = 1
interp_orders = [1, 2, 3]


def permute_hybrid():
    return [
        dict(super_class=HybridInitializationTest, interp_order=interp_order)
        for interp_order in interp_orders
    ]


def permute_mhd():  # interp_order hax todo
    return [dict(super_class=MHDInitializationTest, interp_order=2)]


def permute(hybrid=True, mhd=False):
    return (permute_hybrid() if hybrid else []) + (permute_mhd() if mhd else [])


@ddt
class Initialization1DTest(MHDInitializationTest, HybridInitializationTest):
    @data(*permute())
    @unpack
    def test_B_is_as_provided_by_user(self, super_class, **kwargs):
        print(f"{self._testMethodName}_{ndim}d")
        self.__class__ = super_class  # cast to super class
        self._test_B_is_as_provided_by_user(ndim, **kwargs)

    @data(*permute(hybrid=False, mhd=True))
    @unpack
    def test_B_is_as_provided_by_user_with_external_magnetic(self, super_class, **kwargs):
        print(f"{self._testMethodName}_{ndim}d")
        self.__class__ = super_class  # cast to super class
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

    @data(*permute())
    @unpack
    def test_bulkvel_is_as_provided_by_user(self, super_class, **kwargs):
        print(f"{self._testMethodName}_{ndim}d")
        self.__class__ = super_class  # cast to super class
        self._test_bulkvel_is_as_provided_by_user(ndim, **kwargs)

    @data(*permute())
    @unpack
    def test_density_is_as_provided_by_user(self, super_class, **kwargs):
        print(f"{self._testMethodName}_{ndim}d")
        self.__class__ = super_class  # cast to super class
        self._test_density_is_as_provided_by_user(ndim, **kwargs)

    @data(*permute())
    @unpack
    def test_density_decreases_as_1overSqrtN(self, super_class, **kwargs):
        print(f"{self._testMethodName}_{ndim}d")
        self.__class__ = super_class  # cast to super class
        self._test_density_decreases_as_1overSqrtN(ndim, **kwargs)


if __name__ == "__main__":
    unittest.main()
