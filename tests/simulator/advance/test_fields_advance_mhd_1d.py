import unittest

from ddt import data, ddt, unpack

import pyphare.pharein as ph

from tests.simulator.advance.test_advance_mhd import MHDAdvanceTest

ph.NO_GUI()
ndim = 1


def permute_mhd(boxes={}):
    return [dict(super_class=MHDAdvanceTest, interp_order=2, refinement_boxes=boxes)]


@ddt
class AdvanceMHDTest1D(MHDAdvanceTest):
    @unpack
    @data(*permute_mhd({}))
    def test_dynamic_external_magnetic_matches_baseline_solution(
        self, super_class, **kwargs
    ):
        self.__class__ = super_class
        print(f"{self._testMethodName}_{ndim}d")

        model_kwargs = {
            "density": lambda x: 1.0 + 0.0 * x,
            "vx": lambda x: 0.0 * x,
            "vy": lambda x: 0.0 * x,
            "vz": lambda x: 0.0 * x,
            "bx": lambda x: 1.0 + 0.0 * x,
            "by": lambda x: -0.2 + 0.0 * x,
            "bz": lambda x: 0.15 + 0.0 * x,
            "p": lambda x: 1.0 + 0.0 * x,
        }

        baseline = self.getHierarchy(
            ndim,
            qty="fields",
            time_step=0.001,
            time_step_nbr=3,
            hall=True,
            diag_outputs="mhd_baseline",
            model_kwargs=model_kwargs,
            **kwargs,
        )
        split = self.getHierarchy(
            ndim,
            qty="fields",
            time_step=0.001,
            time_step_nbr=3,
            hall=True,
            diag_outputs="mhd_external_b0",
            model_kwargs={
                **model_kwargs,
                "b0x": lambda x: 0.8 + 0.0 * x,
                "b0y": lambda x: -0.15 + 0.0 * x,
                "b0z": lambda x: 0.05 + 0.0 * x,
            },
            **kwargs,
        )

        self.assert_hierarchies_equal(baseline, split, atol=2e-12)


if __name__ == "__main__":
    unittest.main()
