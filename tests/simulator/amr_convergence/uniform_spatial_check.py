#!/usr/bin/env python3
"""Uniform O4 spatial checks paired with the AMR convergence problems."""

import unittest

import numpy as np

from tests.simulator.amr_convergence.test_alfven2d_amr import AlfvenConvergenceTest
from tests.simulator.amr_convergence.test_whistler2d_amr import WhistlerConvergenceTest


class UniformSweepMixin:
    UNIFORM_NS = [32, 64, 128]
    UNIFORM_ORDER_BAND = (3.70, 4.30)

    def _uniform_sweep(self):
        errors = []
        for N in self.UNIFORM_NS:
            n = self.n_steps(N, self.SPATIAL_SIGMA)
            errors.append(self.run_uniform_case(4, N, n))
        slope = -np.polyfit(np.log(self.UNIFORM_NS), np.log(errors), 1)[0]
        self.assertTrue(
            self.UNIFORM_ORDER_BAND[0] <= slope <= self.UNIFORM_ORDER_BAND[1],
            f"{self.name} uniform O4 slope {slope:.2f} outside "
            f"{self.UNIFORM_ORDER_BAND}; errors {errors}",
        )


class AlfvenUniformSpatial(UniformSweepMixin, AlfvenConvergenceTest):
    def test_uniform_spatial(self):
        self._uniform_sweep()


class WhistlerUniformSpatial(UniformSweepMixin, WhistlerConvergenceTest):
    # Hall dt scales as dx^2, so this lower-cost sweep remains a clean O4 gate.
    UNIFORM_NS = [16, 32, 64]

    def test_uniform_spatial(self):
        self._uniform_sweep()


if __name__ == "__main__":
    unittest.main()
