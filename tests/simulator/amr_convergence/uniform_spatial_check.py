#!/usr/bin/env python3
"""Uniform O4 spatial checks paired with the AMR convergence problems."""

import unittest

import numpy as np

# imported as modules: a class imported by name is collected again from this module
from tests.simulator.amr_convergence import test_alfven2d_amr, test_whistler2d_amr


class UniformSweepMixin:
    UNIFORM_NS = [32, 64, 128]
    UNIFORM_ORDER_BAND = (3.70, 4.30)

    def _uniform_sweep(self):
        errors = []
        for N in self.UNIFORM_NS:
            n = self.n_steps(N, self.SPATIAL_SIGMA)
            errors.append(self.run_uniform_case(4, N, n))
        if any(e is None for e in errors):  # dry run: decks built, nothing to fit
            return
        slope = -np.polyfit(np.log(self.UNIFORM_NS), np.log(errors), 1)[0]
        print(f"\n== {self.name}: uniform O4 errors {errors}, slope {slope:.2f}")
        self.assertTrue(
            self.UNIFORM_ORDER_BAND[0] <= slope <= self.UNIFORM_ORDER_BAND[1],
            f"{self.name} uniform O4 slope {slope:.2f} outside "
            f"{self.UNIFORM_ORDER_BAND}; errors {errors}",
        )


class AlfvenUniformSpatial(UniformSweepMixin, test_alfven2d_amr.AlfvenConvergenceTest):
    def test_uniform_spatial(self):
        self._uniform_sweep()


class WhistlerUniformSpatial(UniformSweepMixin, test_whistler2d_amr.WhistlerConvergenceTest):
    # Hall dt scales as dx^2, so this lower-cost sweep remains a clean O4 gate.
    UNIFORM_NS = [16, 32, 64]

    def test_uniform_spatial(self):
        self._uniform_sweep()


# only the uniform checks: the AMR tests inherited from the problem classes run from their own
# modules (ddt generates them under suffixed names, so they cannot be blanked by name)
def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        [
            AlfvenUniformSpatial("test_uniform_spatial"),
            WhistlerUniformSpatial("test_uniform_spatial"),
        ]
    )


if __name__ == "__main__":
    unittest.main()
