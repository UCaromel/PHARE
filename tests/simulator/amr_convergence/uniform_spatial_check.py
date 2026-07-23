#!/usr/bin/env python3
"""Uniform-grid spatial-order check for the amr_convergence problems.

The AMR suite never asserts the uniform spatial slope (uniform runs are only
fixed-N sigma-sweep controls). This driver sweeps N on single-level runs at
each test's proven operating-point sigma; with SSPRK4_5+WENOZ the measured
slope is expected ~4 on the point-value branch.

Temporal-error isolation differs per problem:
- whistler2d: dt ~ dx^2 makes SSPRK4_5's temporal error O(dx^8) -- the slope
  is the pure spatial order.
- alfven2d: dt ~ dx, so the temporal error is O(dx^4) -- same order as the
  spatial one; the slope measures the combined 4th-order space+time scheme,
  which is still the intended gate.
"""

import unittest

import numpy as np

from tests.simulator.amr_convergence.test_alfven2d_amr import (
    AlfvenConvergenceTest,
)
from tests.simulator.amr_convergence.test_whistler2d_amr import (
    WhistlerConvergenceTest,
)


class UniformSweepMixin:
    UNIFORM_NS = [32, 64, 128]

    def _uniform_sweep(self):
        Ns = self.UNIFORM_NS
        sigma = self.SPATIAL_SIGMA
        print(f"\n== {self.name}: UNIFORM spatial convergence, sigma={sigma} ==")
        errs = []
        for N in Ns:
            n = self.n_steps(N, sigma)
            eps = self.run_uniform_case(N, n)
            errs.append(eps)
            print(
                f"N={N:4d}  dt={self.final_time/n:.3e} (n={n})  "
                f"uniform={eps:.6e}",
                flush=True,
            )
        for (Na, ea), (Nb, eb) in zip(zip(Ns, errs), list(zip(Ns, errs))[1:]):
            print(f"  segment N={Na}->{Nb}: order {np.log(ea/eb)/np.log(Nb/Na):.2f}")
        slope = -np.polyfit(np.log(Ns), np.log(errs), 1)[0]
        print(f"  {self.name} UNIFORM measured spatial order: {slope:.2f}", flush=True)


class WhistlerUniformSpatial(UniformSweepMixin, WhistlerConvergenceTest):
    # N=128 alone costs ~2800 steps (dt ~ dx^2); two segments (16->32->64)
    # give a slope check without paying N=128's cost.
    UNIFORM_NS = [16, 32, 64]

    def test_uniform_spatial(self):
        self._uniform_sweep()


class AlfvenUniformSpatial(UniformSweepMixin, AlfvenConvergenceTest):
    def test_uniform_spatial(self):
        self._uniform_sweep()


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = {
        "whistler": "WhistlerUniformSpatial.test_uniform_spatial",
        "alfven": "AlfvenUniformSpatial.test_uniform_spatial",
    }
    argv = [sys.argv[0]] + (list(names.values()) if which == "all" else [names[which]])
    unittest.main(argv=argv)
