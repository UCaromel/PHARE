#!/usr/bin/env python3
"""
Regression test for Run.GetDivB(merged=True).

make_interpolator's own internal yeeCoordsFor calls (building the merged
interpolator's target grid) used to resolve centering purely by looking the
quantity name up in core.gridlayout.yee_centering -- a fixed table of
primitive quantity names (Bx, By, rho, ...). "divB" is a derived quantity
computed by pharesee.run.utils._compute_divB, which already attaches the
correct centering (["dual", "dual"] in 2D) to the PatchData it builds; that
metadata was simply never propagated down to make_interpolator, so the name
lookup raised KeyError for "divB" specifically -- this is what
tests/simulator/amr_convergence/amr_convergence_base.py's
_merged_divb_growth (real production caller of GetDivB(merged=True)) hit.

This test exercises the real Run.GetDivB(merged=True) path end to end (no
mocking) at both the initial and a later dumped time, using the same 2D
oblique CP-Alfven-wave MHD IC as tests/simulator/amr_convergence's
alfven2d case (see test_alfven2d_amr.py), but single-level and at a small N
so it runs quickly -- this test is not a numerics/order gate (that is
tests/functional/mhd_convergence and tests/simulator/amr_convergence's own
job; those tests must not be relaxed to make this one pass).
"""

import os
import unittest

import numpy as np

import pyphare.pharein as ph
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator

os.environ.setdefault("PHARE_SCOPE_TIMING", "0")

ph.NO_GUI()

# Same oblique CP-Alfven-wave IC as test_alfven2d_amr.py (analytically
# divergence-free by construction), reused here only as a known-good,
# already-proven 2D MHD profile -- this test does not assert anything about
# its convergence order or its divB growth over time.
ALPHA = 30.0 * np.pi / 180.0
COSALPHA = np.cos(ALPHA)
SINALPHA = np.sin(ALPHA)
GAMMA = 5.0 / 3.0
P0, RHO0, DB, DV = 0.1, 1.0, 0.1, 0.1

N = 16
DIAG_DIR = "phare_outputs/test_run_divb_merged"


def _cfl_dt(N):
    dx = (1.0 / N) / COSALPHA
    dy = (1.0 / N) / SINALPHA
    c = DV + np.sqrt(GAMMA * P0 / RHO0 + (1.0 + DB**2) / RHO0)
    return 1.0 / (c / dx + c / dy)


class RunGetDivBMergedTest(unittest.TestCase):
    def setUp(self):
        ph.global_vars.sim = None

        dt = 0.3 * _cfl_dt(N)  # comfortably under the CFL stability limit
        n_steps = 5

        # check_time() (pharein/simulation.py) requires exactly two of
        # {final_time, time_step, time_step_nbr}, not all three -- give it
        # time_step + time_step_nbr and read the resolved final_time back off
        # sim below, rather than also passing a third, independently-computed
        # final_time that would just have to agree with it.
        sim = ph.Simulation(
            smallest_patch_size=8,
            time_step=dt,
            time_step_nbr=n_steps,
            cells=(N, N),
            dl=((1.0 / N) / COSALPHA, (1.0 / N) / SINALPHA),
            refinement="tagging",
            max_mhd_level=1,
            max_nbr_levels=1,
            hyper_resistivity=0.0,
            resistivity=0.0,
            diag_options={
                "format": "phareh5",
                "options": {"dir": DIAG_DIR, "mode": "overwrite"},
            },
            strict=True,
            nesting_buffer=1,
            eta=0.0,
            nu=0.0,
            gamma=GAMMA,
            reconstruction="WENOZ",
            limiter="None",
            riemann="Rusanov",
            mhd_timestepper="SSPRK4_5",
            mhd_order=2,
            hall=False,
            res=False,
            hyper_res=False,
            model_options=["MHDModel"],
        )
        self.final_time = sim.final_time

        def phase(x, y):
            return 2 * np.pi * (x * COSALPHA + y * SINALPHA)

        def density(x, y):
            return RHO0

        def vx(x, y):
            return -DV * np.sin(phase(x, y)) * SINALPHA

        def vy(x, y):
            return DV * np.sin(phase(x, y)) * COSALPHA

        def vz(x, y):
            return DV * np.cos(phase(x, y))

        def bx(x, y):
            return COSALPHA - DB * np.sin(phase(x, y)) * SINALPHA

        def by(x, y):
            return SINALPHA + DB * np.sin(phase(x, y)) * COSALPHA

        def bz(x, y):
            return DB * np.cos(phase(x, y))

        def p(x, y):
            return P0

        ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)
        ph.ElectromagDiagnostics(quantity="B", write_timestamps=[0.0, sim.final_time])

        Simulator(sim).run().reset()
        if sim.dry_run:
            self.skipTest("PHARE_DRY_RUN: setup only, no diagnostics written")
        self.run = Run(DIAG_DIR)

    def _check_merged_divB(self, time):
        merged = self.run.GetDivB(time, merged=True)
        self.assertIn("divB", merged)

        interpolator, coords = merged["divB"]
        self.assertEqual(len(coords), 2)
        x, y = coords

        self.assertGreater(x.shape[0], 0)
        self.assertGreater(y.shape[0], 0)
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertTrue(np.all(np.isfinite(y)))

        # divB is dual/dual centered (see _compute_divB's "centering":
        # ["dual", "dual"]): gridlayout.yeeCoordsFor only offsets the first
        # sample point by half a cell (0.5*dl) when centering resolves to
        # "dual" -- it is exactly 0.0 for "primal". This is the precise,
        # size-independent signature of the bug this test guards: before the
        # fix, make_interpolator had no way to learn "divB" is dual-centered
        # (it isn't in core.gridlayout.yee_centering's fixed name table) and
        # KeyError'd outright; a future regression that silently fell back to
        # a wrong "primal" default instead would show up here as x[0] == 0.0.
        dx = (1.0 / N) / COSALPHA
        dy = (1.0 / N) / SINALPHA
        self.assertAlmostEqual(x[0], 0.5 * dx, places=12)
        self.assertAlmostEqual(y[0], 0.5 * dy, places=12)

        X, Y = np.meshgrid(x, y, indexing="ij")
        values = interpolator(X, Y)
        self.assertEqual(values.shape, (x.shape[0], y.shape[0]))
        self.assertTrue(np.all(np.isfinite(values)))
        # Not a convergence/growth gate (see module docstring): just guards
        # against the interpolator silently returning garbage/huge values,
        # which a wrong or misaligned coordinate grid would produce.
        self.assertLess(np.max(np.abs(values)), 10.0)

    def test_getdivb_merged_initial(self):
        self._check_merged_divB(0.0)

    def test_getdivb_merged_final(self):
        self._check_merged_divB(self.final_time)


if __name__ == "__main__":
    unittest.main()
