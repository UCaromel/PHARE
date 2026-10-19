#!/usr/bin/env python3
"""Uniform O4 self-convergence of the resistive alfven2d problem."""

import unittest

import numpy as np

from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator

# imported as modules: a class imported by name is collected again from this module
from tests.simulator.amr_convergence import compute_errors, test_alfven2d_amr

# cell averages restrict exactly to the coarser grid; cell-centred B does not
CELL_AVERAGES = ["rho", "rhoVx", "rhoVy", "rhoVz", "Etot"]


def restrict(a):
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


class AlfvenResistiveUniform(test_alfven2d_amr.AlfvenConvergenceTest):
    # With eta > 0 the wave decays and has no closed form, so successive resolutions are
    # compared with each other: |u_N - R(u_2N)| shrinks like N^-4 at fourth order.
    ETA = 0.01
    final_time = 0.25
    NS = [16, 32, 64]
    ORDER_BAND = (3.70, 4.30)

    def _common(self, mhd_order, N, n):
        return {**super()._common(mhd_order, N, n), "eta": self.ETA}

    def final_cell_averages(self, N):
        """{var: N x N array} at final_time, None under PHARE_DRY_RUN."""
        sim = self.uniform_simulation(4, N, self.n_steps(N, self.SPATIAL_SIGMA))
        self.add_model_and_diags()
        Simulator(sim).run().reset()
        if sim.dry_run:
            return None
        fields = compute_errors.conserved_fields(
            Run(sim.diag_options["options"]["dir"]), self.final_time
        )
        out = {v: np.full((N, N), np.nan) for v in CELL_AVERAGES}
        for (_, lower, upper), f in fields.items():
            for v in CELL_AVERAGES:
                out[v][lower[0] : upper[0] + 1, lower[1] : upper[1] + 1] = f[v]
        return out

    def test_resistive_self_convergence(self):
        averages = [self.final_cell_averages(N) for N in self.NS]
        if any(a is None for a in averages):  # dry run: decks built, nothing to fit
            return
        for a in averages:
            self.assertTrue(all(np.isfinite(v).all() for v in a.values()))

        diffs = [
            max(np.mean(np.abs(coarse[v] - restrict(fine[v]))) for v in CELL_AVERAGES)
            for coarse, fine in zip(averages, averages[1:])
        ]
        order = np.log2(diffs[0] / diffs[1])
        print(f"\n== {self.name} eta={self.ETA}: self-convergence {diffs}, order {order:.2f}")
        self.assertTrue(
            self.ORDER_BAND[0] <= order <= self.ORDER_BAND[1],
            f"resistive O4 self-convergence order {order:.2f} outside {self.ORDER_BAND}",
        )


# only this check: the AMR tests inherited from the problem class run from their own module
def load_tests(loader, tests, pattern):
    return unittest.TestSuite([AlfvenResistiveUniform("test_resistive_self_convergence")])


if __name__ == "__main__":
    unittest.main()
