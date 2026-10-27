#!/usr/bin/env python3
"""Uniform self-convergence with reflective walls in y.

The state is even about both walls, which a reflective boundary preserves, but not periodic
in y (cos(pi y) on [0, 1]): the run converges at the scheme's order only if the boundary
ghosts are filled consistently with it. Periodic in y, the same state converges at order ~1.
"""

import unittest
from itertools import pairwise

import numpy as np
from ddt import data, ddt

import pyphare.pharein as ph
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator
from tests.simulator import SimulatorTest
from tests.simulator.amr_convergence import compute_errors

ph.NO_GUI()

GAMMA = 5.0 / 3.0
AMPLITUDE = 0.05
FINAL_TIME = 0.2
SIGMA = 0.32
NS = (16, 32, 64)
ORDER_BANDS = {2: (1.70, 2.30), 4: (3.70, 4.90)}

# cell averages restrict exactly to the coarser grid; cell-centred B does not
CELL_AVERAGES = ["rho", "rhoVx", "rhoVy", "rhoVz", "Etot"]


def restrict(a):
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


def shape(x, y):
    return (np.cos(2 * np.pi * x) + 0.5) * np.cos(np.pi * y)


@ddt
class UniformReflectiveCheck(SimulatorTest):
    def tearDown(self):
        super().tearDown()
        ph.global_vars.sim = None

    def final_cell_averages(self, mhd_order, N):
        """{var: N x N array} at FINAL_TIME, None under PHARE_DRY_RUN."""
        fast_speed = np.sqrt(GAMMA + 1.0)  # p = rho = B = 1 at rest
        n = int(np.ceil(FINAL_TIME / (SIGMA * (1.0 / N) / (2 * fast_speed))))
        sim = self.simulation(
            smallest_patch_size=8,
            time_step=FINAL_TIME / n,
            final_time=FINAL_TIME,
            cells=(N, N),
            dl=(1.0 / N, 1.0 / N),
            domain_boundaries={
                "ylower": {"type": "reflective"},
                "yupper": {"type": "reflective"},
            },
            eta=0.0,
            nu=0.0,
            gamma=GAMMA,
            reconstruction="WENOZ",
            limiter="None",
            riemann="Rusanov",
            mhd_timestepper="SSPRK4_5" if mhd_order == 4 else "TVDRK3",
            mhd_order=mhd_order,
            hall=False,
            model_options=["MHDModel"],
            refinement="tagging",
            max_mhd_level=1,
            max_nbr_levels=1,
            diag_options={
                "format": "phareh5",
                "options": {
                    "dir": f"phare_outputs/uniform_reflective/o{mhd_order}_N{N}",
                    "mode": "overwrite",
                },
            },
        )

        def density(x, y):
            return 1.0 + AMPLITUDE * shape(x, y)

        def vx(x, y):
            return AMPLITUDE * np.sin(2 * np.pi * x) * np.cos(np.pi * y)

        def zero(x, y):
            return 0.0 * x

        def bx(x, y):
            return 1.0 + 0.0 * x

        def p(x, y):
            return 1.0 + AMPLITUDE * GAMMA * shape(x, y)

        ph.MHDModel(
            density=density, vx=vx, vy=zero, vz=zero, bx=bx, by=zero, bz=zero, p=p
        )
        for quantity in ["rho", "rhoV", "Etot"]:
            ph.MHDDiagnostics(quantity=quantity, write_timestamps=[FINAL_TIME])
        ph.ElectromagDiagnostics(quantity="B", write_timestamps=[FINAL_TIME])

        Simulator(sim).run().reset()
        if sim.dry_run:
            return None
        fields = compute_errors.conserved_fields(
            Run(sim.diag_options["options"]["dir"]), FINAL_TIME
        )
        out = {v: np.full((N, N), np.nan) for v in CELL_AVERAGES}
        for (_, lower, upper), f in fields.items():
            for v in CELL_AVERAGES:
                out[v][lower[0] : upper[0] + 1, lower[1] : upper[1] + 1] = f[v]
        return out

    @data(2, 4)
    def test_reflective_self_convergence(self, mhd_order):
        averages = [self.final_cell_averages(mhd_order, N) for N in NS]
        if any(a is None for a in averages):  # dry run: decks built, nothing to fit
            return
        for a in averages:
            self.assertTrue(all(np.isfinite(v).all() for v in a.values()))

        diffs = [
            max(np.mean(np.abs(coarse[v] - restrict(fine[v]))) for v in CELL_AVERAGES)
            for coarse, fine in pairwise(averages)
        ]
        order = np.log2(diffs[0] / diffs[1])
        band = ORDER_BANDS[mhd_order]
        print(
            f"\n== reflective walls, O{mhd_order}: self-convergence {diffs}, order {order:.2f}"
        )
        self.assertTrue(
            band[0] <= order <= band[1],
            f"O{mhd_order} self-convergence order {order:.2f} with reflective walls outside {band}",
        )


if __name__ == "__main__":
    unittest.main()
