#!/usr/bin/env python3
"""Fourth-order AMR convergence of the alfven2d problem on three levels.

Level 2 sits two level-1 cells inside level 1, so its ghosts reach past level 1 and are
filled from level 0 through coarse-interpolation temporaries. Those level-0 values must be
taken at the level-2 stage time like the level-1 ones: two-level runs never read them.
"""

import unittest

# imported as a module: a class imported by name is collected again from this module
from tests.simulator.amr_convergence import test_alfven2d_amr

MARGIN = 2  # level-2 box inside the level-1 box, in level-1 cells


class AlfvenThreeLevels(test_alfven2d_amr.AlfvenConvergenceTest):
    SPATIAL_NS = (32, 64)

    def level1_box(self, N):
        (lo, _), (hi, _) = self.fine_box(N)
        lo, hi = 2 * lo + MARGIN, 2 * hi + 1 - MARGIN
        return [[lo, lo], [hi, hi]]

    def covered_boxes(self, N):
        return {**super().covered_boxes(N), 1: self.level1_box(N)}

    def amr_simulation(self, mhd_order, N, n):
        base = f"phare_outputs/{self.name}_amr_convergence/l3_o{mhd_order}_N{N}_n{n}"
        return self.simulation(
            refinement="boxes",
            refinement_boxes={
                "L0": {"B0": self.fine_box(N)},
                "L1": {"B0": self.level1_box(N)},
            },
            max_mhd_level=3,
            diag_options={
                "format": "phareh5",
                "options": {"dir": base, "mode": "overwrite"},
            },
            **self._common(mhd_order, N, n),
        )

    def test_three_level_spatial_convergence(self):
        self.check_spatial_order(
            4, self.SPATIAL_NS, self.SPATIAL_SIGMA, self.SPATIAL_ORDER_BANDS[4]
        )


# only this check: the two-level tests inherited from the problem class run from their own module
def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        [AlfvenThreeLevels("test_three_level_spatial_convergence")]
    )


if __name__ == "__main__":
    unittest.main()
