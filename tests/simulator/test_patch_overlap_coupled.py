#!/usr/bin/env python3
"""
Same-level patch overlap in a coupled MHD-Hybrid hierarchy: level 0 is MHD, levels 1
and 2 are hybrid. The configuration below produces an overlap on level 2 at
initialization, a hybrid level above the MHD-Hybrid boundary level. The checks are those
of the pure hybrid test: the owner alone holds the particles of the shared cells, and
the fields stay identical on the shared nodes. The checks read rank-local patch data, so
the test runs on one rank.
"""

import unittest

import numpy as np
import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator

# module import, so that unittest does not also collect the pure hybrid test case
import tests.simulator.test_patch_overlap as hybrid

ph.NO_GUI()

cells = (64, 64)
dl = (0.4, 0.4)
time_step = 0.001
time_step_nbr = 3
ppc = 20
overlap_level = 2

# Gaussian bumps of Bz tagged for refinement (Bz keeps div B = 0 in 2D), positions
# chosen so that level 2 has overlapping patches at initialization
bump_centers = [(19.92, 13.61), (8.87, 11.28), (3.56, 5.44), (16.14, 15.68)]
bump_widths = [1.34, 1.06, 1.80, 1.78]


def bumps(x, y):
    b = 0.0 * x
    for (cx, cy), w in zip(bump_centers, bump_widths):
        b = b + np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / w**2)
    return b


def config(**overrides):
    kwargs = dict(
        time_step=time_step,
        time_step_nbr=time_step_nbr,
        cells=cells,
        dl=dl,
        refinement="tagging",
        max_mhd_level=1,
        max_nbr_levels=3,
        interp_order=1,
        nesting_buffer=1,
        tagging_threshold=0.1,
        hyper_resistivity=0.002,
        resistivity=0.0,
        strict=True,
        eta=0.0,
        nu=0.0,
        gamma=5.0 / 3.0,
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="TVDRK3",
        hall=True,
        res=False,
        hyper_res=False,
        model_options=["MHDModel", "HybridModel"],
    )
    kwargs.update(overrides)
    sim = ph.Simulation(**kwargs)

    def one(x, y):
        return 1.0 + 0.0 * x

    def zero(x, y):
        return 0.0 * x

    def pressure(x, y):  # total pressure balance
        return 3.0 - 0.5 * bumps(x, y) ** 2

    ph.MHDModel(
        density=one,
        vx=zero,
        vy=zero,
        vz=zero,
        bx=one,
        by=zero,
        bz=bumps,
        p=pressure,
        protons={"charge": 1, "mass": 1, "nbr_part_per_cell": ppc, "init": {"seed": 1}},
    )
    ph.ElectronModel(closure="isothermal", Te=0.0)
    return sim


class CoupledPatchOverlapTest(hybrid.PatchOverlapTest):
    def test_overlapping_patches_partition_particles(self):
        self.simulator = Simulator(config()).initialize()

        ilvl = overlap_level
        pairs = hybrid.overlapping_pairs(self.level_patches(ilvl, "getBx"))
        self.assertGreater(len(pairs), 0, "the configuration no longer has an overlap")

        for step in range(time_step_nbr + 1):
            if step > 0:
                self.simulator.advance()
            pairs = hybrid.overlapping_pairs(self.level_patches(ilvl, "getBx"))
            self.check_particles_are_on_owner_only(ilvl, pairs)
            self.check_shared_nodes_are_equal(ilvl, pairs)
            self.check_density_is_not_duplicated(ilvl, pairs)


if __name__ == "__main__":
    unittest.main()
