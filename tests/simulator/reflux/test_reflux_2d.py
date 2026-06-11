#
# 2D CT reflux tests: Bx/By (CT path) and Bz (hydro-like path).
# Each case: AMR sim + flat reference, three assertions on coarse level B.
# MHD and Hybrid are separate classes so `-k MHD` or `-k Hyb` filters cleanly.

import os
import tempfile
import unittest
from ddt import ddt, data, unpack
from pyphare.core.box import Box, Box2D

from tests.simulator.reflux import (
    run_reflux_pair,
    run_reflux_sim,
    ct_ic_2d,
    hydro_ic_2d,
    cf_cell_set,
    covered_cell_set,
    assert_no_spurious,
    assert_correction_fired,
    assert_divB_zero,
    assert_domain_divB_zero,
    assert_shared_process_seam,
)

CELLS = 40
DL = 0.1
DT = 5e-5
# Lx = Ly = CELLS * DL = 4.0

# 7 geometries, split by solver
# Box2D(lo, hi) → square [lo,lo] to [hi,hi]; Box([lox,loy],[hix,hiy]) for non-square
_CT_MHD_CASES = [
    ({"L0": [Box2D(10, 29)]},                                          "mhd_centered"),
    ({"L0": [Box([0, 10], [15, 29])]},                                 "mhd_touch_boundary"),
    ({"L0": [Box2D(0, 8)]},                                            "mhd_corner"),
    ({"L0": [Box([5, 2], [14, 12]), Box([5, 27], [14, 37])]},          "mhd_two_nonadj"),
    ({"L0": [Box([5, 5], [19, 20]), Box([20, 5], [34, 20])]},          "mhd_two_shared_corner"),
    ({"L0": [Box2D(8, 31)], "L1": [Box2D(18, 43)]},                   "mhd_three_level"),
    ({"L0": [Box2D(4, 35)], "L1": [Box([10, 10], [61, 29])]},         "mhd_three_nested_corner"),
]

_CT_HYB_CASES = [
    ({"L0": [Box2D(10, 29)]},                                          "hyb_centered"),
    ({"L0": [Box([0, 10], [15, 29])]},                                 "hyb_touch_boundary"),
    ({"L0": [Box2D(0, 8)]},                                            "hyb_corner"),
    ({"L0": [Box([5, 2], [14, 12]), Box([5, 27], [14, 37])]},          "hyb_two_nonadj"),
    ({"L0": [Box([5, 5], [19, 20]), Box([20, 5], [34, 20])]},          "hyb_two_shared_corner"),
    ({"L0": [Box2D(8, 31)], "L1": [Box2D(18, 43)]},                   "hyb_three_level"),
    ({"L0": [Box2D(4, 35)], "L1": [Box([10, 10], [61, 29])]},         "hyb_three_nested_corner"),
]

# 2 hard geometries, split by solver
_HYDRO_MHD_CASES = [
    ({"L0": [Box2D(0, 8)]},                                            "mhd_corner"),
    ({"L0": [Box([5, 5], [19, 20]), Box([20, 5], [34, 20])]},          "mhd_two_shared_corner"),
]

_HYDRO_HYB_CASES = [
    ({"L0": [Box2D(0, 8)]},                                            "hyb_corner"),
    ({"L0": [Box([5, 5], [19, 20]), Box([20, 5], [34, 20])]},          "hyb_two_shared_corner"),
]


@ddt
class RefluxCT2DMHDTest(unittest.TestCase):
    """2D CT path, MHD solver: Bx=B0*sin(2πy/Ly), By=B0*sin(2πx/Lx), Bz=0."""

    @data(*_CT_MHD_CASES)
    @unpack
    def test_ct_reflux(self, refinement_boxes, label):
        Lx = Ly = CELLS * DL
        ic = ct_ic_2d(Lx, Ly)
        hier_amr, hier_flat = run_reflux_pair("MHD", 2, refinement_boxes, ic, CELLS, DL, DT, label)

        cf = cf_cell_set(refinement_boxes, CELLS, 2)
        cov = covered_cell_set(refinement_boxes, CELLS, 2)

        assert_no_spurious(hier_amr, hier_flat, cf, cov, components=("Bx", "By"))
        assert_correction_fired(hier_amr, hier_flat, cf, components=("Bx", "By"))
        assert_divB_zero(hier_amr)


@ddt
class RefluxCT2DHybTest(unittest.TestCase):
    """2D CT path, Hybrid solver: Bx=B0*sin(2πy/Ly), By=B0*sin(2πx/Lx), Bz=0."""

    @data(*_CT_HYB_CASES)
    @unpack
    def test_ct_reflux(self, refinement_boxes, label):
        Lx = Ly = CELLS * DL
        ic = ct_ic_2d(Lx, Ly)
        hier_amr, hier_flat = run_reflux_pair("Hybrid", 2, refinement_boxes, ic, CELLS, DL, DT, label)

        cf = cf_cell_set(refinement_boxes, CELLS, 2)
        cov = covered_cell_set(refinement_boxes, CELLS, 2)

        assert_no_spurious(hier_amr, hier_flat, cf, cov, components=("Bx", "By"))
        assert_correction_fired(hier_amr, hier_flat, cf, components=("Bx", "By"))
        assert_divB_zero(hier_amr)


@ddt
class RefluxHydro2DMHDTest(unittest.TestCase):
    """2D hydro-like path, MHD solver: Bz=B0*sin(2πx/Lx)*sin(2πy/Ly), Bx=By=0."""

    @data(*_HYDRO_MHD_CASES)
    @unpack
    def test_hydro_reflux(self, refinement_boxes, label):
        Lx = Ly = CELLS * DL
        ic = hydro_ic_2d(Lx, Ly)
        hier_amr, hier_flat = run_reflux_pair("MHD", 2, refinement_boxes, ic, CELLS, DL, DT, label + "_hydro")

        cf = cf_cell_set(refinement_boxes, CELLS, 2)
        cov = covered_cell_set(refinement_boxes, CELLS, 2)

        assert_no_spurious(hier_amr, hier_flat, cf, cov, components=("Bz",))
        assert_correction_fired(hier_amr, hier_flat, cf, components=("Bz",))
        assert_divB_zero(hier_amr)


@ddt
class RefluxHydro2DHybTest(unittest.TestCase):
    """2D hydro-like path, Hybrid solver: Bz=B0*sin(2πx/Lx)*sin(2πy/Ly), Bx=By=0."""

    @data(*_HYDRO_HYB_CASES)
    @unpack
    def test_hydro_reflux(self, refinement_boxes, label):
        Lx = Ly = CELLS * DL
        ic = hydro_ic_2d(Lx, Ly)
        hier_amr, hier_flat = run_reflux_pair("Hybrid", 2, refinement_boxes, ic, CELLS, DL, DT, label + "_hydro")

        cf = cf_cell_set(refinement_boxes, CELLS, 2)
        cov = covered_cell_set(refinement_boxes, CELLS, 2)

        assert_no_spurious(hier_amr, hier_flat, cf, cov, components=("Bz",))
        assert_correction_fired(hier_amr, hier_flat, cf, components=("Bz",))
        assert_divB_zero(hier_amr)


class RefluxProcessBoundary2DMHDTest(unittest.TestCase):
    """Pathological: a patch (process) boundary coincident on all 3 levels.

    Each level's refined region is 40 cells and is forced to split 20+20
    (smallest==largest==20), so L0, L1 and L2 all carry a patch seam at the domain
    center (2,2). nesting_buffer=1 (reflux requirement). Gate = DOMAIN divB clean
    across the triple-coincident process boundary.
    """

    def test_three_level_shared_process_boundary(self):
        Lx = Ly = CELLS * DL
        ic = ct_ic_2d(Lx, Ly)
        # L1 = coarse [10,29] -> fine 20..59; L2 = L1-index [30,49] -> centered, inset 10.
        boxes = {"L0": [Box2D(10, 29)], "L1": [Box2D(30, 49)]}
        outdir = os.path.join(tempfile.gettempdir(), "phare_reflux_3lvl_shared_seam")
        B = run_reflux_sim(
            "MHD", 2, boxes, ic, CELLS, DL, DT, outdir,
            smallest_patch_size=20, largest_patch_size=20,  # force seam at center
            write_timestamps=[DT],
        )
        assert_shared_process_seam(B, base_cells=CELLS)   # the pathology is realized
        assert_domain_divB_zero(B, atol=1e-12)            # and stays divergence-clean


if __name__ == "__main__":
    unittest.main()
