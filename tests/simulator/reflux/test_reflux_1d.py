#
# 1D hydro-like reflux tests: By and Bz paths.
# Hard geometries only (corner + two patches sharing boundary).
# MHD and Hybrid are separate classes so `-k MHD` or `-k Hyb` filters cleanly.

import unittest
from ddt import ddt, data, unpack
from pyphare.core.box import Box1D

from tests.simulator.reflux import (
    run_reflux_pair,
    hydro_ic_1d_by,
    hydro_ic_1d_bz,
    cf_cell_set,
    covered_cell_set,
    assert_no_spurious,
    assert_correction_fired,
    assert_divB_zero,
)

CELLS = 60
DL = 0.1
DT = 5e-5
# Lx = CELLS * DL = 6.0

_GEOM_MHD_CASES = [
    ({"L0": [Box1D(0, 9)]},                 "mhd_corner"),
    ({"L0": [Box1D(5, 19), Box1D(25, 39)]}, "mhd_two_shared"),
]

_GEOM_HYB_CASES = [
    ({"L0": [Box1D(0, 9)]},                 "hyb_corner"),
    ({"L0": [Box1D(5, 19), Box1D(25, 39)]}, "hyb_two_shared"),
]


@ddt
class RefluxHydro1DByMHDTest(unittest.TestCase):
    """1D By path, MHD solver: Bx=B0 (const), By=B0*sin(2πx/Lx), vx=v0."""

    @data(*_GEOM_MHD_CASES)
    @unpack
    def test_by_reflux(self, refinement_boxes, label):
        Lx = CELLS * DL
        ic = hydro_ic_1d_by(Lx)
        hier_amr, hier_flat = run_reflux_pair("MHD", 1, refinement_boxes, ic, CELLS, DL, DT, label + "_by")

        cf = cf_cell_set(refinement_boxes, CELLS, 1)
        cov = covered_cell_set(refinement_boxes, CELLS, 1)

        assert_no_spurious(hier_amr, hier_flat, cf, cov, components=("By",))
        assert_correction_fired(hier_amr, hier_flat, cf, components=("By",))
        assert_divB_zero(hier_amr)


@ddt
class RefluxHydro1DByHybTest(unittest.TestCase):
    """1D By path, Hybrid solver: Bx=B0 (const), By=B0*sin(2πx/Lx), vx=v0."""

    @data(*_GEOM_HYB_CASES)
    @unpack
    def test_by_reflux(self, refinement_boxes, label):
        Lx = CELLS * DL
        ic = hydro_ic_1d_by(Lx)
        hier_amr, hier_flat = run_reflux_pair("Hybrid", 1, refinement_boxes, ic, CELLS, DL, DT, label + "_by")

        cf = cf_cell_set(refinement_boxes, CELLS, 1)
        cov = covered_cell_set(refinement_boxes, CELLS, 1)

        assert_no_spurious(hier_amr, hier_flat, cf, cov, components=("By",))
        assert_correction_fired(hier_amr, hier_flat, cf, components=("By",))
        assert_divB_zero(hier_amr)


@ddt
class RefluxHydro1DBzMHDTest(unittest.TestCase):
    """1D Bz path, MHD solver: Bx=B0 (const), Bz=B0*sin(2πx/Lx), vx=v0."""

    @data(*_GEOM_MHD_CASES)
    @unpack
    def test_bz_reflux(self, refinement_boxes, label):
        Lx = CELLS * DL
        ic = hydro_ic_1d_bz(Lx)
        hier_amr, hier_flat = run_reflux_pair("MHD", 1, refinement_boxes, ic, CELLS, DL, DT, label + "_bz")

        cf = cf_cell_set(refinement_boxes, CELLS, 1)
        cov = covered_cell_set(refinement_boxes, CELLS, 1)

        assert_no_spurious(hier_amr, hier_flat, cf, cov, components=("Bz",))
        assert_correction_fired(hier_amr, hier_flat, cf, components=("Bz",))
        assert_divB_zero(hier_amr)


@ddt
class RefluxHydro1DBzHybTest(unittest.TestCase):
    """1D Bz path, Hybrid solver: Bx=B0 (const), Bz=B0*sin(2πx/Lx), vx=v0."""

    @data(*_GEOM_HYB_CASES)
    @unpack
    def test_bz_reflux(self, refinement_boxes, label):
        Lx = CELLS * DL
        ic = hydro_ic_1d_bz(Lx)
        hier_amr, hier_flat = run_reflux_pair("Hybrid", 1, refinement_boxes, ic, CELLS, DL, DT, label + "_bz")

        cf = cf_cell_set(refinement_boxes, CELLS, 1)
        cov = covered_cell_set(refinement_boxes, CELLS, 1)

        assert_no_spurious(hier_amr, hier_flat, cf, cov, components=("Bz",))
        assert_correction_fired(hier_amr, hier_flat, cf, components=("Bz",))
        assert_divB_zero(hier_amr)


if __name__ == "__main__":
    unittest.main()
