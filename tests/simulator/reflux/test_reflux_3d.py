#
# 3D CT reflux tests — marked @expectedFailure.
# 3D accumulation code path exists but is unverified; failing tests drive the fix.
# Remove @expectedFailure once 3D accumulation is verified correct.
# MHD and Hybrid are separate classes so `-k MHD` or `-k Hyb` filters cleanly.

import unittest
import numpy as np
from ddt import ddt, data, unpack
from pyphare.core.box import Box, Box3D

from tests.simulator.reflux import (
    run_reflux_pair,
    run_reflux_sim,
    ct_ic_3d,
    cf_cell_set,
    covered_cell_set,
    assert_no_spurious,
    assert_correction_fired,
    assert_divB_zero,
    assert_domain_divB_zero,
    assert_shared_process_seam,
)

CELLS = 20   # 20^3 domain — smaller to keep 3D tests tractable
DL = 0.2
DT = 5e-5
# Lx = Ly = Lz = CELLS * DL = 4.0

# 7 geometries, split by solver
# Box3D(lo, hi) → cube [lo,lo,lo] to [hi,hi,hi]; Box([lox,loy,loz],[hix,hiy,hiz]) for non-cube
# Geometries migrate from PENDING (@expectedFailure) to PASSING (undecorated, real gate)
# as each is verified green on the all-levels domain-divB gate. Order roughly simplest-first:
# centered → corner/touch_boundary → two_* → three_level/nested.
_CT_3D_MHD_ALL = [
    ({"L0": [Box3D(5, 14)]},                                               "mhd_centered"),
    ({"L0": [Box([0, 5, 5], [7, 14, 14])]},                                "mhd_touch_boundary"),
    ({"L0": [Box3D(0, 4)]},                                                "mhd_corner"),
    ({"L0": [Box3D(2, 7), Box([2, 12, 2], [7, 17, 7])]},                   "mhd_two_nonadj"),
    ({"L0": [Box3D(2, 9), Box([10, 2, 2], [16, 9, 9])]},                   "mhd_two_shared_corner"),
    ({"L0": [Box3D(3, 15)], "L1": [Box3D(8, 21)]},                         "mhd_three_level"),
    ({"L0": [Box3D(2, 17)], "L1": [Box([5, 5, 5], [29, 14, 29])]},         "mhd_three_nested_corner"),
]

_CT_3D_HYB_ALL = [
    ({"L0": [Box3D(5, 14)]},                                               "hyb_centered"),
    ({"L0": [Box([0, 5, 5], [7, 14, 14])]},                                "hyb_touch_boundary"),
    ({"L0": [Box3D(0, 4)]},                                                "hyb_corner"),
    ({"L0": [Box3D(2, 7), Box([2, 12, 2], [7, 17, 7])]},                   "hyb_two_nonadj"),
    ({"L0": [Box3D(2, 9), Box([10, 2, 2], [16, 9, 9])]},                   "hyb_two_shared_corner"),
    ({"L0": [Box3D(3, 15)], "L1": [Box3D(8, 21)]},                         "hyb_three_level"),
    ({"L0": [Box3D(2, 17)], "L1": [Box([5, 5, 5], [29, 14, 29])]},         "hyb_three_nested_corner"),
]

# Migration state — labels that pass the gate move to *_PASSING. Start: all pending.
_MHD_PASSING_LABELS = set()
_HYB_PASSING_LABELS = set()

_CT_3D_MHD_PASSING = [c for c in _CT_3D_MHD_ALL if c[1] in _MHD_PASSING_LABELS]
_CT_3D_MHD_PENDING = [c for c in _CT_3D_MHD_ALL if c[1] not in _MHD_PASSING_LABELS]
_CT_3D_HYB_PASSING = [c for c in _CT_3D_HYB_ALL if c[1] in _HYB_PASSING_LABELS]
_CT_3D_HYB_PENDING = [c for c in _CT_3D_HYB_ALL if c[1] not in _HYB_PASSING_LABELS]


class RefluxGateCalibration3DMHDTest(unittest.TestCase):
    """Calibrate the 3D domain-divB gate BEFORE trusting it to judge reflux results.
    (a) GREEN on a known-divB=0 flat hierarchy; (b) RED on an injected divergence.
    A shape/offset bug in the gate would false-green everything → this is a hard
    prerequisite for un-@expectedFailure'ing the 3D suite."""

    def test_divB_gate_calibration(self):
        Lx = Ly = Lz = CELLS * DL
        ic = ct_ic_3d(Lx, Ly, Lz)
        _, hier_flat = run_reflux_pair("MHD", 3, {"L0": [Box3D(5, 14)]}, ic, CELLS, DL, DT, "cal")

        # (a) GREEN: CT on a uniform (no-CF) hierarchy preserves divB → ~machine zero
        assert_domain_divB_zero(hier_flat, atol=1e-12)

        # (b) RED: a ramp along x makes d(Bx)/dx constant nonzero → divB != 0 everywhere.
        # (A constant offset would NOT — np.diff is unchanged — so use a ramp.)
        # dataset is a read-only h5py view; reassign .dataset to an in-memory numpy copy.
        bx = hier_flat.level(0).patches[0].patch_datas["Bx"]
        arr = np.array(bx.dataset[:])
        ramp = np.arange(arr.shape[0]).reshape(-1, 1, 1)
        bx.dataset = arr + ramp
        with self.assertRaises(AssertionError):
            assert_domain_divB_zero(hier_flat, atol=1e-12)


def _run_ct_3d_case(test, solver, refinement_boxes, label):
    """Shared body: run AMR+flat pair, gate on all-levels domain divB (+ placement
    when REFLUX_STRICT, default on). Used by both PASSING and PENDING methods."""
    Lx = Ly = Lz = CELLS * DL
    ic = ct_ic_3d(Lx, Ly, Lz)
    hier_amr, hier_flat = run_reflux_pair(solver, 3, refinement_boxes, ic, CELLS, DL, DT, label + "_3d")

    cf = cf_cell_set(refinement_boxes, CELLS, 3)
    cov = covered_cell_set(refinement_boxes, CELLS, 3)

    assert_no_spurious(hier_amr, hier_flat, cf, cov, components=("Bx", "By", "Bz"))
    assert_correction_fired(hier_amr, hier_flat, cf, components=("Bx", "By", "Bz"))
    assert_domain_divB_zero(hier_amr)   # all levels (was vacuous assert_divB_zero level=0)


@ddt
class RefluxCT3DMHDTest(unittest.TestCase):
    """3D CT path, MHD solver: all three B components face-centered and refluxed via edge E."""

    @data(*_CT_3D_MHD_PASSING)
    @unpack
    def test_ct_reflux(self, refinement_boxes, label):
        _run_ct_3d_case(self, "MHD", refinement_boxes, label)

    @unittest.expectedFailure
    @data(*_CT_3D_MHD_PENDING)
    @unpack
    def test_ct_reflux_pending(self, refinement_boxes, label):
        _run_ct_3d_case(self, "MHD", refinement_boxes, label)


@ddt
class RefluxCT3DHybTest(unittest.TestCase):
    """3D CT path, Hybrid solver: all three B components face-centered and refluxed via edge E."""

    @data(*_CT_3D_HYB_PASSING)
    @unpack
    def test_ct_reflux(self, refinement_boxes, label):
        _run_ct_3d_case(self, "Hybrid", refinement_boxes, label)

    @unittest.expectedFailure
    @data(*_CT_3D_HYB_PENDING)
    @unpack
    def test_ct_reflux_pending(self, refinement_boxes, label):
        _run_ct_3d_case(self, "Hybrid", refinement_boxes, label)


if __name__ == "__main__":
    unittest.main()
