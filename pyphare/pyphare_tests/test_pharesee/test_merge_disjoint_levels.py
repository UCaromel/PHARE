"""
Pure-Python unit test for
pyphare.pharesee.hierarchy.hierarchy_utils.merge_disjoint_levels

Builds small synthetic single-quantity 1D hierarchies (no C++ bindings,
no h5 files) and checks the merge contract:
  - merged hierarchy's level-number set is the union of the inputs
  - each validation failure raises ValueError
"""

import unittest

import numpy as np

import pyphare.core.box as boxm
from pyphare.core.box import Box
from pyphare.core.gridlayout import GridLayout
from pyphare.pharesee.hierarchy import PatchHierarchy
from pyphare.pharesee.hierarchy.hierarchy_utils import merge_disjoint_levels
from pyphare.pharesee.hierarchy.patch import Patch
from pyphare.pharesee.hierarchy.patchdata import FieldData
from pyphare.pharesee.hierarchy.patchlevel import PatchLevel

GHOST_NBR = 5
RATIO = 2
CELL_WIDTH = 0.1
DOMAIN_BOX = Box(0, 63)


def field_patch(box, ilvl, qty):
    dl = CELL_WIDTH / RATIO**ilvl
    origin = box.lower * dl
    layout = GridLayout(box, origin, dl, interp_order=1, field_ghosts_nbr=GHOST_NBR)
    # Bx is primal in x: nbr nodes = box cells + 2*ghosts + 1
    data = np.zeros(box.shape[0] + 2 * GHOST_NBR + 1)
    return Patch({qty: FieldData(layout, qty, data)})


def make_hier(
    level_numbers,
    times=(0.0,),
    qty="Bx",
    domain_box=DOMAIN_BOX,
    refinement_ratio=RATIO,
):
    """one patch per level, each level covering the (refined) domain"""
    patch_levels_per_time = []
    for _ in times:
        levels = {}
        for ilvl in level_numbers:
            box = boxm.refine(domain_box, refinement_ratio**ilvl)
            levels[ilvl] = PatchLevel(ilvl, [field_patch(box, ilvl, qty)])
        patch_levels_per_time.append(levels)
    return PatchHierarchy(
        patch_levels_per_time, domain_box, refinement_ratio, list(times)
    )


class MergeDisjointLevelsTest(unittest.TestCase):
    def test_merged_levels_are_union_of_inputs(self):
        times = (0.0, 0.5)
        coarse = make_hier((0,), times=times)
        fine = make_hier((1, 2), times=times)

        merged = merge_disjoint_levels([coarse, fine])

        for time in times:
            self.assertEqual(set(merged.levels(time).keys()), {0, 1, 2})
            # patch levels are taken as-is from the inputs
            self.assertIs(merged.levels(time)[0], coarse.levels(time)[0])
            self.assertIs(merged.levels(time)[1], fine.levels(time)[1])
        self.assertEqual(merged.domain_box, coarse.domain_box)
        self.assertEqual(merged.refinement_ratio, coarse.refinement_ratio)
        self.assertEqual(sorted(merged.time_hier.keys()), sorted(coarse.time_hier.keys()))

    def test_time_key_mismatch_raises(self):
        coarse = make_hier((0,), times=(0.0,))
        fine = make_hier((1,), times=(0.0, 0.5))
        with self.assertRaises(ValueError):
            merge_disjoint_levels([coarse, fine])

    def test_domain_box_mismatch_raises(self):
        coarse = make_hier((0,))
        fine = make_hier((1,), domain_box=Box(0, 127))
        with self.assertRaises(ValueError):
            merge_disjoint_levels([coarse, fine])

    def test_overlapping_level_numbers_raise(self):
        coarse = make_hier((0, 1))
        fine = make_hier((1, 2))
        with self.assertRaises(ValueError):
            merge_disjoint_levels([coarse, fine])

    def test_quantity_name_mismatch_raises(self):
        coarse = make_hier((0,), qty="Bx")
        fine = make_hier((1,), qty="By")
        with self.assertRaises(ValueError):
            merge_disjoint_levels([coarse, fine])


if __name__ == "__main__":
    unittest.main()
