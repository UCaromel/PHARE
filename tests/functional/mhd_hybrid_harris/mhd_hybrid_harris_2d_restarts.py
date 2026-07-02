#!/usr/bin/env python3
"""
Restart test for coupled MHD-Hybrid runs (Harris 2D config).

Run 1 dumps a restart at t=RESTART_TIME and diags at [RESTART_TIME, FINAL_TIME].
Run 2 restarts from that file and dumps the same diags to a second dir.
Both diag sets must match at FINAL_TIME (same layout, same values), following
tests/simulator/test_restarts.py::check_diags.
"""

import os
import importlib.util
from pathlib import Path

import numpy as np

from pyphare import cpp
import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator, startMPI
from pyphare.pharesee.hierarchy.fromh5 import hierarchy_fromh5

from tests.simulator import SimulatorTest

ph.NO_GUI()

# reuse the sibling smoke test's config()
HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location(
    "_base", os.path.join(HERE, "mhd_hybrid_harris_2d.py")
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

TIME_STEP = base.time_step
# env overrides: level counts + coarse-step counts (repro_3lvl.py convention)
NBR = int(os.environ.get("RESTART_NBR", "2"))  # max_nbr_levels
MHD_LEVEL = int(os.environ.get("RESTART_MHD_LEVEL", "1"))  # max_mhd_level
RESTART_STEP = int(os.environ.get("RESTART_STEP", "10"))  # coarse steps to restart dump
FINAL_STEP = int(os.environ.get("RESTART_FINAL_STEP", str(2 * RESTART_STEP)))
RESTART_TIME = RESTART_STEP * TIME_STEP
FINAL_TIME = FINAL_STEP * TIME_STEP
TIMESTAMPS = [RESTART_TIME, FINAL_TIME]

DIAG_DIR0 = "phare_outputs/mhd_hybrid_harris_2d_restarts"
DIAG_DIR1 = DIAG_DIR0 + "_n2"


def config(diag_dir):
    """base.config() with shortened run and diag/restart-friendly timestamps.

    base.config() reads its module globals at call time, so overriding them
    here retargets the Simulation and every diagnostic it registers.
    """
    base.final_time = FINAL_TIME
    base.timestamps = TIMESTAMPS
    base.B_TIMESTAMPS = TIMESTAMPS
    base.diag_dir = diag_dir
    ph.global_vars.sim = None
    sim = base.config()
    sim.max_mhd_level = MHD_LEVEL
    sim.max_nbr_levels = NBR
    return sim


def compare_diags(test, diag_dir0, diag_dir1, times):
    """Per-file comparison: coupled diags span different level subsets per file
    (MHD quantities on MHD levels, hybrid/fluid on hybrid levels), so the
    all-files-merged loader cannot be used here.
    """
    if cpp.mpi_rank() > 0:
        return
    h5_files = sorted(p.name for p in Path(diag_dir0).glob("*.h5"))
    test.assertGreater(len(h5_files), 0)
    seen_levels = set()
    for time in times:
        for h5_name in h5_files:
            datahier0 = hierarchy_fromh5(str(Path(diag_dir0) / h5_name), time)
            datahier1 = hierarchy_fromh5(str(Path(diag_dir1) / h5_name), time)

            test.assertEqual(
                set(datahier0.quantities()), set(datahier1.quantities()), h5_name
            )
            test.assertEqual(
                sorted(datahier0.levels(time)), sorted(datahier1.levels(time)), h5_name
            )
            seen_levels |= set(datahier0.levels(time))

            for ilvl, lvl0 in datahier0.levels(time).items():
                lvl1 = datahier1.levels(time)[ilvl]
                test.assertEqual(len(lvl0.patches), len(lvl1.patches), h5_name)
                for patch0, patch1 in zip(lvl0.patches, lvl1.patches):
                    test.assertEqual(patch0.box, patch1.box, h5_name)
                    for pd_key, pd0 in patch0.patch_datas.items():
                        pd1 = patch1.patch_datas[pd_key]
                        np.testing.assert_equal(
                            pd0.dataset[:],
                            pd1.dataset[:],
                            err_msg=f"{h5_name} t={time} L{ilvl} {patch0.box} {pd_key}",
                        )
    test.assertIn(NBR - 1, seen_levels)  # finest level covered by the comparison


class MHDHybridRestartsTest(SimulatorTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulator = None

    def tearDown(self):
        super().tearDown()
        if self.simulator is not None:
            self.simulator.reset()
        self.simulator = None
        ph.global_vars.sim = None

    def test_restarts(self):
        self.register_diag_dir_for_cleanup(DIAG_DIR0)
        self.register_diag_dir_for_cleanup(DIAG_DIR1)

        # first simulation: dump restart at RESTART_TIME
        sim = config(DIAG_DIR0)
        sim.restart_options = dict(
            dir=DIAG_DIR0, mode="overwrite", timestamps=[RESTART_TIME]
        )
        Simulator(sim).run().reset()

        # second simulation: restart from RESTART_TIME
        sim = config(DIAG_DIR1)
        sim.restart_options = dict(
            dir=DIAG_DIR0, mode="overwrite", restart_time=RESTART_TIME
        )
        Simulator(sim).run().reset()

        compare_diags(self, DIAG_DIR0, DIAG_DIR1, [FINAL_TIME])
        cpp.mpi_barrier()
        return self


if __name__ == "__main__":
    startMPI()
    MHDHybridRestartsTest("test_restarts").test_restarts().tearDown()
