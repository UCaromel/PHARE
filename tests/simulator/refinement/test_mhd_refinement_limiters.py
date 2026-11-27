#
#
import os
import re
import time
import unittest

import h5py
import numpy as np
from ddt import data, ddt, unpack

import pyphare.pharein as ph
from pyphare import cpp
from pyphare.pharein.simulation import deserialize
from pyphare.simulator.simulator import Simulator
from tests.simulator import SimulatorTest

ph.NO_GUI()

# 3-level tagged 2D MHD rotor, one run per field-refinement rule.
#
# The unlimited refinement (master) takes the rotor's refined levels to negative pressure within
# a few steps; with a slope rule and the energy row, each run must complete. Each case checks that
#   - the run completes and the hierarchy reaches 3 levels at least once,
#   - the output file's serialized simulation round-trips and carries the refinement_limiting
#     dict that was asked for,
#   - for the probe case, that the probe report reaches the process log.

cells = (64, 64)
dl = (1.0 / cells[0], 1.0 / cells[1])
final_time = 0.03
timestamps = np.linspace(0, final_time, 4)

ENERGY_SWEEPS = {"tensor": "sweeps", "energy_row": True}
# (name, refinement_limiting, completes). The unlimited rule cannot create level 1 on this rotor: its
# children reach rho < 0 (refine probe), so initialisation stops on a non-finite wave speed. It runs
# last: a failed initialisation leaves the simulator's error state set for the rest of the process.
# "blob_none" is a smooth density blob on which the unlimited rule, sweeps and the energy row do
# complete: the 3-level coverage of that combination.
CASES = [
    ("vanleer", {"limiter": "vanleer", **ENERGY_SWEEPS}, True),
    ("mc", {"limiter": "mc", **ENERGY_SWEEPS}, True),
    ("mc_gate", {"limiter": "mc_gate", **ENERGY_SWEEPS}, True),
    ("ao5z", {"limiter": "ao5z", **ENERGY_SWEEPS}, True),
    ("mp", {"limiter": "mp", **ENERGY_SWEEPS}, True),
    ("ghost_none", {"limiter": "vanleer", "ghost": {"limiter": "none"}}, True),
    ("probe", {"limiter": "vanleer", **ENERGY_SWEEPS, "probe": True}, True),
    ("blob_none", {"limiter": "none", **ENERGY_SWEEPS}, True),
    ("none", {"limiter": "none", **ENERGY_SWEEPS, "probe": True}, False),
]


def log_text(log, offset):
    """What the process log gained since `offset` (the log restarts when it is shorter)."""
    with open(log) as stream:
        if os.path.getsize(log) >= offset:
            stream.seek(offset)
        return stream.read()


def config(name, refinement_limiting, diag_dir):
    blob = name.startswith("blob")
    sim = ph.Simulation(
        smallest_patch_size=8,
        time_step={"mode": "adaptive", "cfl_wave": 0.8},
        final_time=final_time,
        cells=cells,
        dl=dl,
        refinement="tagging",
        tagging={"method": "default", "quantities": {"rho": 0.2}},
        max_mhd_level=3,
        max_nbr_levels=3,
        nesting_buffer=1,
        hyper_resistivity=0.0,
        resistivity=0.0,
        diag_options={
            "format": "phareh5",
            "options": {"dir": diag_dir, "mode": "overwrite"},
            "fine_dump_lvl_max": 10,
        },
        strict=True,
        eta=0.0,
        nu=0.0,
        gamma=5.0 / 3.0,
        reconstruction="Linear",
        limiter="VanLeer",
        riemann="Rusanov",
        mhd_timestepper="TVDRK2",
        hall=False,
        model_options=["MHDModel"],
        refinement_limiting=refinement_limiting,
    )

    B0 = 5 / (np.sqrt(4 * np.pi))
    v0, r0, r1 = 2, 0.1, 0.115

    def r(x, y):
        return np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)

    def f(r):
        return (r1 - r) / (r1 - r0)

    def density(x, y):
        r_ = r(x, y)
        return np.where(r_ <= r0, 10.0, np.where(r_ < r1, 1.0 + 9.0 * f(r_), 1.0))

    def vx(x, y):
        r_ = r(x, y)
        return np.where(
            r_ <= r0,
            -v0 * (y - 0.5) / r0,
            np.where(r_ < r1, -f(r_) * v0 * (y - 0.5) / r_, 0.0),
        )

    def vy(x, y):
        r_ = r(x, y)
        return np.where(
            r_ <= r0,
            v0 * (x - 0.5) / r0,
            np.where(r_ < r1, f(r_) * v0 * (x - 0.5) / r_, 0.0),
        )

    if blob:

        def density(x, y):  # noqa: F811
            return 1.0 + 4.0 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * 0.08**2))

        def vx(x, y):  # noqa: F811
            return 0.5 + 0.0 * x

        def vy(x, y):  # noqa: F811
            return 0.3 + 0.0 * x

    ph.MHDModel(
        density=density,
        vx=vx,
        vy=vy,
        vz=lambda x, y: 0.0,
        bx=lambda x, y: B0,
        by=lambda x, y: 0.0,
        bz=lambda x, y: 0.0,
        p=lambda x, y: 1.0,
    )
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    for quantity in ["rho", "rhoV", "Etot"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)
    return sim


def levels_reached(diag_dir):
    """Highest level present in the density dump, over all dumped times."""
    best = -1
    with h5py.File(os.path.join(diag_dir, "mhd_rho.h5"), "r") as h5:
        for t in h5["t"]:
            for level in h5["t"][t]:
                if level.startswith("pl"):
                    best = max(best, int(level[2:]))
    return best + 1


def all_finite(diag_dir, name):
    """No NaN or inf in any dump of the file."""
    found = []
    with h5py.File(os.path.join(diag_dir, name), "r") as h5:
        h5.visititems(
            lambda _, obj: found.append(obj[:])
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind == "f"
            else None
        )
    assert found, f"no float dataset in {name}"
    return all(np.isfinite(a).all() for a in found)


def stored_settings(diag_dir):
    with h5py.File(os.path.join(diag_dir, "EM_B.h5"), "r") as h5:
        return deserialize(h5["py_attrs"].attrs["serialized_simulation"])


@ddt
class MHDRefinementLimitersTest(SimulatorTest):
    """3-level tagged rotor per refinement rule (perm 2,Linear,VanLeer,Rusanov,false)."""

    @data(*CASES)
    @unpack
    def test_rotor_three_levels(self, name, refinement_limiting, completes):
        diag_dir = f"rotor_limiters_{name}"
        self.register_diag_dir_for_cleanup(diag_dir)
        sim = config(name, refinement_limiting, diag_dir)
        log = ".log/0.out"
        offset = os.path.getsize(log) if os.path.exists(log) else 0
        then = time.time()
        simulator = Simulator(sim)
        if not completes and not sim.dry_run:
            with self.assertRaises(ValueError) as failure:
                simulator.run()
            simulator.reset()
            ph.global_vars.sim = None
            if cpp.mpi_rank() == 0:
                # the specific cause: level 1 creation at t = 0 gives a negative density (refine
                # probe), then initialisation stops on a non-finite wave speed on level 1
                self.assertIn(
                    "non-finite wave speed on level 1 at t = 0", str(failure.exception.__cause__)
                )
                text = log_text(log, offset)
                probe = re.search(r"refine probe level 1 init: min rho (\S+),", text)
                self.assertIsNotNone(probe, text)
                self.assertLess(float(probe.group(1)), 0.0)
            return
        simulator.run().reset()
        wall = time.time() - then
        dry_run = sim.dry_run
        ph.global_vars.sim = None
        if dry_run:
            self.skipTest("dry run: setup only")
        if cpp.mpi_rank() != 0:
            return

        nlevels = levels_reached(diag_dir)
        print(f"CASE {name}: wall {wall:.1f} s, levels reached {nlevels}", flush=True)
        self.assertEqual(nlevels, 3)

        stored = stored_settings(diag_dir)
        self.assertEqual(stored.refinement_limiting, refinement_limiting)

        for quantity in ["rho", "Etot"]:
            self.assertTrue(all_finite(diag_dir, f"mhd_{quantity}.h5"), quantity)

        if refinement_limiting.get("probe", False):
            self.assertTrue(os.path.exists(log))
            text = log_text(log, offset)
            self.assertIn("refine probe init", text)
            self.assertIn("refine probe ghost", text)


if __name__ == "__main__":
    unittest.main()
