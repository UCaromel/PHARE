#!/usr/bin/env python3


import numpy as np
from pathlib import Path

import pyphare.pharein as ph
from pyphare.cpp import cpp_lib
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator, startMPI
from pyphare.pharesee.hierarchy.hierarchy_utils import diff_hierarchy

from tests.simulator import SimulatorTest
from tests.simulator.test_advance import AdvanceTestBase


ph.NO_GUI()

cpp = cpp_lib()

start_time = 0
cells = (150, 75)
time_step = 0.005
time_step_nbr = 2
final_time = start_time + (time_step * time_step_nbr)
timestamps = np.arange(start_time, final_time + time_step, time_step)
diag_dir = "phare_outputs/harris"
print("timestamps", timestamps)  # timestamps.shape


def config():
    L = 0.5

    sim = ph.Simulation(
        # smallest_patch_size=10,
        # largest_patch_size=25,
        time_step=time_step,
        time_step_nbr=time_step_nbr,
        cells=cells,
        dl=(0.40, 0.40),
        refinement="tagging",
        max_nbr_levels=3,
        hyper_resistivity=0.002,
        resistivity=0.001,
        diag_options={
            "format": "phareh5",
            "options": {"dir": diag_dir, "mode": "overwrite"},
        },
        # restart_options={
        #     "dir": "checkpoints",
        #     "mode": "overwrite",
        #     "timestamps": [final_time],
        #     # "restart_time": start_time,
        # },
        strict=True,
        nesting_buffer=1,
        tag_buffer=3,
    )

    def density(x, y):
        Ly = sim.simulation_domain()[1]
        return (
            0.4
            + 1.0 / np.cosh((y - Ly * 0.3) / L) ** 2
            + 1.0 / np.cosh((y - Ly * 0.7) / L) ** 2
        )

    def S(y, y0, l):
        return 0.5 * (1.0 + np.tanh((y - y0) / l))

    def by(x, y):
        Lx = sim.simulation_domain()[0]
        Ly = sim.simulation_domain()[1]
        sigma = 1.0
        dB = 0.1

        x0 = x - 0.5 * Lx
        y1 = y - 0.3 * Ly
        y2 = y - 0.7 * Ly

        dBy1 = 2 * dB * x0 * np.exp(-(x0**2 + y1**2) / (sigma) ** 2)
        dBy2 = -2 * dB * x0 * np.exp(-(x0**2 + y2**2) / (sigma) ** 2)

        return dBy1 + dBy2

    def bx(x, y):
        Lx = sim.simulation_domain()[0]
        Ly = sim.simulation_domain()[1]
        sigma = 1.0
        dB = 0.1

        x0 = x - 0.5 * Lx
        y1 = y - 0.3 * Ly
        y2 = y - 0.7 * Ly

        dBx1 = -2 * dB * y1 * np.exp(-(x0**2 + y1**2) / (sigma) ** 2)
        dBx2 = 2 * dB * y2 * np.exp(-(x0**2 + y2**2) / (sigma) ** 2)

        v1 = -1
        v2 = 1.0
        return v1 + (v2 - v1) * (S(y, Ly * 0.3, L) - S(y, Ly * 0.7, L)) + dBx1 + dBx2

    def bz(x, y):
        return 0.0

    def b2(x, y):
        return bx(x, y) ** 2 + by(x, y) ** 2 + bz(x, y) ** 2

    def T(x, y):
        K = 0.7
        temp = 1.0 / density(x, y) * (K - b2(x, y) * 0.5)
        assert np.all(temp > 0)
        return temp

    def vx(x, y):
        return 0.0

    def vy(x, y):
        return 0.0

    def vz(x, y):
        return 0.0

    def vthx(x, y):
        return np.sqrt(T(x, y))

    def vthy(x, y):
        return np.sqrt(T(x, y))

    def vthz(x, y):
        return np.sqrt(T(x, y))

    vvv = {
        "vbulkx": vx,
        "vbulky": vy,
        "vbulkz": vz,
        "vthx": vthx,
        "vthy": vthy,
        "vthz": vthz,
        "nbr_part_per_cell": 100,
    }

    ph.MaxwellianFluidModel(
        bx=bx,
        by=by,
        bz=bz,
        protons={"charge": 1, "density": density, **vvv, "init": {"seed": 12334}},
    )
    ph.ElectronModel(closure="isothermal", Te=0.0)

    for quantity in ["E", "B"]:
        ph.ElectromagDiagnostics(quantity=quantity, write_timestamps=timestamps)
    for quantity in ["charge_density", "mass_density", "bulkVelocity"]:
        ph.FluidDiagnostics(quantity=quantity, write_timestamps=timestamps)

    for quantity in ["density", "pressure_tensor"]:
        ph.FluidDiagnostics(
            quantity=quantity, write_timestamps=timestamps, population_name="protons"
        )

    ph.InfoDiagnostics(quantity="particle_count")

    ph.LoadBalancer(active=True, auto=True, mode="nppc", tol=0.05)

    return sim


def plot_file_for_qty(plot_dir, qty, time, extra=""):
    return f"{plot_dir}/harris_t{"{:.10f}".format(time)}_{qty}_{extra}.png"


test = AdvanceTestBase(rethrow=True)  # change to False for debugging images


def diff(new_time):
    print("diff", new_time)

    plot_dir = Path(f"{diag_dir}_plots") / str(cpp.mpi_size())
    plot_dir.mkdir(parents=True, exist_ok=True)
    run = Run(diag_dir)
    ranks = run.GetRanks(new_time)

    for ilvl in range(3):
        ranks.plot(
            filename=plot_file_for_qty(plot_dir, f"ranks", new_time, f"L{ilvl}"),
            plot_patches=True,
            levels=(ilvl,),
            dpi=2000,
        )

    differ = diff_hierarchy(run.GetB(new_time, all_primal=False))
    print("B max: ", differ.max())
    print("Bx max: ", differ.max("Bx"))
    print("By max: ", differ.max("By"))
    print("Bz max: ", differ.max("Bz"))
    if differ.has_non_zero():
        for c in ["x", "y", "z"]:
            for ilvl in range(3):
                differ.plot(
                    filename=plot_file_for_qty(
                        plot_dir, f"diffB{c}", new_time, f"L{ilvl}"
                    ),
                    plot_patches=True,
                    vmin=0,
                    vmax=+1e-16,
                    qty=f"B{c}",
                    levels=(ilvl,),
                    dpi=2000,
                )

    differ = diff_hierarchy(run.GetNi(new_time))
    print("ion charge rho max: ", differ.max())

    if differ.has_non_zero():
        for ilvl in range(3):
            differ.plot(
                filename=plot_file_for_qty(
                    plot_dir, f"ionCharge", new_time, f"L{ilvl}"
                ),
                plot_patches=True,
                vmin=0,
                vmax=+1e-16,
                levels=(ilvl,),
                dpi=2000,
            )

    differ = diff_hierarchy(run.GetMassDensity(new_time))
    print("ion mass rho max: ", differ.max())

    if differ.has_non_zero():
        for ilvl in range(3):
            differ.plot(
                filename=plot_file_for_qty(plot_dir, f"ionMass", new_time, f"L{ilvl}"),
                plot_patches=True,
                vmin=0,
                vmax=+1e-16,
                levels=(ilvl,),
                dpi=1000,
            )

    differ = diff_hierarchy(run.GetE(new_time, all_primal=False))
    print("E max: ", differ.max())
    print("Ex max: ", differ.max("Ex"))
    print("Ey max: ", differ.max("Ey"))
    print("Ez max: ", differ.max("Ez"))

    if differ.has_non_zero():
        for c in ["x", "y", "z"]:
            for ilvl in range(3):
                differ.plot(
                    filename=plot_file_for_qty(
                        plot_dir, f"diffE{c}", new_time, f"L{ilvl}"
                    ),
                    plot_patches=True,
                    vmin=0,
                    vmax=+1e-16,
                    qty=f"E{c}",
                    levels=(ilvl,),
                    dpi=2000,
                )


def get_time(path, time=None, datahier=None):
    if time is not None:
        time = "{:.10f}".format(time)
    from pyphare.pharesee.hierarchy import hierarchy_from

    datahier = hierarchy_from(h5_filename=path + "/EM_E.h5", times=time, hier=datahier)
    datahier = hierarchy_from(h5_filename=path + "/EM_B.h5", times=time, hier=datahier)
    return datahier


def post_advance(new_time):
    if cpp.mpi_rank() == 0:
        try:
            diff(new_time)

            # hier = get_time(diag_dir, new_time)
            # errors = test.base_test_overlaped_fields_are_equal(hier, new_time)
            # if isinstance(errors, list):
            #     print("ERROR AT TIME: ", new_time)
        except KeyError as e:
            err = str(e)
            if not "Unable to synchronously open object" in err:  # no diag for time
                import traceback

                print(f"Exception caught: \n{e}")
                print(traceback.format_exc())

    cpp.mpi_barrier()


def check_diags():
    run = Run(diag_dir)
    for time in run.all_times()["B"]:
        post_advance(time)
        # break


class HarrisTest(SimulatorTest):
    def __init__(self, *args, **kwargs):
        super(HarrisTest, self).__init__(*args, **kwargs)
        self.simulator = None

    def tearDown(self):
        super(HarrisTest, self).tearDown()
        if self.simulator is not None:
            self.simulator.reset()
        self.simulator = None
        ph.global_vars.sim = None

    def test_run(self):
        Simulator(config()).run().reset()
        check_diags()
        return self


if __name__ == "__main__":
    startMPI()
    HarrisTest().test_run().tearDown()
