#!/usr/bin/env python3
import os

import numpy as np
from pathlib import Path

import pyphare.pharein as ph
from pyphare import cpp
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator, startMPI

from tests.simulator import SimulatorTest

os.environ["PHARE_SCOPE_TIMING"] = "1"  # turn on scope timing

ph.NO_GUI()

cells = (4096, 4096)
time_step = 0.002
final_time = 4000

dump_frequency = 1000

diag_dir = "phare_outputs/mhd_harris"

hall = True
res = True
hyper_res = True

def config():
    L = 12

    sim = ph.Simulation(
        time_step=time_step,
        final_time=final_time,
        cells=cells,
        dl=(0.2, 0.2),
        interp_order=2,
        refinement="tagging",
        max_mhd_level=1,
        max_nbr_levels=1,
        hyper_resistivity=0.0,
        resistivity=0.0,
        diag_options={
            "format": "phareh5",
            "options": {"dir": diag_dir, "mode": "overwrite", "allow_emergency_dumps": True},
        },
        restart_options={
            "dir": "checkpoints",
            "mode": "overwrite",
            "elapsed_timestamps": [10000, 20000, 40000, 60000, 80000],
        },  # "restart_time":start_time },
        strict=True,
        nesting_buffer=1,
        hyper_mode="constant",
        eta=0.0008,
        nu=1.75e-4,
        gamma=5.0 / 3.0,
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="SSPRK4_5",
        hall=hall,
        res=res,
        hyper_res=hyper_res,
        model_options=["MHDModel"],
    )

    sim = ph.global_vars.sim

    dt = sim.time_step
    t0 = sim.start_time()
    tf = sim.final_time

    n0 = int(np.rint(t0 / dt))
    n1 = int(np.rint(tf / dt))

    if t0 != 0.0:
        n0 += 1

    step_numbers = np.arange(n0, n1+1, dump_frequency, dtype=np.int64)
    timestamps = dt * step_numbers

    N_modes = int(1./(10* sim.dl[1]))
    np.random.seed(0)
    phases = np.random.uniform(0, 2 * np.pi, N_modes)
    modes = np.arange(1, N_modes + 1)


    def S(y, y0, l):
        return 0.5 * (1.0 + np.tanh((y - y0) / l))

    def vx(x, y):
        return 0.0

    def vy(x, y):
        return 0.0

    def vz(x, y):
        return 0.0

    def bx(x, y):
        Lx = sim.simulation_domain()[0]
        Ly = sim.simulation_domain()[1]

        psi0 = 1.6e-2
        y1 = 0.25 * Ly
        y2 = 0.75 * Ly

        kx = 2 * np.pi / Lx
        ky = 2 * np.pi / Ly

        term1 = ky * np.sin(ky * (y - y1))
        term2 = -ky * np.sin(ky * (y - y2))

        dBx = psi0 * np.cos(kx * x) * (term1 + term2)

        v1 = -1.0
        v2 = 1.0
        return v1 + (v2 - v1) * (S(y, y1, L) - S(y, y2, L)) + dBx

    def by(x, y):
        Lx = sim.simulation_domain()[0]
        Ly = sim.simulation_domain()[1]

        psi0 = 1.6e-2
        y1 = 0.25 * Ly
        y2 = 0.75 * Ly

        kx = 2 * np.pi / Lx
        ky = 2 * np.pi / Ly

        term1 = np.cos(ky * (y - y1))
        term2 = -np.cos(ky * (y - y2))

        dBy = -psi0 * kx * np.sin(kx * x) * (term1 + term2)

        return dBy

    def bz(x, y):
        return 0.0

    def p(x, y):
        return 1.5 - (bx(x, y) ** 2) / 2.0

    def density(x, y): 
        return p(x,y) 
 
    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)

    for quantity in ["rho", "rhoV", "Etot"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)

    return sim


def plot_file_for_qty(plot_dir, qty, time):
    return f"{plot_dir}/harris_{qty}_t{time}.png"


def plot(diag_dir, plot_dir):
    run = Run(diag_dir)
    for time in timestamps:
        run.GetDivB(time).plot(
            filename=plot_file_for_qty(plot_dir, "divb", time),
            plot_patches=True,
            vmin=-1e-11,
            vmax=1e-11,
        )
        run.GetRanks(time).plot(
            filename=plot_file_for_qty(plot_dir, "Ranks", time), plot_patches=True
        )
        run.GetMHDrho(time).plot(
            filename=plot_file_for_qty(plot_dir, "rho", time), plot_patches=True
        )
        for c in ["x", "y", "z"]:
            run.GetMHDV(time).plot(
                filename=plot_file_for_qty(plot_dir, f"v{c}", time),
                plot_patches=True,
                qty=f"{c}",
            )
            run.GetB(time).plot(
                filename=plot_file_for_qty(plot_dir, f"b{c}", time),
                plot_patches=True,
                qty=f"{c}",
            )
        run.GetMHDP(time).plot(
            filename=plot_file_for_qty(plot_dir, "p", time), plot_patches=True
        )
        if hall:
            run.GetJ(time).plot(
                filename=plot_file_for_qty(plot_dir, "jz", time),
                qty="z",
                plot_patches=True,
            )


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
        # self.register_diag_dir_for_cleanup(diag_dir)
        Simulator(config()).run().reset()
        # if cpp.mpi_rank() == 0:
        #     plot_dir = Path(f"{diag_dir}_plots") / str(cpp.mpi_size())
        #     plot_dir.mkdir(parents=True, exist_ok=True)
        #     plot(diag_dir, plot_dir)
        cpp.mpi_barrier()
        return self


if __name__ == "__main__":
    startMPI()
    HarrisTest().test_run().tearDown()


