#!/usr/bin/env python3
"""A 2D super-magnetofast MHD flow around a cylinder.

A cylinder is placed in the domain.  A super-magnetofast inflow with a
guide field B_y is prescribed at the inlet.  Look at the pretty shocks!

@todo Does not work amr due to the ghost cell method for immersed boundaries.
"""

import numpy as np
import pyphare.pharein as ph
from pyphare import cpp
from pyphare.simulator.simulator import Simulator, startMPI

ph.NO_GUI()

Lx, Ly = 50., 50.
Nx, Ny = 500, 500
p_in = 1.0
rho_in = 1.0
cylinder_radius = 1.0
gamma = 5. / 3.
cfl = 0.20  # wrt the inlet state
B0 = 1.0
magnetofast_mach = 1.5

dx, dy = Lx / Nx, Ly / Ny
cells = (Nx, Ny)
dl = (dx, dy)
domain_size = (Lx, Ly) 
center = (domain_size[0] / 3, domain_size[1] / 2)
xB = 0.5 * center[0]
deltaB = 5 * dl[0]
radius = 1.0
amr_nlevels = 1 # does not work with amr for the moment

sound_speed = np.sqrt(gamma * p_in / rho_in)
alfven_speed = B0 / np.sqrt(rho_in)
magnetofast_speed = np.sqrt(sound_speed ** 2 + alfven_speed ** 2)
u_in = magnetofast_mach * sound_speed
time_step = cfl * dl[0] / (u_in + magnetofast_speed)

dump_period = 0.1 * cylinder_radius / magnetofast_speed
dump_niter_period = int(np.ceil(dump_period / time_step)) 
final_time = domain_size[0] / magnetofast_speed
time_step_nbr = int(final_time / time_step)

timestamps = np.arange(0., final_time, dump_niter_period * time_step)

def step(x, delta):
    return 0.5 * (1.0 + np.tanh(x / delta))

def config():
    sim = ph.Simulation(
        nesting_buffer=3,
        smallest_patch_size=10,
        time_step=time_step,
        time_step_nbr=time_step_nbr,
        cells=cells,
        dl=dl,
        interp_order=2,
        refinement="tagging",
        max_mhd_level=amr_nlevels,
        max_nbr_levels=amr_nlevels,
        hall=False,
        res=False,
        hyper_res=False,
        hyper_resistivity=0.0,
        resistivity=0.0,
        diag_options={
            "format": "pharevtkhdf",
            "options": {"dir": f"phare_outputs/cfl_{cfl:.3f}_amr_{amr_nlevels}_nprocs_{cpp.mpi_size()}", "mode": "overwrite", "allow_emergency_dump": True},
        },
        strict=True,
        eta=0.0,
        nu=0.0,
        gamma=5.0 / 3.0,
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="TVDRK3",
        model_options=["MHDModel"],
        inner_boundary={
            "name": "cylinder",
            "shape": "sphere",
            "center": list(center),
            "radius": radius,
        },
        boundary_types=("physical", "periodic"),
        boundary_conditions = {
            "xlower": {
                "type": "super-magnetofast-inflow",
                "data": {
                    "density": 1.0,
                    "pressure": 1.0,
                    "B": [0.0, B0, 0.0],
                    "velocity": u_in,
                }, 
            },
            "xupper": {"type": "super-magnetofast-outflow"},
        },
    )

    def density(x, y):
        return 1.0

    def vx(x, y):
        return u_in

    def vy(x, y):
        return 0.0

    def vz(x, y):
        return 0.0

    def bx(x, y):
        return 0.0

    def by(x, y):
        return B0 * step(x=(xB - x), delta=deltaB)

    def bz(x, y):
        return 0.0

    def p(x, y):
        return 0.5 * (by(x, y) ** 2 - B0 ** 2) + p_in

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    # Inner boundary mesh data, dumped once at init.
    ph.MHDDiagnostics(quantity="IBSignedDistance", write_timestamps=[0.0])
    ph.MHDDiagnostics(quantity="IBCellStatus", write_timestamps=[0.0])

    # MHD quantities.
    ph.MHDDiagnostics(quantity="rho", write_timestamps=timestamps)
    ph.MHDDiagnostics(quantity="P", write_timestamps=timestamps)
    ph.MHDDiagnostics(quantity="Etot", write_timestamps=timestamps)
    ph.MHDDiagnostics(quantity="V", write_timestamps=timestamps)
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    ph.ElectromagDiagnostics(quantity="divB", write_timestamps=timestamps)

    return sim


def main():
    Simulator(config()).run().reset()
    ph.global_vars.sim = None


if __name__ == "__main__":
    startMPI()
    main()
