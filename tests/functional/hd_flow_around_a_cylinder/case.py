#!/usr/bin/env python3
"""A 2D supersonic hydrodynamic flow around a cylinder.

A cylinder is placed in the domain. A supersonic inflow is prescribed at the
inlet, with zero magnetic field (pure HD regime of the MHD model). Look at
the pretty shocks!
"""

import numpy as np
import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator, startMPI

ph.NO_GUI()

p_in = 1.0
rho_in = 1.0
cylinder_radius = 1.0
gamma = 5. / 3.
cfl = 0.125  # wrt the inlet state
mach = 1.5

cells = (500, 500)
dl = (0.2, 0.2)
domain_size = (cells[0] * dl[0], cells[1] * dl[1])  # 10 x 10
center = (domain_size[0] / 3, domain_size[1] / 2)   # (5, 5)
radius = 1.0
amr_nlevels = 2

diag_dir = f"phare_outputs/cfl_{cfl:.3f}_amr_{amr_nlevels}"
speed_sound = np.sqrt(gamma * p_in / rho_in)
u_in = mach * speed_sound
time_step = cfl * dl[0] / (u_in + speed_sound)

dump_period = 0.5 * cylinder_radius / speed_sound
dump_niter_period = int(np.ceil(dump_period / time_step)) 
final_time = domain_size[0] / speed_sound

timestamps = np.arange(0., final_time, dump_niter_period * time_step)

def config():
    sim = ph.Simulation(
        smallest_patch_size=10,
        time_step=time_step,
        final_time=final_time,
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
            "options": {"dir": diag_dir, "mode": "overwrite"},
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
                    "B": [0.0, 0.0, 0.0],
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
        return 0.0

    def bz(x, y):
        return 0.0

    def p(x, y):
        return 1.0

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    # Inner boundary mesh data, dumped once at init.
    ph.MHDDiagnostics(quantity="IBSignedDistance", write_timestamps=[0.0])
    ph.MHDDiagnostics(quantity="IBCellStatus", write_timestamps=[0.0])

    # Hydrodynamic quantities.
    ph.MHDDiagnostics(quantity="rho", write_timestamps=timestamps)
    ph.MHDDiagnostics(quantity="P", write_timestamps=timestamps)
    ph.MHDDiagnostics(quantity="Etot", write_timestamps=timestamps)
    ph.MHDDiagnostics(quantity="V", write_timestamps=timestamps)
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)

    return sim


def main():
    Simulator(config()).run().reset()
    ph.global_vars.sim = None


if __name__ == "__main__":
    startMPI()
    main()
