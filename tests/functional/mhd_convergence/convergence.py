import numpy as np
import pyphare.mock_mhd_simulator.mhd_model as m
import pyphare.mock_mhd_simulator.simulation as s
from pyphare.mock_mhd_simulator.simulator import MHDMockSimulator


def config(nx, Dx):
    sim = s.Simulation(
        ndim=1,
        order=2,
        timestep=5e-4,
        final_time=1.0,
        cells=(nx,),
        dl=(Dx,),
        origin=(0.0,),
        eta=0.0,
        nu=0.0,
        gamma=5.0 / 3.0,
        reconstruction="wenoz",
        limiter="",
        riemann="rusanov",
        time_integrator="tvdrk3",
    )

    kx = 2.0 * np.pi

    def density(x):
        return 1.0

    def vx(x):
        return 0.0

    def vy(x):
        return -1e-6 * np.cos(kx * x)

    def vz(x):
        return 0.0

    def bx(x):
        return 1.0

    def by(x):
        return 1e-6 * np.cos(kx * x)

    def bz(x):
        return 0.0

    def p(x):
        return 0.1

    m.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    return sim


def main():
    initial_Dx = 0.1
    initial_nx = 10
    Dx = initial_Dx
    nx = initial_nx
    step = 0

    while Dx > initial_Dx / 32.0:
        mhd = MHDMockSimulator(config(nx, Dx)).run(
            f"alfven1d{step}.h5", dumpfrequency=100
        )
        mhd.clear_simulation()
        step += 1
        Dx /= 2.0
        nx *= 2


if __name__ == "__main__":
    main()
