import numpy as np
import pyphare.mock_mhd_simulator.mhd_model as md
import pyphare.mock_mhd_simulator.simulation as s
from pyphare.mock_mhd_simulator.simulator import MHDMockSimulator


def config(nx, Dx, Dt):

    cells = (nx,)
    dl = (Dx,)
    lx = cells[0] * dl[0]
    k = 2 * np.pi / lx

    m = 1

    kt = 2 * np.pi / lx * m
    w = (kt**2 / 2) * (np.sqrt(1 + 4 / kt**2) + 1)
    nbr_periods = 10
    final_time = (2 * np.pi / w) * nbr_periods

    sim = s.Simulation(
        ndim=1,
        order=1,
        timestep=Dt,
        final_time=final_time,
        cells=cells,
        dl=dl,
        origin=(0.0,),
        eta=0.0,
        nu=0.0,
        gamma=5.0 / 3.0,
        reconstruction="weno3",
        limiter="",
        riemann="rusanov",
        time_integrator="tvdrk3",
        hall=True,
    )

    np.random.seed(0)

    modes = [1, 2, 4, 8]
    phases = np.random.rand(len(modes))

    def density(x):
        return 1.0

    def vx(x):
        return 0.0

    def vy(x):
        return sum(-np.cos(k * x * m + phi) * 1e-2 * k for m, phi in zip(modes, phases))

    def vz(x):
        return sum(np.sin(k * x * m + phi) * 1e-2 * k for m, phi in zip(modes, phases))

    def bx(x):
        return 1.0

    def by(x):
        return sum(np.cos(k * x * m + phi) * 1e-2 for m, phi in zip(modes, phases))

    def bz(x):
        return sum(-np.sin(k * x * m + phi) * 1e-2 for m, phi in zip(modes, phases))

    def p(x):
        return 1.0

    md.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    return sim


def main():
    configs = {
        "low": (128, 0.8, 0.077),
        "high": (128, 0.05, 0.0006),
    }

    for name, (nx, Dx, Dt) in configs.items():
        mhd = MHDMockSimulator(config(nx, Dx, Dt)).run(f"whistler1D_{name}.h5")
        mhd.clear_simulation()


if __name__ == "__main__":
    main()
