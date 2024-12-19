import numpy as np
import pyphare.mock_mhd_simulator.mhd_model as md
import pyphare.mock_mhd_simulator.simulation as s
from pyphare.mock_mhd_simulator.simulator import MHDMockSimulator


def config():

    cells = (128,)
    dl = (0.05,)
    lx = cells[0] * dl[0]
    k = 2 * np.pi / lx

    m = 4

    kt = 2 * np.pi / lx * m
    w = (kt**2 / 2) * (np.sqrt(1 + 4 / kt**2) + 1)
    final_time = 2 * np.pi / w

    sim = s.Simulation(
        ndim=1,
        order=1,
        timestep=0.0006,
        final_time=final_time,
        cells=cells,
        dl=dl,
        origin=(0.0,),
        eta=0.0,
        nu=0.0,
        gamma=5.0 / 3.0,
        terms="hall",
    )

    np.random.seed(0)

    modes = [4]
    phases = np.random.rand(len(modes))

    def density(x, y):
        return 1.0

    def vx(x, y):
        return 0.0

    def vy(x, y):
        ret = np.zeros((x.shape[0], y.shape[1]))

        for m, phi in zip(modes, phases):
            ret[:, :] += -np.cos(k * x * m + phi) * 1e-2 * k
        return ret

    def vz(x, y):
        ret = np.zeros((x.shape[0], y.shape[1]))
        for m, phi in zip(modes, phases):
            ret[:, :] += np.sin(k * x * m + phi) * 1e-2 * k
        return ret

    def bx(x, y):
        return 1.0

    def by(x, y):
        ret = np.zeros((x.shape[0], y.shape[1]))
        for m, phi in zip(modes, phases):
            ret[:, :] += np.cos(k * x * m + phi) * 1e-2
        return ret

    def bz(x, y):
        ret = np.zeros((x.shape[0], y.shape[1]))
        for m, phi in zip(modes, phases):
            ret[:, :] += -np.sin(k * x * m + phi) * 1e-2
        return ret

    def p(x, y):
        return 1.0

    md.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    return sim


def main():
    MHDMockSimulator(config()).run("whitsler1d.h5")


if __name__ == "__main__":
    main()