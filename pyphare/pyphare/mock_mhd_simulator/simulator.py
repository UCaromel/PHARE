from . import clearDict, populateDict


def pyMHD():
    import importlib

    return importlib.import_module("pybindlibs.pyMHD")


def make_cpp_simulator(
    dim,
    interp,
    time_integrator,
    reconstruction,
    riemann,
    hall=False,
    resistivity=False,
    hyper_resistivity=False,
    limiter="",
):
    import pybindlibs.pyMHD

    if hall:
        hall = "_hall"
    else:
        hall = ""

    if resistivity:
        resistivity = "_res"
    else:
        resistivity = ""

    if hyper_resistivity:
        hyper_resistivity = "_hyperres"
    else:
        hyper_resistivity = ""

    if limiter != "":
        limiter = f"_{limiter}"

    make_sim = f"make_mhd_mock_simulator_{dim}_{interp}_{time_integrator}_{reconstruction}{limiter}_{riemann}{hall}{resistivity}{hyper_resistivity}"
    return getattr(pyMHD(), make_sim)()


class MHDMockSimulator:
    def __init__(self, simulation):
        self.cpp_sim = None
        self.simulation = simulation

    def __del__(self):
        self.reset()

    def reset(self):
        if self.cpp_sim is not None:
            clearDict()
        self.cpp_sim = None
        return self

    def initialize(self):
        if self.cpp_sim is not None:
            raise ValueError(
                "Simulator already initialized: requires reset to re-initialize"
            )

        populateDict()

        hall = getattr(self.simulation, "hall", False)
        resistivity = getattr(self.simulation, "resistivity", False)
        hyper_resistivity = getattr(self.simulation, "hyper_resistivity", False)
        limiter = getattr(self.simulation, "limiter", "")

        self.cpp_sim = make_cpp_simulator(
            self.simulation.ndim,
            self.simulation.order,
            self.simulation.time_integrator,
            self.simulation.reconstruction,
            self.simulation.riemann,
            hall,
            resistivity,
            hyper_resistivity,
            limiter,
        )

    def _check_init(self):
        if self.cpp_sim is None:
            self.initialize()

    def run(self, filename, dumpfrequency=1):
        self._check_init()
        self.cpp_sim.advance(filename, dumpfrequency)
        return self

    def clear_simulation(self):
        self.simulation.clear()
        return self
