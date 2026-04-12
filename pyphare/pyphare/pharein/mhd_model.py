from . import global_vars


class MHDModel(object):
    def defaulter(self, input, value):
        if input is not None:
            import inspect

            params = list(inspect.signature(input).parameters.values())
            assert len(params)
            param_per_dim = len(params) == self.dim
            has_vargs = params[0].kind == inspect.Parameter.VAR_POSITIONAL
            assert param_per_dim or has_vargs
            return input
        if self.dim == 1:
            return lambda x: value + x * 0
        if self.dim == 2:
            return lambda x, y: value
        if self.dim == 3:
            return lambda x, y, z: value

    def __init__(
        self,
        density=None,
        vx=None,
        vy=None,
        vz=None,
        bx=None,
        by=None,
        bz=None,
        p=None,
        **kwargs,
    ):
        if global_vars.sim is None:
            raise RuntimeError("A simulation must be declared before a model")

        if global_vars.sim.model is not None:
            raise RuntimeError("A model is already created")

        self.dim = global_vars.sim.ndim

        density = self.defaulter(density, 1.0)
        vx = self.defaulter(vx, 1.0)
        vy = self.defaulter(vy, 0.0)
        vz = self.defaulter(vz, 0.0)
        bx = self.defaulter(bx, 1.0)
        by = self.defaulter(by, 0.0)
        bz = self.defaulter(bz, 0.0)
        p = self.defaulter(p, 1.0)

        self.model_dict = {}

        self.model_dict.update(
            {
                "density": density,
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "bx": bx,
                "by": by,
                "bz": bz,
                "p": p,
            }
        )

        self.populations = list(kwargs.keys())
        for population in self.populations:
            self.add_population(population, **kwargs[population])

        global_vars.sim.set_model(self)

    def nbr_populations(self):
        return len(self.populations)

    def add_population(
        self,
        name,
        charge=1.0,
        mass=1.0,
        nbr_part_per_cell=100,
        density=None,
        vbulkx=None,
        vbulky=None,
        vbulkz=None,
        vthx=None,
        vthy=None,
        vthz=None,
        init={},
        density_cut_off=1e-16,
    ):
        init["seed"] = init.get("seed", None)

        density = self.defaulter(density, 1.0)
        vbulkx = self.defaulter(vbulkx, 0.0)
        vbulky = self.defaulter(vbulky, 0.0)
        vbulkz = self.defaulter(vbulkz, 0.0)
        vthx = self.defaulter(vthx, 1.0)
        vthy = self.defaulter(vthy, 1.0)
        vthz = self.defaulter(vthz, 1.0)

        new_population = {
            name: {
                "charge": charge,
                "mass": mass,
                "density": density,
                "vx": vbulkx,
                "vy": vbulky,
                "vz": vbulkz,
                "vthx": vthx,
                "vthy": vthy,
                "vthz": vthz,
                "nbrParticlesPerCell": nbr_part_per_cell,
                "init": init,
                "density_cut_off": density_cut_off,
            }
        }

        if name in self.model_dict:
            raise ValueError("population already registered")

        self.model_dict.update(new_population)
