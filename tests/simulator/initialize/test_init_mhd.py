#

import os
import numpy as np


import pyphare.pharein as ph
from pyphare.core import phare_utilities as phut
from pyphare.simulator.simulator import Simulator
from pyphare.pharesee.hierarchy import hierarchy_from

from tests.diagnostic import all_timestamps
from tests.simulator.test_initialization import InitializationTest


class MHDInitializationTest(InitializationTest):
    def getHierarchy(
        self,
        ndim,
        interp_order,  # torm?
        qty,
        refinement_boxes={},
        density=None,
        time_step_nbr=1,
        time_step=0.001,
        smallest_patch_size=None,
        largest_patch_size=10,
        cells=120,
        dl=0.1,
        hall=True,
        res=False,
        hyper_res=True,
        extra_diag_options=None,
        timestamps=None,
        diag_outputs="",
        **kwargs
    ):
        if smallest_patch_size is None:
            from pyphare.pharein.simulation import check_patch_size

            _, smallest_patch_size = check_patch_size(
                ndim, interp_order=interp_order, cells=cells
            )

        # ----------------------------------------------------------------------
        # simulation setup and running
        # ----------------------------------------------------------------------
        base_diag_dir = "phare_outputs/advance"
        base_diag_dir = (
            os.path.join(base_diag_dir, diag_outputs) if diag_outputs else base_diag_dir
        )
        extra_diag_options = extra_diag_options or dict()
        extra_diag_options["dir"] = base_diag_dir
        extra_diag_options["mode"] = "overwrite"
        sim = self.simulation(
            smallest_patch_size=smallest_patch_size,
            largest_patch_size=largest_patch_size,
            time_step_nbr=time_step_nbr,
            time_step=time_step,
            boundary_types=["periodic"] * ndim,
            cells=phut.np_array_ify(cells, ndim),
            dl=phut.np_array_ify(dl, ndim),
            interp_order=interp_order,
            refinement_boxes=refinement_boxes,
            diag_options={"format": "phareh5", "options": extra_diag_options},
            strict=True,
            nesting_buffer=1,
            hyper_mode="spatial",
            eta=0.0,
            nu=0.02,
            gamma=5.0 / 3.0,
            reconstruction="WENOZ",
            limiter="None",
            riemann="Rusanov",
            mhd_timestepper="TVDRK3",
            hall=hall,
            res=res,
            hyper_res=hyper_res,
            model_options=["MHDModel"],
        )
        diag_outputs = sim.diag_options["options"]["dir"]
        L = sim.simulation_domain()

        def _density(*xyz):
            hL = np.array(sim.simulation_domain()) / 2
            return 0.3 + np.exp(
                sum([-((xyz[i] - hL[i]) ** 2) for i in range(len(xyz))])
            )

        def bx(*xyz):
            return 1.0

        def by(*xyz):
            return np.asarray(
                [0.1 * np.cos(2 * np.pi * xyz[i] / L[i]) for i in range(len(xyz))]
            ).prod(axis=0)

        def bz(*xyz):
            return np.asarray(
                [0.1 * np.cos(2 * np.pi * xyz[i] / L[i]) for i in range(len(xyz))]
            ).prod(axis=0)

        def vx(*xyz):
            return np.asarray(
                [0.1 * np.cos(2 * np.pi * xyz[i] / L[i]) for i in range(len(xyz))]
            ).prod(axis=0)

        def vy(*xyz):
            return np.asarray(
                [0.1 * np.cos(2 * np.pi * xyz[i] / L[i]) for i in range(len(xyz))]
            ).prod(axis=0)

        def vz(*xyz):
            return np.asarray(
                [0.1 * np.cos(2 * np.pi * xyz[i] / L[i]) for i in range(len(xyz))]
            ).prod(axis=0)

        def p(*xyz):
            return 1.0

        density_fn = density or kwargs.get("density", _density)
        vx_fn = kwargs.get("vx", vx)
        vy_fn = kwargs.get("vy", vy)
        vz_fn = kwargs.get("vz", vz)
        p_fn = kwargs.get("p", p)

        user_total_magnetic = any(
            key in kwargs and kwargs[key] is not None for key in ("bx", "by", "bz")
        )
        user_perturbation_magnetic = any(
            key in kwargs and kwargs[key] is not None for key in ("b1x", "b1y", "b1z")
        )
        user_external_magnetic = any(
            key in kwargs and kwargs[key] is not None for key in ("b0x", "b0y", "b0z")
        )

        magnetic_kwargs = {
            key: kwargs[key]
            for key in ("b0x", "b0y", "b0z", "b1x", "b1y", "b1z")
            if key in kwargs and kwargs[key] is not None
        }

        if user_total_magnetic:
            magnetic_kwargs.update(
                {
                    "bx": kwargs.get("bx", bx),
                    "by": kwargs.get("by", by),
                    "bz": kwargs.get("bz", bz),
                }
            )
        elif not user_perturbation_magnetic and not user_external_magnetic:
            magnetic_kwargs.update({"bx": bx, "by": by, "bz": bz})

        ph.MHDModel(
            density=density_fn,
            vx=vx_fn,
            vy=vy_fn,
            vz=vz_fn,
            p=p_fn,
            **magnetic_kwargs,
        )

        if timestamps is None:
            timestamps = all_timestamps(ph.global_vars.sim)

        ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)

        for quantity in ["rho", "V", "P"]:
            ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)

        simulator = Simulator(sim)
        if kwargs.get("initialize_only", False):
            simulator.initialize()
        else:
            simulator.run()

        eb_hier = None
        if qty in ["b", "eb", "fields"]:
            eb_hier = hierarchy_from(
                h5_filename=diag_outputs + "/EM_B.h5", hier=eb_hier
            )
        if qty in ["e", "b", "eb"]:
            return eb_hier

        if qty == "moments" or qty == "fields":
            mom_hier = hierarchy_from(
                h5_filename=diag_outputs + "/ions_charge_density.h5", hier=eb_hier
            )
            mom_hier = hierarchy_from(
                h5_filename=diag_outputs + "/ions_bulkVelocity.h5", hier=mom_hier
            )
            return mom_hier
