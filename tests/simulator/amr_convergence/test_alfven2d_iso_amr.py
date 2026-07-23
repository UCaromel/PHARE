#!/usr/bin/env python3
"""ISOTROPIC-GRID discriminator for the AMR 2nd-order cap (2026-07-17 spatial
audit): identical to test_alfven2d_amr.py except alpha = 45 deg, which makes
dx = dy = sqrt(2)/N exactly. The ADPT divB touch-up
(adpt_magnetic_refine_patch_strategy.hpp) assumes dx = dy (equal-mesh closed
forms, S6b DECISION); the standard gate runs alpha = 30 deg (dy/dx ~ 1.73)
where the touch-up injects O(h^2) into fine B ghost faces. If that is the cap,
this square-cell run should measure well above 2; if it stays ~2, the
anisotropy hypothesis is dead for the dominant term.

Band is deliberately wide (informational run, not a regression gate).

Requires the build permutation
  2,1,4,SSPRK4_5,WENOZ,None,Rusanov,false,false,false  (in res/sim/all.txt).
"""

import os
import unittest

import numpy as np

import pyphare.pharein as ph
from tests.simulator.amr_convergence.amr_convergence_base import ConvergenceTestBase

os.environ.setdefault("PHARE_SCOPE_TIMING", "0")

ph.NO_GUI()

alpha = 45.0 * np.pi / 180.0
cosalpha = np.cos(alpha)
sinalpha = np.sin(alpha)

GAMMA = 5.0 / 3.0
P0, RHO0, DB, DV = 0.1, 1.0, 0.1, 0.1

TIMESTEPPER = "SSPRK4_5"
RECONSTRUCTION = "WENOZ"
LIMITER = "None"


class AlfvenIsoConvergenceTest(ConvergenceTestBase):
    name = "alfven2d_iso"
    final_time = 1.0

    SPATIAL_NS = [32, 64, 128]
    SPATIAL_SIGMA = 0.32
    # real gate since the anisotropic ADPT touch-up landed (was a wide
    # informational discriminator): measured 4.02 (segments 4.03/4.00) on kaa
    # 2026-07-17 and again 2026-07-18 (equal mesh is unchanged by the aniso
    # generalisation -- reduction identity).
    SPATIAL_ORDER_BAND = (3.70, 4.30)

    def cfl_dt(self, N):
        dx = (1.0 / N) / cosalpha
        dy = (1.0 / N) / sinalpha
        c = DV + np.sqrt(GAMMA * P0 / RHO0 + (1.0 + DB**2) / RHO0)
        return 1.0 / (c / dx + c / dy)

    def _common(self, N, n):
        return dict(
            smallest_patch_size=8,
            time_step=self.final_time / n,
            final_time=self.final_time,
            cells=(N, N),
            dl=((1.0 / N) / cosalpha, (1.0 / N) / sinalpha),
            hyper_resistivity=0.0,
            resistivity=0.0,
            strict=True,
            nesting_buffer=1,
            eta=0.0,
            nu=0.0,
            gamma=GAMMA,
            reconstruction=RECONSTRUCTION,
            limiter=LIMITER,
            riemann="Rusanov",
            mhd_timestepper=TIMESTEPPER,
            hall=False,
            res=False,
            hyper_res=False,
            model_options=["MHDModel"],
        )

    def amr_simulation(self, order, N, n):
        extra = {}
        if order != 0:
            extra["refinement_order"] = order
            extra["refinement_limiter"] = "none"
        base = f"phare_outputs/{self.name}_amr_convergence/o{order}_N{N}_n{n}"
        return self.simulation(
            refinement="boxes",
            refinement_boxes={"L0": {"B0": self.fine_box(N)}},
            max_mhd_level=2,
            diag_options={
                "format": "phareh5",
                "options": {"dir": base, "mode": "overwrite"},
            },
            **self._common(N, n),
            **extra,
        )

    def uniform_simulation(self, N, n):
        base = f"phare_outputs/{self.name}_amr_convergence/uniform_N{N}_n{n}"
        return self.simulation(
            refinement="tagging",
            max_mhd_level=1,
            max_nbr_levels=1,
            diag_options={
                "format": "phareh5",
                "options": {"dir": base, "mode": "overwrite"},
            },
            **self._common(N, n),
        )

    def add_model_and_diags(self):
        def density(x, y):
            return 1.0

        def phase(x, y):
            return 2 * np.pi * (x * cosalpha + y * sinalpha)

        def vx(x, y):
            return -DV * np.sin(phase(x, y)) * sinalpha

        def vy(x, y):
            return DV * np.sin(phase(x, y)) * cosalpha

        def vz(x, y):
            return DV * np.cos(phase(x, y))

        def bx(x, y):
            return cosalpha - DB * np.sin(phase(x, y)) * sinalpha

        def by(x, y):
            return sinalpha + DB * np.sin(phase(x, y)) * cosalpha

        def bz(x, y):
            return DB * np.cos(phase(x, y))

        def p(x, y):
            return P0

        ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

        timestamps = [0.0, self.final_time]
        ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
        for quantity in ["rho", "rhoV", "Etot"]:
            ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)

    def test_spatial_convergence_order4_isotropic(self):
        self.check_spatial_order(
            4, self.SPATIAL_NS, self.SPATIAL_SIGMA, self.SPATIAL_ORDER_BAND
        )


if __name__ == "__main__":
    unittest.main()
