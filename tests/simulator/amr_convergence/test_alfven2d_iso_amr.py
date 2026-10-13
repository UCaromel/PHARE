#!/usr/bin/env python3
"""AMR spatial convergence: isotropic-grid (dx=dy) variant of
test_alfven2d_amr, 2D oblique circularly polarized Alfven wave (Toth, JCP
161, 2000). alpha=45deg makes dx=dy=sqrt(2)/N exactly, complementing the
anisotropic gate in test_alfven2d_amr (alpha=30deg, dy/dx~1.73) with a
regression check on the equal-mesh path through the anisotropic ADPT divB
touch-up (adpt_magnetic_refine_patch_strategy.hpp). See
amr_convergence_base for the protocol and compute_errors for the norm.

Requires the O4 ideal SSPRK4_5+WENOZ AMR permutation
  2,O4,SSPRK4_5,WENOZ,None,Rusanov,false,false,false  (in res/sim/amr_convergence.txt).
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
    final_time = 1.0  # one period (v_A = 1, wavelength 1 along e1)

    SPATIAL_NS = [32, 64, 128]
    SPATIAL_SIGMA = 0.32  # the historical dt = 0.2/N convention
    SPATIAL_ORDER_BAND = (3.70, 4.30)

    def cfl_dt(self, N):
        """Multi-D LLF sum-form CFL bound dt_CFL = 1/(c/dx + c/dy), with the
        direction-independent worst case c = |u|max + sqrt(cs^2 + vA^2) over
        the CP-Alfven init (|B|^2 = 1 + DB^2, rho = 1)."""
        dx = (1.0 / N) / cosalpha
        dy = (1.0 / N) / sinalpha
        c = DV + np.sqrt(GAMMA * P0 / RHO0 + (1.0 + DB**2) / RHO0)
        return 1.0 / (c / dx + c / dy)

    def _common(self, mhd_order, N, n):
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
            mhd_order=mhd_order,
            hall=False,
            res=False,
            hyper_res=False,
            model_options=["MHDModel"],
        )

    def amr_simulation(self, mhd_order, N, n):
        tag = f"o{mhd_order}"
        base = f"phare_outputs/{self.name}_amr_convergence/{tag}_N{N}_n{n}"
        return self.simulation(
            refinement="boxes",
            refinement_boxes={"L0": {"B0": self.fine_box(N)}},
            max_mhd_level=2,
            diag_options={
                "format": "phareh5",
                "options": {"dir": base, "mode": "overwrite"},
            },
            **self._common(mhd_order, N, n),
        )

    def uniform_simulation(self, mhd_order, N, n):
        base = f"phare_outputs/{self.name}_amr_convergence/uniform_o{mhd_order}_N{N}_n{n}"
        return self.simulation(
            refinement="tagging",
            max_mhd_level=1,
            max_nbr_levels=1,
            diag_options={
                "format": "phareh5",
                "options": {"dir": base, "mode": "overwrite"},
            },
            **self._common(mhd_order, N, n),
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

        # dump the conserved set natively so the comparison variables are the
        # dumped cell/face averages directly (no primitive->conserved
        # reconstruction in the analysis).
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
