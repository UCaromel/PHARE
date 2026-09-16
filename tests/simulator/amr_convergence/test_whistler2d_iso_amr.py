#!/usr/bin/env python3
"""AMR spatial convergence: isotropic-grid (dx=dy) variant of
test_whistler2d_amr, 2D rotated Hall-MHD whistler wave (Toth, JCP 227, 2008,
Table 2). alpha=45deg makes dx=dy=L*sqrt(2)/N exactly, complementing the
anisotropic gate in test_whistler2d_amr (alpha=arctan(0.5)~=26.56deg,
dy/dx=2) with a regression check on the equal-mesh path through the
anisotropic ADPT divB touch-up (adpt_magnetic_refine_patch_strategy.hpp).
See amr_convergence_base for the protocol and compute_errors for the norm.

Reconstruction is fixed at WENOZ -- the whistler is dispersive, so Linear
would not reach order 2 and could not expose an order defect.

Requires the O4 Hall SSPRK4_5+WENOZ AMR permutation
  2,O4,SSPRK4_5,WENOZ,None,Rusanov,true,false,false  (in res/sim/amr_convergence.txt
  and res/sim/whistler2d.txt).
"""

import os
import unittest

import numpy as np

import pyphare.pharein as ph
from tests.simulator.amr_convergence.amr_convergence_base import ConvergenceTestBase

os.environ.setdefault("PHARE_SCOPE_TIMING", "0")

ph.NO_GUI()

# Isotropic whistler: alpha = 45 deg (unlike the standard gate's
# arctan(0.5) ~= 26.56 deg), which makes dx = dy exactly.
tan_alpha = 1.0
cosalpha = 1.0 / np.sqrt(1 + tan_alpha**2)
sinalpha = tan_alpha / np.sqrt(1 + tan_alpha**2)

B0, RHO0, P0 = 1.0, 1.0, 1.0
GAMMA = 5.0 / 3.0
DELTA = 1e-3  # wave amplitude (Toth dB/B0 = 1e-3)

c_A = B0 / np.sqrt(RHO0)
c_s = np.sqrt(GAMMA * P0 / RHO0)
v_fast = np.sqrt(c_s**2 + c_A**2)

# Toth k*d_i ~= 0.19 (PHARE d_i = 1 -> k = 0.19); one mode over the domain.
K = 0.19
L = 2 * np.pi / K  # wavelength = domain length along e1

OMEGA = 0.5 * K**2 + np.sqrt((K * c_A) ** 2 + (0.5 * K**2) ** 2)
C_W = OMEGA / K  # whistler phase speed
V_AMP = DELTA * c_A / C_W  # transverse velocity amplitude (Toth eq. 55)

# Wave frame (x-y plane tilt): e1 = k || B0, e2/e3 transverse.
e1 = np.array([cosalpha, sinalpha, 0.0])
e2 = np.array([-sinalpha, cosalpha, 0.0])
e3 = np.cross(e1, e2)  # = (0, 0, 1)
B0_vec = B0 * e1

# Lengthened rotated domain: one wavelength along e1 fits each periodic axis.
# At alpha=45deg, Lx == Ly -> dx == dy on an N x N grid.
Lx = L / cosalpha
Ly = L / sinalpha

TIMESTEPPER = "SSPRK4_5"
RECONSTRUCTION = "WENOZ"
LIMITER = "None"


class WhistlerIsoConvergenceTest(ConvergenceTestBase):
    name = "whistler2d_iso"
    final_time = L / C_W  # one traversal = 2*pi/omega = exact return to IC

    SPATIAL_NS = [32, 64, 128]
    SPATIAL_SIGMA = 0.4  # same operating point as the standard whistler gate
    SPATIAL_ORDER_BAND = (3.70, 4.30)

    def cfl_dt(self, N):
        """Per-unit-sigma stable step. Hall dispersive bound dt ~ dx^2/pi
        dominates at fine resolution; the fast-mode bound dx/v_fast caps it at
        coarse N. Uses the smaller grid spacing (worst case; here dx == dy)."""
        dxmin = min(Lx / N, Ly / N)
        return min(dxmin**2 / np.pi, dxmin / v_fast)

    def _common(self, mhd_order, N, n):
        return dict(
            smallest_patch_size=8,
            time_step=self.final_time / n,
            final_time=self.final_time,
            cells=(N, N),
            dl=(Lx / N, Ly / N),
            interp_order=1,
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
            hall=True,
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
        def phase(x, y):
            return K * (x * e1[0] + y * e1[1])  # t=0

        def dB(x, y):
            p = phase(x, y)
            # right-hand circularly polarized (Toth)
            return DELTA * (np.cos(p)[:, None] * e2 - np.sin(p)[:, None] * e3)

        def dV(x, y):
            p = phase(x, y)
            return V_AMP * (-np.cos(p)[:, None] * e2 + np.sin(p)[:, None] * e3)

        def density(x, y):
            return RHO0

        def p(x, y):
            return P0

        def vx(x, y):
            return dV(x, y)[:, 0]

        def vy(x, y):
            return dV(x, y)[:, 1]

        def vz(x, y):
            return dV(x, y)[:, 2]

        def bx(x, y):
            return B0_vec[0] + dB(x, y)[:, 0]

        def by(x, y):
            return B0_vec[1] + dB(x, y)[:, 1]

        def bz(x, y):
            return B0_vec[2] + dB(x, y)[:, 2]

        ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

        # dump the conserved set natively so the comparison variables are the
        # dumped cell/face averages directly.
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
