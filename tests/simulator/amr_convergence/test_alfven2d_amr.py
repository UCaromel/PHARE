#!/usr/bin/env python3
"""AMR space-time convergence: 2D oblique circularly polarized Alfven wave
(Toth, JCP 161, 2000). CP Alfven wave, alpha=30deg, v_A=1, period T=1; the
exact state at t=T equals the initial condition. See amr_convergence_base for
the protocol and compute_errors for the norm.

Requires the build permutation
  2,1,4,SSPRK4_5,WENOZ,None,Rusanov,false,false,false  (in res/sim/all.txt).
"""

import os
import unittest

import numpy as np
from ddt import data, ddt

import pyphare.pharein as ph
from tests.simulator.amr_convergence.amr_convergence_base import ConvergenceTestBase

os.environ.setdefault("PHARE_SCOPE_TIMING", "0")

ph.NO_GUI()

alpha = 30.0 * np.pi / 180.0
cosalpha = np.cos(alpha)
sinalpha = np.sin(alpha)

GAMMA = 5.0 / 3.0
P0, RHO0, DB, DV = 0.1, 1.0, 0.1, 0.1

TIMESTEPPER = "SSPRK4_5"
RECONSTRUCTION = "WENOZ"
LIMITER = "None"


@ddt
class AlfvenConvergenceTest(ConvergenceTestBase):
    name = "alfven2d"
    final_time = 1.0  # one period (v_A = 1, wavelength 1 along e1)

    # N=16 is preasymptotic (calibration 2026-07-09, then-TVDRK2+Linear:
    # pairwise order 1.44 on the 16->32 segment vs 1.82 on 32->64); the band
    # was calibrated on [32, 64, 128]. TVDRK2+Linear measurements (kaa
    # 2026-07-09/13): order=2 slope 1.91, order=4 slope 1.92, legacy 1.58.
    SPATIAL_NS = [32, 64, 128]
    SPATIAL_SIGMA = 0.32  # the historical dt = 0.2/N convention
    # Per-order bands since the anisotropic ADPT touch-up (1/D_c-weighted
    # subzoneDiv + D_c prefactor, derivation S11-S15): the old shared cap at 2
    # was the touch-up's unweighted divergence proxy on dy/dx = tan(30deg),
    # not the C-F time interpolation. Measured kaa 2026-07-18 (30 ranks):
    # order=4 slope 4.02 (segments 4.03/4.00) -- matching the iso
    # discriminators; order=2 slope 3.36 (3.41/3.32), mixed Linear2-kernel /
    # 4th-order-scheme regime, still drifting across segments, so its band is
    # deliberately loose (lower edge only needs to beat the old 2.25 cap to
    # stay discriminating).
    SPATIAL_ORDER_BANDS = {2: (2.50, 4.30), 4: (3.70, 4.30)}
    # legacy (order-0) gate keeps the defect-era band it is expected to fail
    SPATIAL_ORDER_BAND = (1.70, 2.25)

    # sigma sweep upward from the spatially-dominated operating point toward
    # the stability limit, at fixed N. Gate = drift of the AMR/uniform error
    # ratio (see base class). N=64 rather than 32: the smaller spatial floor
    # makes any dt-dependent C-F error stand out more. Same sigma set as the
    # whistler test.
    SWEEP_N = 64
    SWEEP_SIGMAS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # calibration 2026-07-09 (then-TVDRK2+Linear, N=64): ratio drifts 0.0097
    # (composite/uniform) and 0.0205 (L1-fine/uniform) on kaa at 30 ranks
    # (0.0034 single-rank) while raw errors drift ~15% (TVDRK2's own dt^2
    # error). At sigma=0.9 TVDRK2 sat near its stability bound and the UNIFORM
    # control's own temporal error broke the ratio's common-mode assumption
    # (kaa 2026-07-13: ratio flat for 5 sigmas then a lone 0.748 outlier) --
    # one motivation for moving to SSPRK4_5+WENOZ, whose dt^4 error stays far
    # below the spatial floor across the whole span (as in the whistler test,
    # ratio drift 3e-4 there). The C-F time-interp defect this family
    # diagnosed produced a ~6% error drift over a comparable sigma span
    # (J-refiner sigma-sweep study, 2026-07-07); the gate sits between.
    MAX_AMR_SIGMA_DRIFT = 0.05

    def cfl_dt(self, N):
        """Multi-D LLF sum-form CFL bound dt_CFL = 1/(c/dx + c/dy), with the
        direction-independent worst case c = |u|max + sqrt(cs^2 + vA^2) over
        the CP-Alfven init (|B|^2 = 1 + DB^2, rho = 1)."""
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

        # dump the conserved set natively so the comparison variables are the
        # dumped cell/face averages directly (no primitive->conserved
        # reconstruction in the analysis).
        timestamps = [0.0, self.final_time]
        ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
        for quantity in ["rho", "rhoV", "Etot"]:
            ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)

    @data(2, 4)  # Linear2 / Cubic4 runtime kernels
    def test_spatial_convergence(self, refinement_order):
        self.check_spatial_order(
            refinement_order, self.SPATIAL_NS, self.SPATIAL_SIGMA,
            self.SPATIAL_ORDER_BANDS[refinement_order],
        )

    # the legacy default refinement is really an order-1 refinement (order-0
    # polynomial; linear only on non-shared edges/faces): measured 1.58 here
    # (segments 1.61 -> 1.56), so it cannot meet the 2nd-order band --
    # documented defect, expected to fail until the legacy op is replaced
    # (an unexpected success flags that it was).
    @unittest.expectedFailure
    def test_spatial_convergence_legacy(self):
        self.check_spatial_order(
            0, self.SPATIAL_NS, self.SPATIAL_SIGMA, self.SPATIAL_ORDER_BAND
        )

    def test_temporal_sigma_sweep(self):
        self.check_sigma_sweep(order=2, N=self.SWEEP_N, sigmas=self.SWEEP_SIGMAS)


if __name__ == "__main__":
    unittest.main()
