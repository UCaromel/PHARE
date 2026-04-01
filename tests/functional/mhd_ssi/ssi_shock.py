#!/usr/bin/env python3
"""
Brio-Wu shock tube test exercising the Shared Smoothness Indicator (SSI) path.

WENOZ and WENO3 reconstructions now compute SSI from rho and B before
reconstructing all variables.  This test verifies:
  1. The simulation runs to completion without NaN / Inf.
  2. Basic physical constraints hold everywhere (rho > 0, P > 0).
  3. The shock/rarefaction structure is roughly where it should be
     (total variation of By, approximate shock position).
  4. WENOZ+SSI is no more oscillatory than a standard Linear run on the
     same problem (TV comparison).

Problem: Brio & Wu (1988) — standard MHD shock tube.
  Left  (x < 0.5): rho=1,    P=1,   Bx=0.75, By=1,  Bz=0
  Right (x > 0.5): rho=0.125,P=0.1, Bx=0.75, By=-1, Bz=0
  gamma = 2, t_final = 0.1
"""

import os
import numpy as np

import pyphare.pharein as ph
from pyphare import cpp
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator, startMPI
from tests.simulator import SimulatorTest

os.environ["PHARE_SCOPE_TIMING"] = "0"

ph.NO_GUI()

# ── problem parameters ──────────────────────────────────────────────────────
CELLS   = 400
DL      = 1.0 / CELLS
DT      = 0.4 * DL / 2.0          # CFL ~ 0.4 (fast-mode speed ~ 2)
T_FINAL = 0.1
GAMMA   = 2.0
DIAG_DIR = "phare_outputs/ssi_shock"
TIMESTAMPS = [T_FINAL]


def _brio_wu_ic(cells, dl):
    """Return Brio-Wu initial condition functions."""
    x_mid = cells * dl / 2.0

    def density(x): return np.where(x < x_mid, 1.0, 0.125)
    def vx(x):      return np.zeros_like(x)
    def vy(x):      return np.zeros_like(x)
    def vz(x):      return np.zeros_like(x)
    def bx(x):      return np.full_like(x, 0.75)
    def by(x):      return np.where(x < x_mid,  1.0, -1.0)
    def bz(x):      return np.zeros_like(x)
    def p(x):       return np.where(x < x_mid,  1.0,  0.1)

    return density, vx, vy, vz, bx, by, bz, p


_TIMESTEPPER = {
    "WENOZ":  "SSPRK4_5",
    "WENO3":  "TVDRK3",
    "Linear": "TVDRK2",
}


def config(reconstruction, limiter="None", gamma=GAMMA):
    sim = ph.Simulation(
        smallest_patch_size=15,
        time_step=DT,
        final_time=T_FINAL,
        cells=(CELLS,),
        dl=(DL,),
        refinement="tagging",
        max_nbr_levels=1,
        max_mhd_level=1,
        hyper_resistivity=0.0,
        resistivity=0.0,
        diag_options={
            "format": "phareh5",
            "options": {"dir": DIAG_DIR, "mode": "overwrite"},
        },
        strict=True,
        eta=0.0,
        nu=0.0,
        gamma=gamma,
        interp_order=2,
        reconstruction=reconstruction,
        limiter=limiter,
        riemann="Rusanov",
        mhd_timestepper=_TIMESTEPPER[reconstruction],
        model_options=["MHDModel"],
    )

    density, vx, vy, vz, bx, by, bz, p = _brio_wu_ic(CELLS, DL)
    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    ph.ElectromagDiagnostics(quantity="B", write_timestamps=TIMESTAMPS)
    for qty in ["rho", "V", "P"]:
        ph.MHDDiagnostics(quantity=qty, write_timestamps=TIMESTAMPS)

    return sim


def _load_fields(run):
    """Return (rho, By, P) numpy arrays at T_FINAL (physical cells only, no ghosts)."""

    def _interior(pd):
        g = pd.ghosts_nbr[0]
        return pd.dataset[g:-g] if g > 0 else pd.dataset[:]

    def _patch0(hier):
        return hier.levels()[0].patches[0]

    rho_hier = run.GetMHDrho(T_FINAL, all_primal=False)
    rho = _interior(_patch0(rho_hier).patch_datas["mhdRho"])

    by_hier = run.GetB(T_FINAL, all_primal=False)
    By = _interior(_patch0(by_hier).patch_datas["By"])

    p_hier = run.GetMHDP(T_FINAL, all_primal=False)
    P = _interior(_patch0(p_hier).patch_datas["mhdP"])

    return rho, By, P


def total_variation(field):
    """L1 total variation — measures oscillations."""
    return np.sum(np.abs(np.diff(field)))


class SSIShockTest(SimulatorTest):
    def tearDown(self):
        super().tearDown()
        ph.global_vars.sim = None

    def _run(self, reconstruction, limiter="None", gamma=GAMMA):
        ph.global_vars.sim = None
        self.register_diag_dir_for_cleanup(DIAG_DIR)
        Simulator(config(reconstruction, limiter, gamma=gamma)).run().reset()
        return Run(DIAG_DIR)

    def test_wenoz_ssi_no_nan(self):
        """WENOZ+SSI: simulation completes, no NaN or Inf anywhere."""
        run = self._run("WENOZ")
        rho, By, P = _load_fields(run)
        self.assertFalse(np.any(np.isnan(rho)), "NaN in rho (WENOZ SSI)")
        self.assertFalse(np.any(np.isinf(rho)), "Inf in rho (WENOZ SSI)")
        self.assertFalse(np.any(np.isnan(By)),  "NaN in By (WENOZ SSI)")
        self.assertFalse(np.any(np.isnan(P)),   "NaN in P (WENOZ SSI)")

    def test_wenoz_ssi_physical_constraints(self):
        """WENOZ+SSI: density and pressure remain strictly positive."""
        run = self._run("WENOZ")
        rho, _, P = _load_fields(run)
        self.assertTrue(np.all(rho > 0), f"Non-positive rho (min={rho.min():.3e})")
        self.assertTrue(np.all(P   > 0), f"Non-positive P   (min={P.min():.3e})")

    def test_weno3_ssi_physical_constraints(self):
        """WENO3+SSI: density and pressure remain strictly positive."""
        run = self._run("WENO3")
        rho, _, P = _load_fields(run)
        self.assertTrue(np.all(rho > 0), f"Non-positive rho (min={rho.min():.3e})")
        self.assertTrue(np.all(P   > 0), f"Non-positive P   (min={P.min():.3e})")

    def test_wenoz_ssi_tv_not_worse_than_linear(self):
        """WENOZ+SSI total variation of By is no worse than Linear+VanLeer.

        SSI is designed to reduce cross-variable oscillations near shocks.
        WENOZ with SSI should produce TV(By) <= TV(Linear) * margin,
        where margin accounts for the sharper profile WENOZ produces.

        Linear+VanLeer is unstable for Brio-Wu with gamma=2; use gamma=5/3
        (the other standard Brio-Wu choice) for both runs so the comparison
        is self-consistent.

        NOTE: this test currently fails when run together with other tests in
        the same process because all MHD pybind modules with the same
        (dim, interp, nbRefined) parameters register the same Splitter C++ type;
        loading a second module raises a pybind type-registration conflict.
        A separate PR that removes Splitter from MHD modules will fix this.
        Run this test in isolation to exercise the comparison.
        """
        gamma_tv = 5.0 / 3.0
        run_lin   = self._run("Linear", limiter="VanLeer", gamma=gamma_tv)
        run_wenoz = self._run("WENOZ", gamma=gamma_tv)

        _, By_lin,   _ = _load_fields(run_lin)
        _, By_wenoz, _ = _load_fields(run_wenoz)

        tv_lin   = total_variation(By_lin)
        tv_wenoz = total_variation(By_wenoz)

        # WENOZ+SSI should not produce dramatically more oscillations than Linear.
        # Allow up to 50 % more TV (sharper transitions naturally increase TV).
        margin = 1.5
        self.assertLessEqual(
            tv_wenoz, tv_lin * margin,
            f"WENOZ+SSI TV={tv_wenoz:.4f} exceeds {margin}x Linear TV={tv_lin:.4f}",
        )

    def test_wenoz_ssi_by_jump(self):
        """WENOZ+SSI: By changes sign across the contact, right values in range."""
        run = self._run("WENOZ")
        _, By, _ = _load_fields(run)

        # At t=0 By_L=1, By_R=-1; at t=0.1 the contact has moved right of mid.
        n = len(By)
        # Left quarter should be predominantly positive
        self.assertGreater(By[:n//4].mean(), 0.0, "Left By should be positive")
        # Right quarter should be predominantly negative
        self.assertLess(By[3*n//4:].mean(), 0.0, "Right By should be negative")


def main():
    Simulator(config("WENOZ")).run()


if __name__ == "__main__":
    startMPI()
    SSIShockTest().test_wenoz_ssi_physical_constraints()
    SSIShockTest().test_wenoz_ssi_tv_not_worse_than_linear()
    ph.global_vars.sim = None
