#!/usr/bin/env python3
"""
Convergence test for WENOZ+SSI and WENO3+SSI on a smooth Alfvén wave.

Verifies that the Shared Smoothness Indicator path does not degrade the
formal convergence order of the reconstruction scheme for smooth problems.
Expected orders (theory):
  WENO3 + SSI  →  ~3
  WENOZ + SSI  →  ~4  (CWENO4 optimal)

Initial condition: small-amplitude Alfvén wave propagating along Bx.
  rho=1, Bx=1, P=0.1, gamma=5/3
  vy = ε cos(2π x),  By = ε cos(2π x)   (ε = 1e-4)
After one period (t_final = 1 / c_A with c_A = Bx/sqrt(rho) = 1) the wave
should return exactly to the initial state → error = |By_final - By_initial|.
"""

import os
import numpy as np

import pyphare.pharein as ph
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator
from tests.simulator import SimulatorTest

os.environ["PHARE_SCOPE_TIMING"] = "0"

ph.NO_GUI()

# ── problem parameters ──────────────────────────────────────────────────────
EPS      = 1e-4
BX0      = 1.0
RHO0     = 1.0
P0       = 0.1
GAMMA    = 5.0 / 3.0
T_FINAL  = 1.0            # one Alfvén period (c_A = 1)
DT_BASE  = 2e-3           # conservative; refined grids halve dt too
DIAG_DIR = "phare_outputs/ssi_convergence"

# Reconstruction → (expected_order, limiter, timestepper)
SCHEMES = {
    "WENO3": (3.0, "None", "TVDRK3"),
    "WENOZ": (4.0, "None", "SSPRK4_5"),
}

TOLERANCE  = 0.2    # relative tolerance on measured vs expected slope
N_LEVELS   = 4      # number of grid refinements
N0         = 32     # coarsest grid


def config(reconstruction, limiter, timestepper, nx, dt):
    dx = 1.0 / nx
    sim = ph.Simulation(
        smallest_patch_size=15,
        time_step=dt,
        final_time=T_FINAL,
        cells=(nx,),
        dl=(dx,),
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
        gamma=GAMMA,
        interp_order=2,
        reconstruction=reconstruction,
        limiter=limiter,
        riemann="Rusanov",
        mhd_timestepper=timestepper,
        model_options=["MHDModel"],
    )

    def density(x): return np.full_like(x, RHO0)
    def vx(x):      return np.zeros_like(x)
    def vy(x):      return  EPS * np.cos(2 * np.pi * x)
    def vz(x):      return np.zeros_like(x)
    def bx(x):      return np.full_like(x, BX0)
    def by(x):      return  EPS * np.cos(2 * np.pi * x)
    def bz(x):      return np.zeros_like(x)
    def p(x):       return np.full_like(x, P0)

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    ph.ElectromagDiagnostics(quantity="B", write_timestamps=[0.0, T_FINAL])

    return sim


def l1_error(run):
    """L1 error: |By(T_FINAL) - By(0)|."""
    from pyphare.pharesee.hierarchy.hierarchy_utils import single_patch_for_LO

    def get_by(t):
        return (
            single_patch_for_LO(run.GetB(t, all_primal=False).By)
            .levels()[0].patches[0].patch_datas["By"].dataset[:]
        )

    by0 = get_by(0.0)
    byf = get_by(T_FINAL)
    return np.mean(np.abs(byf - by0))


def measure_order(reconstruction, limiter, timestepper):
    """Run N_LEVELS grid refinements and return the measured convergence slope."""
    nx   = N0
    dt   = DT_BASE
    dx_vals, err_vals = [], []

    for _ in range(N_LEVELS):
        ph.global_vars.sim = None
        Simulator(config(reconstruction, limiter, timestepper, nx, dt)).run().reset()
        run = Run(DIAG_DIR)
        err = l1_error(run)
        dx_vals.append(1.0 / nx)
        err_vals.append(err)
        nx *= 2
        dt /= 2.0

    log_dx  = np.log(dx_vals)
    log_err = np.log(err_vals)
    slope, _ = np.polyfit(log_dx, log_err, 1)
    return slope, dx_vals, err_vals


class SSIConvergenceTest(SimulatorTest):
    def setUp(self):
        super().setUp()
        self.register_diag_dir_for_cleanup(DIAG_DIR)

    def tearDown(self):
        super().tearDown()
        ph.global_vars.sim = None

    def _check_order(self, reconstruction):
        expected_order, limiter, timestepper = SCHEMES[reconstruction]
        slope, dx_vals, err_vals = measure_order(
            reconstruction, limiter, timestepper
        )
        rel_err = abs(slope - expected_order) / expected_order
        self.assertLess(
            rel_err,
            TOLERANCE,
            f"{reconstruction}+SSI: measured order {slope:.2f}, "
            f"expected {expected_order:.1f} (rel error {rel_err:.2%})",
        )

    def test_weno3_ssi_convergence_order(self):
        """WENO3+SSI achieves ~3rd-order convergence on smooth Alfvén wave."""
        self._check_order("WENO3")

    def test_wenoz_ssi_convergence_order(self):
        """WENOZ+SSI achieves ~4th-order convergence on smooth Alfvén wave."""
        self._check_order("WENOZ")


def main():
    import sys
    rec = sys.argv[1] if len(sys.argv) > 1 else "WENOZ"
    assert rec in SCHEMES, f"Unknown reconstruction '{rec}'. Choose from {list(SCHEMES)}"
    expected_order, limiter, timestepper = SCHEMES[rec]
    slope, dx_vals, err_vals = measure_order(rec, limiter, timestepper)
    print(f"{rec}+SSI: measured order = {slope:.3f}  (expected ~{expected_order})")
    for dx, err in zip(dx_vals, err_vals):
        print(f"  dx={dx:.4f}  L1={err:.2e}")


if __name__ == "__main__":
    main()
