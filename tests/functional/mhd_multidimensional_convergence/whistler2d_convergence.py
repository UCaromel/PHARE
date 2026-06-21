#!/usr/bin/env python3
"""
2D Hall-MHD whistler-wave convergence test.

Derived directly from whistler_multid_convergence.py (the 3D version), collapsed
to two dimensions.  The wave is tilted in the x-y plane (angle alpha) so that
BOTH transverse flux directions are exercised non-trivially -- a grid-aligned
wave would be uniform in y and would not test 2D cross-coupling.

Coherence with the Toth (JCP 227, 2008) whistler IC is exact:
  - dispersion  omega = 0.5 k^2 + sqrt((k c_A)^2 + (0.5 k^2)^2)   (c_A = 1)
  - amplitude   v_amp = delta * c_A / c_w,  c_w = omega / k        (B0=rho0=1)
  - polarization dB = delta (cos phi e2 - sin phi e3)
                 dV = v_amp (-cos phi e2 + sin phi e3)
The only differences from the Toth script are the 3D-faithful conventions kept
here: k = 2*pi (one wavelength along e1 over the periodic domain), delta = 1e-6,
fixed dt, and the combined-conserved L2 error metric (not Toth eq. 60).

Periodic-rotated-wave domain follows the established PHARE pattern (cf.
mhd_alfven2d/alfven2d.py): Lx = 1/cos(alpha), Ly = 1/sin(alpha) so the phase
k*(x cos a + y sin a) advances exactly 2*pi across each axis.

final_time = 2*pi/omega is one full period, so the wave returns to its initial
state and |final - initial| is the valid error metric.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import pyphare.pharein as ph
from pyphare import cpp
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator

from tests.simulator import SimulatorTest

os.environ["PHARE_SCOPE_TIMING"] = "1"

ph.NO_GUI()

k = 2 * np.pi


def whistler_omega(k):
    return 0.5 * k**2 + np.sqrt(k**2 + (0.5 * k**2) ** 2)


omega = whistler_omega(k)

final_time = 2 * np.pi / omega
timestamps = [0.0, final_time]

# Fixed dt (as in the 3D whistler_multid_convergence.py): SSPRK4_5 keeps the
# temporal error negligible, and dt = 5e-5 stays inside the Hall dispersion
# stability bound dt <= dx^2/2 for all resolutions up to N=64.
time_step = 5.0e-5

reconstruction = "WENOZ"
limiter = "None"
mhd_timestepper = "SSPRK4_5"
ghosts = 6

# Tilt angle: 30 deg (matches mhd_alfven2d).  e1 = wave direction in x-y plane.
alpha = 30.0 * np.pi / 180.0
cos_a = np.cos(alpha)
sin_a = np.sin(alpha)

Lx = 1.0 / cos_a
Ly = 1.0 / sin_a


def config(N, diag_dir):
    sim = ph.Simulation(
        time_step=time_step,
        final_time=final_time,
        cells=(N, N),
        dl=(Lx / N, Ly / N),
        interp_order=1,
        refinement="tagging",
        max_mhd_level=1,
        max_nbr_levels=1,
        hyper_resistivity=0.0,
        resistivity=0.0,
        hall=True,
        hyper_res=False,
        diag_options={
            "format": "phareh5",
            "options": {"dir": diag_dir, "mode": "overwrite"},
        },
        strict=True,
        nesting_buffer=1,
        eta=0.0,
        nu=0.0,
        gamma=5.0 / 3.0,
        reconstruction=reconstruction,
        limiter=limiter,
        riemann="Rusanov",
        mhd_timestepper=mhd_timestepper,
        model_options=["MHDModel"],
    )

    # ---------------------------
    # Physical parameters
    # ---------------------------
    delta = 1e-6
    B0 = 1.0
    rho0 = 1.0

    c_w = omega / k
    v_amp = delta * B0 / (rho0 * c_w)

    # ---------------------------
    # Wave frame (x-y plane tilt)
    # ---------------------------
    e1 = np.array([cos_a, sin_a, 0.0])  # k || B0
    e2 = np.array([-sin_a, cos_a, 0.0])
    e3 = np.cross(e1, e2)  # = (0, 0, 1)

    B0_vec = B0 * e1

    # ---------------------------
    # Phase
    # ---------------------------
    def xi(x, y):
        return x * e1[0] + y * e1[1]

    def phase(x, y, t):
        return k * xi(x, y) - omega * t

    # ---------------------------
    # Fields
    # ---------------------------
    def density(x, y):
        return rho0

    def p(x, y):
        return 1.0

    def dB(x, y, t):
        ph = phase(x, y, t)
        cosph = np.cos(ph)
        sinph = np.sin(ph)
        return delta * (cosph[:, None] * e2 + sinph[:, None] * (-e3))

    def dV(x, y, t):
        ph = phase(x, y, t)
        cosph = np.cos(ph)
        sinph = np.sin(ph)
        return v_amp * (cosph[:, None] * (-e2) + sinph[:, None] * (e3))

    def vx(x, y):
        return dV(x, y, 0)[:, 0]

    def vy(x, y):
        return dV(x, y, 0)[:, 1]

    def vz(x, y):
        return dV(x, y, 0)[:, 2]

    def bx(x, y):
        return B0_vec[0] + dB(x, y, 0)[:, 0]

    def by(x, y):
        return B0_vec[1] + dB(x, y, 0)[:, 1]

    def bz(x, y):
        return B0_vec[2] + dB(x, y, 0)[:, 2]

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    ph.MHDDiagnostics(quantity="rho", write_timestamps=timestamps)
    ph.MHDDiagnostics(quantity="rhoV", write_timestamps=timestamps)
    ph.MHDDiagnostics(quantity="Etot", write_timestamps=timestamps)

    return sim


def _l1_component(getter, field, final_time, nghosts):
    """Interior-cell L1 error (mean |final - initial|) of one conserved component."""
    from pyphare.pharesee.hierarchy.hierarchy_utils import single_patch_for_LO

    g = nghosts
    computed = (
        single_patch_for_LO(getter(final_time, all_primal=False))
        .levels()[0]
        .patches[0]
        .patch_datas[field]
        .dataset[g:-g, g:-g]
    )
    expected = (
        single_patch_for_LO(getter(0.0, all_primal=False))
        .levels()[0]
        .patches[0]
        .patch_datas[field]
        .dataset[g:-g, g:-g]
    )
    return np.nanmean(np.abs(computed - expected))


def compute_error(run, final_time, nghosts=6):
    """Combined error: L2 norm of the per-component L1 errors across all
    conserved quantities (rhoV, B, rho, Etot) -- the standard MHD convergence
    metric (cf. whistler_multid_convergence.py)."""
    errors = []
    for comp in ("mhdRhoVx", "mhdRhoVy", "mhdRhoVz"):
        errors.append(_l1_component(run.GetMHDrhoV, comp, final_time, nghosts))
    for comp in ("Bx", "By", "Bz"):
        errors.append(_l1_component(run.GetB, comp, final_time, nghosts))
    errors.append(_l1_component(run.GetMHDrho, "mhdRho", final_time, nghosts))
    errors.append(_l1_component(run.GetMHDEtot, "mhdEtot", final_time, nghosts))
    return np.sqrt(np.sum(np.array(errors) ** 2))


def main():
    N_values = [16, 32, 64]
    h_values, errors = [], []

    for N in N_values:
        h = 1.0 / N
        diag_dir = f"phare_outputs/convergence_Whistler2d_{N}"

        ph.global_vars.sim = None
        Simulator(config(N, diag_dir)).run().reset()

        run = Run(diag_dir)
        error = compute_error(run, final_time)

        h_values.append(h)
        errors.append(error)
        print(f"  N={N:3d}  h=1/N={h:.4e}  combined_L1L2_error={error:.6e}")

    h_arr = np.array(h_values)
    err_arr = np.array(errors)

    print("\n--- pairwise slopes ---")
    for i in range(len(N_values) - 1):
        s = (np.log(errors[i]) - np.log(errors[i + 1])) / (
            np.log(h_values[i]) - np.log(h_values[i + 1])
        )
        print(f"  N={N_values[i]:3d}->{N_values[i+1]:3d}: {s:.2f}")

    slope, intercept = np.polyfit(np.log(h_arr), np.log(err_arr), 1)
    fitted = np.exp(intercept) * h_arr**slope
    print(f"\nCOMBINED overall slope: {slope:.2f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(h_arr, err_arr, "o-", label=f"Data (Slope: {slope:.2f})")
    ax.loglog(h_arr, fitted, "--", label="Fit")
    ax.set_title("Whistler 2D convergence (combined)", fontsize=18)
    ax.set_xlabel("1/N", fontsize=14)
    ax.set_ylabel("Error (combined L1, L2 norm)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    out_dir = f"phare_outputs/convergence_Whistler2d_{N_values[-1]}"
    plt.savefig(f"{out_dir}/convergence.png", dpi=200)


if __name__ == "__main__":
    main()
