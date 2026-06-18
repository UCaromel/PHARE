#!/usr/bin/env python3
import os

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import pyphare.pharein as ph
from pyphare import cpp
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator

from tests.simulator import SimulatorTest

os.environ["PHARE_SCOPE_TIMING"] = "1"

ph.NO_GUI()

k = 2 * np.pi

def whistler_omega(k):
    return 0.5 * k**2 + np.sqrt(k**2 + (0.5 * k**2)**2)

omega = whistler_omega(k)

final_time = 2 * np.pi / omega
timestamps = [0.0, final_time]

# Hall MHD whistler waves are dispersive: stability requires dt <= dx^2 / 2.
# At N=64, dx^2 ~ 5.5e-4, so dt=2e-4 was borderline (dt/dx^2 ~ 0.36).
# Use dt=5e-5 (matching the 1D whistler_convergence.py) to keep dt/dx^2 ~ 0.09
# safely inside the stability region for all resolutions up to at least N=128.
time_step = 5.e-5

reconstruction = "WENOZ"
limiter="None"
mhd_timestepper = "SSPRK4_5"
ghosts = 6


def config(nx, diag_dir):
    sim = ph.Simulation(
        time_step=time_step,
        final_time=final_time,
        cells=(2*nx, nx, nx),
        dl=(3.0 / (2 * nx), 1.5 / nx, 1.5 / nx),
        interp_order=2,
        refinement="tagging",
        max_mhd_level=1,
        max_nbr_levels=1,
        hyper_resistivity=0.0,
        resistivity=0.0,
        hall=True,
        hyper_res=True,
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
    # Rotation
    # ---------------------------
    sin_a = 2/3
    cos_a = np.sqrt(1 - sin_a**2)

    sin_b = 2/np.sqrt(5)
    cos_b = np.sqrt(1 - sin_b**2)

    e1 = np.array([cos_a*cos_b, cos_a*sin_b, sin_a])  # k || B0
    e2 = np.array([-sin_b, cos_b, 0.0])
    e3 = np.cross(e1, e2)

    B0_vec = B0 * e1

    # ---------------------------
    # Phase
    # ---------------------------
    def xi(x,y,z):
        return x*e1[0] + y*e1[1] + z*e1[2]

    def phase(x,y,z,t):
        return k * xi(x,y,z) - omega * t

    # ---------------------------
    # Fields
    # ---------------------------
    def density(x,y,z):
        return rho0

    def p(x,y,z):
        return 1.0

    def dB(x,y,z,t):
        ph = phase(x,y,z,t)
        cosph = np.cos(ph)
        sinph = np.sin(ph)

        return delta * (
            cosph[:, None]*e2 +
            sinph[:, None]*(-e3)
        )

    def dV(x,y,z,t):
        ph = phase(x,y,z,t)
        cosph = np.cos(ph)
        sinph = np.sin(ph)

        return v_amp * (
            cosph[:, None]*(-e2) +
            sinph[:, None]*( e3)
        )

    def vx(x,y,z):
        return dV(x,y,z,0)[:,0]

    def vy(x,y,z):
        return dV(x,y,z,0)[:,1]

    def vz(x,y,z):
        return dV(x,y,z,0)[:,2]

    def bx(x,y,z):
        return B0_vec[0] + dB(x,y,z,0)[:,0]

    def by(x,y,z):
        return B0_vec[1] + dB(x,y,z,0)[:,1]

    def bz(x,y,z):
        return B0_vec[2] + dB(x,y,z,0)[:,2]

    ph.MHDModel(
        density=density,
        vx=vx, vy=vy, vz=vz,
        bx=bx, by=by, bz=bz,
        p=p
    )

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
        .dataset[g:-g, g:-g, g:-g]
    )
    expected = (
        single_patch_for_LO(getter(0.0, all_primal=False))
        .levels()[0]
        .patches[0]
        .patch_datas[field]
        .dataset[g:-g, g:-g, g:-g]
    )
    return np.nanmean(np.abs(computed - expected))


def compute_error(run, final_time, nghosts=6):
    """Combined error: L2 norm of the per-component L1 errors across all
    conserved quantities (rhoV, B, rho, Etot) -- the standard MHD convergence
    metric (cf. convergence3d_plots.py)."""
    errors = []
    for comp in ("mhdRhoVx", "mhdRhoVy", "mhdRhoVz"):
        errors.append(_l1_component(run.GetMHDrhoV, comp, final_time, nghosts))
    for comp in ("Bx", "By", "Bz"):
        errors.append(_l1_component(run.GetB, comp, final_time, nghosts))
    errors.append(_l1_component(run.GetMHDrho, "mhdRho", final_time, nghosts))
    errors.append(_l1_component(run.GetMHDEtot, "mhdEtot", final_time, nghosts))
    return np.sqrt(np.sum(np.array(errors) ** 2))


def main():
    N_base = 16
    dx_values, N_values, errors = [], [], []

    while N_base <= 64:
        Nx = 2 * N_base
        Dx = 3.0 / Nx
        diag_dir = f"phare_outputs/convergence_Whistler_{N_base}"

        ph.global_vars.sim = None
        Simulator(config(N_base, diag_dir)).run().reset()

        run = Run(diag_dir)
        error = compute_error(run, final_time)

        dx_values.append(Dx)
        N_values.append(N_base)
        errors.append(error)
        print(f"  N={N_base:3d}  dx={Dx:.4e}  combined_L1L2_error={error:.6e}")

        N_base *= 2

    # --- combined convergence slope (single fit over the L2-combined error) ---
    dx_arr = np.array(dx_values)
    err_arr = np.array(errors)
    slope, intercept = np.polyfit(np.log(dx_arr), np.log(err_arr), 1)
    fitted = np.exp(intercept) * dx_arr**slope
    print(f"COMBINED overall slope: {slope:.2f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(dx_arr, err_arr, "o-", label=f"Data (Slope: {slope:.2f})")
    ax.loglog(dx_arr, fitted, "--", label="Fit")
    ax.set_title("Whistler 3D convergence (combined)", fontsize=18)
    ax.set_xlabel("Δx", fontsize=14)
    ax.set_ylabel("Error (combined L1, L2 norm)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    out_dir = f"phare_outputs/convergence_Whistler_{N_values[-1]}"
    plt.savefig(f"{out_dir}/convergence.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
