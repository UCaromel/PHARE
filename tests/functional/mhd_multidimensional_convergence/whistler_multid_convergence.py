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

# 2 things with this test: It does not handle mpi yet, and it only does one reconstruction at a time.
# For mpi, it would be possible but requires to deal with several patches and gather the data on rank 0.
# For the reconstructions, it would make sense when we will have a better way to compile all the reconstructions at once. see: https://github.com/PHAREHUB/PHARE/pull/1047

os.environ["PHARE_SCOPE_TIMING"] = "1"

ph.NO_GUI()

k = 2 * np.pi

def whistler_omega(k):
    return 0.5 * k**2 + np.sqrt(k**2 + (0.5 * k**2)**2)

omega = whistler_omega(k)

final_time = 2 * np.pi / omega
timestamps = [0.0, final_time]
diag_dir = "phare_outputs/convergence"

time_step = 2.e-4

reconstruction = "WENOZ"
limiter="None"
mhd_timestepper = "SSPRK4_5"
ghosts = 6


def config(nx):
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
            sinph[:, None]*e3
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

    return sim

# using by error is arbitrary now, add the error for everyone now
def compute_error(run, final_time, Nx, Dx, nghosts=6):
    from pyphare.pharesee.hierarchy.hierarchy_utils import single_patch_for_LO
    computed_by = single_patch_for_LO(run.GetB(final_time, all_primal=False).By).levels()[0].patches[0].patch_datas["By"].dataset[nghosts:-nghosts]

    expected_by = single_patch_for_LO(run.GetB(0., all_primal=False).By).levels()[0].patches[0].patch_datas["By"].dataset[nghosts:-nghosts]

    # expected_by = 1e-6 * np.cos(2 * np.pi * (coords - final_time))
    return np.sum(np.abs(computed_by - expected_by)) / len(computed_by)


def main():
    N_base = 16
    dx_values, errors, N_values = [], [], []

    while N_base <= 32:
        Nx, Ny, Nz = 2*N_base, N_base, N_base
        Dx, Dy, Dz = 3.0/Nx, 1.5/Ny, 1.5/Nz

        ph.global_vars.sim = None
        Simulator(config(N_base)).run().reset()

        run = Run(diag_dir)
        error = compute_error(run, final_time, Nx, Dx)

        dx_values.append(Dx)
        N_values.append(N_base)
        errors.append(error)

        N_base *= 2

    slope, intercept = np.polyfit(np.log(dx_values), np.log(errors), 1)

    fitted_line = np.exp(intercept) * dx_values**slope
    plt.figure(figsize=(10, 6))
    plt.loglog(dx_values, errors, "o-", label=f"Data (Slope: {slope:.2f})")
    plt.loglog(dx_values, fitted_line, "--", label="Fitted Line")
    plt.xlabel("Δx", fontsize=16)
    plt.ylabel("Error (L1 Norm)", fontsize=16)
    plt.title(f"Whistler", fontsize=20)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=20)
    plt.savefig(f"{diag_dir}/convergence.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
