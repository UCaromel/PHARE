#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

import pyphare.pharein as ph
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator

ph.NO_GUI()

diag_dir = "phare_outputs/whistler_convergence"

# --- Whistler parameters ---
# normalized by d_i, so d_i = 1 in code units
d_i    = 1.0
B0     = 1.0
rho0   = 1.0
k      = 2 * np.pi        # one mode, L=1 domain
delta  = 1e-6             # small amplitude

# Whistler dispersion (normalized, R-wave branch propagating along B0):
#   omega = (k^2 / 2) + k * sqrt(1 + (k/2)^2)   [in d_i-normalized units]
# This is the exact dispersion for the R-circularly-polarized whistler.
def whistler_omega(k, d_i=1.0):
    return 0.5 * d_i * k**2 + np.sqrt(k**2 + (0.5 * d_i * k**2)**2)

omega = whistler_omega(k, d_i)
T_whistler = 2 * np.pi / omega

c_w = omega / k
v_amp = delta * B0 / (rho0 * c_w)

print(f"omega = {omega:.6f}")
print(f"T_whistler = {T_whistler:.6f}")
print(f"v_amp = {v_amp:.6f}")

# run for exactly one period
final_time = T_whistler
# small enough timestep - whistler waves can be stiff, check CFL
# dt < dx / v_whistler_phase = dx * k / omega
# start conservative
time_step = 5.e-5

timestamps = [0.0, final_time]

reconstruction = "WENOZ"   # use your highest order reconstruction
limiter        = "None"
mhd_timestepper = "SSPRK4_5"   # need high order time integration too


def exact_by(x, t):
    return delta * np.cos(k * x - omega * t)

def exact_bz(x, t):
    return -delta * np.sin(k * x - omega * t)

def exact_vy(x, t):
    return -v_amp * np.cos(k * x - omega * t)

def exact_vz(x, t):
    return  v_amp * np.sin(k * x - omega * t)


def config(nx, dx):

    sim = ph.Simulation(
        smallest_patch_size=nx,   # single patch for clean convergence measurement
        time_step=time_step,
        final_time=final_time,
        cells=(nx,),
        dl=(dx,),
        refinement="tagging",
        max_mhd_level=1,
        max_nbr_levels=1,
        interp_order=2,
        hyper_resistivity=0.0,
        resistivity=0.0,
        hall=True,          # <-- enable Hall
        hyper_res=True,
        diag_options={
            "format": "phareh5",
            "options": {"dir": diag_dir, "mode": "overwrite"},
        },
        strict=True,
        nesting_buffer=1,
        hyper_mode="constant",
        eta=0.0,
        nu=0.0,
        gamma=5.0 / 3.0,
        reconstruction=reconstruction,
        limiter=limiter,
        riemann="Rusanov",
        mhd_timestepper=mhd_timestepper,
        model_options=["MHDModel"],
    )

    def density(x):
        return rho0

    def vx(x):
        return 0.0

    def vy(x):
        return exact_vy(x, 0.0)

    def vz(x):
        return exact_vz(x, 0.0)

    def bx(x):
        return B0

    def by(x):
        return exact_by(x, 0.0)

    def bz(x):
        return exact_bz(x, 0.0)

    def p(x):
        return 1.0   # plasma beta ~ 1, above whistler scale so ok

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=timestamps)
    ph.FluidDiagnostics(quantity="V", write_timestamps=timestamps)   # to check v too

    return sim


def compute_error(run, t, nx, dx, nghosts=6):
    from pyphare.pharesee.hierarchy.hierarchy_utils import single_patch_for_LO

    By_num = single_patch_for_LO(
        run.GetB(t, all_primal=False).By
    ).levels()[0].patches[0].patch_datas["By"].dataset[nghosts:-nghosts]

    By_ex = single_patch_for_LO(
        run.GetB(0., all_primal=False).By
    ).levels()[0].patches[0].patch_datas["By"].dataset[nghosts:-nghosts]

    return np.sum(np.abs(By_num - By_ex)) / len(By_num)


def main():
    Nx0 = 32
    Dx0 = 1.0 / Nx0

    dx_values, errors = [], []

    Nx, Dx = Nx0, Dx0
    # 5 refinement levels: 32 -> 512
    for _ in range(3):
        ph.global_vars.sim = None
        Simulator(config(Nx, Dx)).run().reset()
        run = Run(diag_dir)
        error = compute_error(run, final_time, Nx, Dx)
        dx_values.append(Dx)
        errors.append(error)
        print(f"Nx={Nx:4d}  dx={Dx:.4e}  L2 error={error:.4e}")
        Dx /= 2.0
        Nx *= 2

    log_dx     = np.log(dx_values)
    log_errors = np.log(errors)
    slope, intercept = np.polyfit(log_dx, log_errors, 1)

    print(f"\nConvergence slope: {slope:.3f}  (expected ~4 for WENOZ + RK4)")

    fitted = np.exp(intercept) * np.array(dx_values)**slope
    plt.figure(figsize=(8, 6))
    plt.loglog(dx_values, errors,  "o-",  label=f"L2 error (slope={slope:.2f})")
    plt.loglog(dx_values, fitted,  "--",  label="fit")
    plt.xlabel("Δx")
    plt.ylabel("Error (By)")
    plt.title(f"Whistler wave convergence  (ω={omega:.3f}, T={T_whistler:.3f})")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(f"{diag_dir}/whistler_convergence.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
