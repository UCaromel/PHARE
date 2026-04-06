#!/usr/bin/env python3
"""Short-run 3D whistler convergence test.

Runs for n_steps timesteps (cheap) and measures L1 errors against the
analytical solution for all conservative variables: rho, rhoV, B, Etot.
Use this to diagnose spatial convergence order quickly before committing
to the expensive full-period test (whistler_multid_convergence.py).
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import pyphare.pharein as ph
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator

os.environ["PHARE_SCOPE_TIMING"] = "1"

ph.NO_GUI()

# ---------------------------
# Wave parameters
# ---------------------------
k = 2 * np.pi


def whistler_omega(k):
    return 0.5 * k**2 + np.sqrt(k**2 + (0.5 * k**2)**2)


omega = whistler_omega(k)

# Hall MHD whistler waves are dispersive: stability requires dt <= dx^2 / 2.
# Use dt=5e-5 to keep dt/dx^2 ~ 0.09 safely inside the stability region for
# all resolutions up to at least N=128.
time_step = 5.e-5

# Number of timesteps per run — keep small for a cheap spatial convergence
# check.  The error is measured against the analytical solution so any
# positive n_steps gives a valid convergence signal.
n_steps = 10
final_time = n_steps * time_step
timestamps = [0.0, final_time]

reconstruction = "WENOZ"
limiter = "None"
mhd_timestepper = "SSPRK4_5"
ghosts = 6
gamma = 5.0 / 3.0

# ---------------------------
# Physical parameters (module-level — needed in compute_error)
# ---------------------------
delta = 1e-6
B0 = 1.0
rho0 = 1.0

c_w = omega / k
v_amp = delta * B0 / (rho0 * c_w)

# ---------------------------
# Rotation vectors
# ---------------------------
sin_a = 2 / 3
cos_a = np.sqrt(1 - sin_a**2)
sin_b = 2 / np.sqrt(5)
cos_b = np.sqrt(1 - sin_b**2)

e1 = np.array([cos_a * cos_b, cos_a * sin_b, sin_a])   # k || B0
e2 = np.array([-sin_b, cos_b, 0.0])
e3 = np.cross(e1, e2)

B0_vec = B0 * e1


# ---------------------------
# Analytical solution
# [..., None] broadcasting works for any-shape x,y,z (1-D from init or 3-D
# from meshgrid).
# ---------------------------
def _xi(x, y, z):
    return x * e1[0] + y * e1[1] + z * e1[2]


def _phase(x, y, z, t):
    return k * _xi(x, y, z) - omega * t


def B_exact(x, y, z, t):
    ph = _phase(x, y, z, t)
    dB = delta * (np.cos(ph)[..., None] * e2 + np.sin(ph)[..., None] * e3)
    return B0_vec + dB


def rhoV_exact(x, y, z, t):
    ph = _phase(x, y, z, t)
    dV = v_amp * (-np.cos(ph)[..., None] * e2 + np.sin(ph)[..., None] * e3)
    return rho0 * dV


def Etot_exact(x, y, z, t):
    ph = _phase(x, y, z, t)
    dV = v_amp * (-np.cos(ph)[..., None] * e2 + np.sin(ph)[..., None] * e3)
    B = B_exact(x, y, z, t)
    return (1.0 / (gamma - 1.0)
            + 0.5 * rho0 * np.sum(dV**2, axis=-1)
            + 0.5 * np.sum(B**2, axis=-1))


# ---------------------------
# Simulation config
# ---------------------------
def config(nx, diag_dir):
    sim = ph.Simulation(
        time_step=time_step,
        final_time=final_time,
        cells=(2 * nx, nx, nx),
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
        gamma=gamma,
        reconstruction=reconstruction,
        limiter=limiter,
        riemann="Rusanov",
        mhd_timestepper=mhd_timestepper,
        model_options=["MHDModel"],
    )

    def density(x, y, z):
        return rho0 * np.ones_like(x)

    def p(x, y, z):
        return np.ones_like(x)

    def vx(x, y, z): return rhoV_exact(x, y, z, 0)[..., 0] / rho0
    def vy(x, y, z): return rhoV_exact(x, y, z, 0)[..., 1] / rho0
    def vz(x, y, z): return rhoV_exact(x, y, z, 0)[..., 2] / rho0

    def bx(x, y, z): return B_exact(x, y, z, 0)[..., 0]
    def by(x, y, z): return B_exact(x, y, z, 0)[..., 1]
    def bz(x, y, z): return B_exact(x, y, z, 0)[..., 2]

    ph.MHDModel(
        density=density,
        vx=vx, vy=vy, vz=vz,
        bx=bx, by=by, bz=bz,
        p=p,
    )

    ph.ElectromagDiagnostics(quantity="B",   write_timestamps=timestamps)
    ph.MHDDiagnostics(quantity="rho",        write_timestamps=timestamps)
    ph.MHDDiagnostics(quantity="rhoV",       write_timestamps=timestamps)
    ph.MHDDiagnostics(quantity="Etot",       write_timestamps=timestamps)

    return sim


# ---------------------------
# Error measurement
# ---------------------------
def _l1(pd, exact_3d, g):
    """L1 norm of (numerical - exact) after stripping g ghost layers."""
    num = pd.dataset[g:-g, g:-g, g:-g]
    ana = exact_3d[g:-g, g:-g, g:-g]
    return np.sum(np.abs(num - ana)) / num.size


def compute_error(run, t, nghosts=6):
    """Return dict of L1 errors vs analytical solution at time t."""
    from pyphare.pharesee.hierarchy.hierarchy_utils import single_patch_for_LO

    def _pd(hier_attr, key):
        return (single_patch_for_LO(hier_attr)
                .levels()[0].patches[0].patch_datas[key])

    errors = {}

    # B
    B_hier = run.GetB(t, all_primal=False)
    for comp, idx in (("Bx", 0), ("By", 1), ("Bz", 2)):
        pd = _pd(getattr(B_hier, comp), comp)
        X, Y, Z = np.meshgrid(pd.x, pd.y, pd.z, indexing='ij')
        errors[comp] = _l1(pd, B_exact(X, Y, Z, t)[..., idx], nghosts)

    # rho
    rho_hier = run.GetMHDrho(t, all_primal=False)
    pd = _pd(rho_hier.mhdRho, "mhdRho")
    X, Y, Z = np.meshgrid(pd.x, pd.y, pd.z, indexing='ij')
    errors["rho"] = _l1(pd, rho0 * np.ones_like(X), nghosts)

    # rhoV
    rhoV_hier = run.GetMHDrhoV(t, all_primal=False)
    for key, idx in (("mhdRhoVx", 0), ("mhdRhoVy", 1), ("mhdRhoVz", 2)):
        pd = _pd(getattr(rhoV_hier, key), key)
        X, Y, Z = np.meshgrid(pd.x, pd.y, pd.z, indexing='ij')
        errors[key] = _l1(pd, rhoV_exact(X, Y, Z, t)[..., idx], nghosts)

    # Etot
    Etot_hier = run.GetMHDEtot(t, all_primal=False)
    pd = _pd(Etot_hier.mhdEtot, "mhdEtot")
    X, Y, Z = np.meshgrid(pd.x, pd.y, pd.z, indexing='ij')
    errors["Etot"] = _l1(pd, Etot_exact(X, Y, Z, t), nghosts)

    return errors


# ---------------------------
# Main
# ---------------------------
def main():
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))

    N_base = 16
    dx_values, N_values = [], []
    qty_names = ["Bx", "By", "Bz", "rho", "mhdRhoVx", "mhdRhoVy", "mhdRhoVz", "Etot"]
    all_errors = {q: [] for q in qty_names}

    while N_base <= 64:
        Dx = 3.0 / (2 * N_base)
        diag_dir = f"phare_outputs/convergence_Whistler_short_{N_base}"

        ph.global_vars.sim = None
        Simulator(config(N_base, diag_dir)).run().reset()

        run = Run(diag_dir)
        errs = compute_error(run, final_time)

        dx_values.append(Dx)
        N_values.append(N_base)
        for q in qty_names:
            all_errors[q].append(errs[q])
            if rank == 0:
                print(f"  N={N_base:3d}  {q:12s}  error={errs[q]:.3e}")

        N_base *= 2

    if rank != 0:
        return

    # pairwise slopes
    dx_arr = np.array(dx_values)
    print("\n--- pairwise slopes ---")
    for q in qty_names:
        errs_arr = np.array(all_errors[q])
        parts = []
        for i in range(len(dx_arr) - 1):
            s = np.log(errs_arr[i] / errs_arr[i + 1]) / np.log(dx_arr[i] / dx_arr[i + 1])
            parts.append(f"N={N_values[i]}→{N_values[i+1]}: {s:.2f}")
        print(f"  {q:12s}  " + "  ".join(parts))

    # convergence plot
    ncols, nrows = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
    for ax, q in zip(axes.ravel(), qty_names):
        errs_arr = np.array(all_errors[q])
        slope, intercept = np.polyfit(np.log(dx_arr), np.log(errs_arr), 1)
        fitted = np.exp(intercept) * dx_arr**slope
        ax.loglog(dx_arr, errs_arr, "o-", label=f"slope={slope:.2f}")
        ax.loglog(dx_arr, fitted, "--")
        ax.set_title(q, fontsize=12)
        ax.set_xlabel("Δx")
        ax.set_ylabel("L1 error")
        ax.legend(fontsize=10)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.suptitle(
        f"Whistler 3D short convergence  (n_steps={n_steps}, dt={time_step})",
        fontsize=14,
    )
    plt.tight_layout()
    out_dir = f"phare_outputs/convergence_Whistler_short_{N_values[-1]}"
    plt.savefig(f"{out_dir}/convergence.png", dpi=200)
    print(f"Saved {out_dir}/convergence.png")


if __name__ == "__main__":
    main()
