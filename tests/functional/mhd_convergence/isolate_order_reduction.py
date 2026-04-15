#!/usr/bin/env python3
"""
Minimal test to isolate where order reduction occurs in 3D Hall MHD.

We test each step of the pipeline independently:
1. Ampere: J = curl(B) - does 4th order derivative give 4th order J?
2. Projection: edge J → cell center J - does tensor product projection preserve order?
3. WENO reconstruction: cell-centered J → flux interface J - is this 4th order?
4. UCT transverse reconstruction: flux-interface J → edge J - is this where order breaks?

Uses smooth analytic functions so we know exact solutions.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pyphare.pharein as ph
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator

ph.NO_GUI()

diag_dir = "phare_outputs/isolate_order"

# Smooth test function: B = sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z) type
# This gives smooth J = curl(B)

k = 2 * np.pi  # wavenumber


def exact_bx(x, y, z):
    """Bx = sin(ky)*sin(kz) - gives smooth curl"""
    return np.sin(k * y) * np.sin(k * z)


def exact_by(x, y, z):
    """By = sin(kx)*sin(kz)"""
    return np.sin(k * x) * np.sin(k * z)


def exact_bz(x, y, z):
    """Bz = sin(kx)*sin(ky)"""
    return np.sin(k * x) * np.sin(k * y)


def exact_jx(x, y, z):
    """Jx = dBz/dy - dBy/dz = k*cos(kx)*cos(ky) - k*sin(kx)*cos(kz)"""
    return k * np.cos(k * x) * np.cos(k * y) - k * np.sin(k * x) * np.cos(k * z)


def exact_jy(x, y, z):
    """Jy = dBx/dz - dBz/dx = k*sin(ky)*cos(kz) - k*cos(kx)*sin(ky)"""
    return k * np.sin(k * y) * np.cos(k * z) - k * np.cos(k * x) * np.sin(k * y)


def exact_jz(x, y, z):
    """Jz = dBy/dx - dBx/dy = k*cos(kx)*sin(kz) - k*cos(ky)*sin(kz)"""
    return k * np.cos(k * x) * np.sin(k * z) - k * np.cos(k * y) * np.sin(k * z)


def config_2d(nx, dx):
    """2D Hall MHD configuration - tests transverse reconstruction"""
    sim = ph.Simulation(
        smallest_patch_size=nx,
        time_step=1e-6,  # tiny timestep - we only care about initial state
        final_time=1e-6,
        cells=(nx, nx),
        dl=(dx, dx),
        refinement="tagging",
        max_mhd_level=1,
        max_nbr_levels=1,
        interp_order=2,
        hyper_resistivity=0.0,
        resistivity=0.0,
        hall=True,  # Enable Hall
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
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="Euler",
        model_options=["MHDModel"],
    )

    # 2D test functions: Bz = sin(kx)*sin(ky) -> Jx = k*sin(kx)*cos(ky), Jy = -k*cos(kx)*sin(ky)
    def density(x, y):
        return 1.0

    def vx(x, y):
        return 0.0

    def vy(x, y):
        return 0.0

    def vz(x, y):
        return 0.0

    def bx(x, y):
        return 1.0  # constant background

    def by(x, y):
        return 1.0  # constant background

    def bz(x, y):
        return 1e-3 * np.sin(k * x) * np.sin(k * y)  # small perturbation

    def p(x, y):
        return 1.0

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=[0.0])
    ph.ElectromagDiagnostics(quantity="J", write_timestamps=[0.0])

    return sim


def config_1d(nx, dx):
    """1D Hall MHD configuration for comparison"""
    sim = ph.Simulation(
        smallest_patch_size=nx,
        time_step=1e-6,
        final_time=1e-6,
        cells=(nx,),
        dl=(dx,),
        refinement="tagging",
        max_mhd_level=1,
        max_nbr_levels=1,
        interp_order=2,
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
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="Euler",
        model_options=["MHDModel"],
    )

    def density(x):
        return 1.0

    def vx(x):
        return 0.0

    def vy(x):
        return 0.0

    def vz(x):
        return 0.0

    def bx(x):
        return 1.0

    def by(x):
        return np.sin(k * x)

    def bz(x):
        return np.cos(k * x)

    def p(x):
        return 1.0

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=[0.0])
    ph.ElectromagDiagnostics(quantity="J", write_timestamps=[0.0])

    return sim


def get_patch_data(run, qty, comp, t=0.0):
    """Get field data from single patch, stripping ghosts"""
    from pyphare.pharesee.hierarchy.hierarchy_utils import single_patch_for_LO

    if qty == "B":
        hier = run.GetB(t, all_primal=False)
    elif qty == "J":
        hier = run.GetJ(t, all_primal=False)
    elif qty == "E":
        hier = run.GetE(t, all_primal=False)
    else:
        raise ValueError(f"Unknown quantity {qty}")

    field = getattr(hier, comp)
    patch_data = single_patch_for_LO(field).levels()[0].patches[0].patch_datas[comp]
    return patch_data.dataset[:], patch_data.x, getattr(patch_data, "y", None), getattr(
        patch_data, "z", None
    )


def compute_error_2d(run, comp, exact_func, nghosts=6):
    """Compute L1 error for a 2D field component"""
    data, x, y, _ = get_patch_data(run, "J", comp)

    # Strip ghosts
    data = data[nghosts:-nghosts, nghosts:-nghosts]
    x = x[nghosts:-nghosts]
    y = y[nghosts:-nghosts]

    # Compute exact solution on grid
    X, Y = np.meshgrid(x, y, indexing="ij")
    exact = exact_func(X, Y)

    error = np.mean(np.abs(data - exact))
    return error


def compute_error_3d(run, comp, exact_func, nghosts=6):
    """Compute L1 error for a 3D field component"""
    data, x, y, z = get_patch_data(run, "J", comp)

    # Strip ghosts
    data = data[nghosts:-nghosts, nghosts:-nghosts, nghosts:-nghosts]
    x = x[nghosts:-nghosts]
    y = y[nghosts:-nghosts]
    z = z[nghosts:-nghosts]

    # Compute exact solution on grid
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    exact = exact_func(X, Y, Z)

    error = np.mean(np.abs(data - exact))
    return error


def compute_error_1d(run, comp, exact_func, nghosts=6):
    """Compute L1 error for a 1D field component"""
    data, x, _, _ = get_patch_data(run, "J", comp)

    # Strip ghosts
    data = data[nghosts:-nghosts]
    x = x[nghosts:-nghosts]

    # For 1D with By=sin(kx), Bz=cos(kx):
    # Jy = -dBz/dx = k*sin(kx)
    # Jz = dBy/dx = k*cos(kx)
    exact = exact_func(x)

    error = np.mean(np.abs(data - exact))
    return error


def test_ampere_convergence_2d():
    """Test if Ampere (J = curl B) achieves expected order in 2D"""
    print("\n" + "=" * 60)
    print("TEST: Ampere (J = curl B) convergence in 2D")
    print("=" * 60)

    nx_values = [16, 32, 64]
    errors_jx, errors_jy = [], []
    dx_values = []

    # For Bz = sin(kx)*sin(ky), we have:
    # Jx = dBz/dy = k*sin(kx)*cos(ky)
    # Jy = -dBz/dx = -k*cos(kx)*sin(ky)
    amplitude = 1e-3

    for nx in nx_values:
        dx = 1.0 / nx
        dx_values.append(dx)

        ph.global_vars.sim = None
        sim = config_2d(nx, dx)
        Simulator(sim).run().reset()

        run = Run(diag_dir)
        errors_jx.append(compute_error_2d(run, "Jx", lambda x, y: amplitude * k * np.sin(k * x) * np.cos(k * y)))
        errors_jy.append(compute_error_2d(run, "Jy", lambda x, y: -amplitude * k * np.cos(k * x) * np.sin(k * y)))

        print(f"  nx={nx:3d}  dx={dx:.4f}  err_Jx={errors_jx[-1]:.2e}  err_Jy={errors_jy[-1]:.2e}")

    for name, errors in [("Jx", errors_jx), ("Jy", errors_jy)]:
        log_dx = np.log(dx_values)
        log_err = np.log(errors)
        slope, _ = np.polyfit(log_dx, log_err, 1)
        print(f"  {name} convergence order: {slope:.2f}")

    return dx_values, errors_jx, errors_jy


def test_ampere_convergence_3d():
    """Test if Ampere (J = curl B) achieves expected order in 3D"""
    print("\n" + "=" * 60)
    print("TEST: Ampere (J = curl B) convergence in 3D")
    print("=" * 60)

    nx_values = [16, 32, 64]
    errors_jx, errors_jy, errors_jz = [], [], []
    dx_values = []

    for nx in nx_values:
        dx = 1.0 / nx
        dx_values.append(dx)

        ph.global_vars.sim = None
        sim = config_3d(nx, dx)
        Simulator(sim).run().reset()

        run = Run(diag_dir)
        errors_jx.append(compute_error_3d(run, "Jx", exact_jx))
        errors_jy.append(compute_error_3d(run, "Jy", exact_jy))
        errors_jz.append(compute_error_3d(run, "Jz", exact_jz))

        print(f"  nx={nx:3d}  dx={dx:.4f}  err_Jx={errors_jx[-1]:.2e}  err_Jy={errors_jy[-1]:.2e}  err_Jz={errors_jz[-1]:.2e}")

    # Compute convergence orders
    for name, errors in [("Jx", errors_jx), ("Jy", errors_jy), ("Jz", errors_jz)]:
        log_dx = np.log(dx_values)
        log_err = np.log(errors)
        slope, _ = np.polyfit(log_dx, log_err, 1)
        print(f"  {name} convergence order: {slope:.2f}")

    return dx_values, errors_jx, errors_jy, errors_jz


def test_ampere_convergence_1d():
    """Test if Ampere achieves expected order in 1D (known to work)"""
    print("\n" + "=" * 60)
    print("TEST: Ampere (J = curl B) convergence in 1D (reference)")
    print("=" * 60)

    nx_values = [32, 64, 128]
    errors_jy, errors_jz = [], []
    dx_values = []

    for nx in nx_values:
        dx = 1.0 / nx
        dx_values.append(dx)

        ph.global_vars.sim = None
        sim = config_1d(nx, dx)
        Simulator(sim).run().reset()

        run = Run(diag_dir)
        # In 1D with By=sin(kx), Bz=cos(kx): Jy=-k*sin(kx), Jz=k*cos(kx)
        errors_jy.append(compute_error_1d(run, "Jy", lambda x: k * np.sin(k * x)))
        errors_jz.append(compute_error_1d(run, "Jz", lambda x: k * np.cos(k * x)))

        print(f"  nx={nx:3d}  dx={dx:.4f}  err_Jy={errors_jy[-1]:.2e}  err_Jz={errors_jz[-1]:.2e}")

    for name, errors in [("Jy", errors_jy), ("Jz", errors_jz)]:
        log_dx = np.log(dx_values)
        log_err = np.log(errors)
        slope, _ = np.polyfit(log_dx, log_err, 1)
        print(f"  {name} convergence order: {slope:.2f}")

    return dx_values, errors_jy, errors_jz


def main():
    print("Isolating order reduction in Hall MHD")
    print("========================================")

    # First test: Is J (from Ampere) 4th order accurate?
    # This tests: curl computation with 4th order derivatives
    test_ampere_convergence_1d()
    test_ampere_convergence_2d()

    # TODO: Add more targeted tests:
    # - Test projection operator alone
    # - Test WENO reconstruction alone
    # - Test UCT transverse reconstruction alone
    # - Compare E field evolution between 1D and 2D


if __name__ == "__main__":
    main()
