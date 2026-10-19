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

# Runs at the 4 ranks CMakeLists.txt registers: every rank reads all patches back and
# single_patch_for_LO assembles the whole domain, so the error below is rank-complete.

os.environ["PHARE_SCOPE_TIMING"] = "1"

ph.NO_GUI()

mode_speed = {
    "Alfven": 1.0,
    "Fast": 2.0,
    "Slow": 0.5,
    "Entropy": 1.0,
}

mode = "Alfven"
final_time = 1.0 / mode_speed[mode]
# a whole number of steps per period: PHARE snaps final_time to the step grid, and the error
# below compares against the initial condition, which only holds after exactly one period
time_step = final_time / np.ceil(final_time / 0.0007)
timestamps = [0.0, final_time]
diag_dir = "phare_outputs/convergence"

# At second order the midpoint flux quadrature and the face projections cap every
# reconstruction at 2, regardless of its 1D order. Fourth order lifts both, and only WENOZ and
# MP5 have a fourth-order branch. Limiters are only valid with Linear.
CASES = [
    # (reconstruction, limiter, mhd_order, expected slope)
    ("Linear", "VanLeer", 2, 2.0),
    ("WENO3", "None", 2, 2.0),
    ("WENOZ", "None", 2, 2.0),
    ("MP5", "None", 2, 2.0),
    ("WENOZ", "None", 4, 4.0),
    ("MP5", "None", 4, 4.0),
]

# SSPRK4_5 so the temporal error never dominates the spatial convergence.
mhd_timestepper = "SSPRK4_5"
ghosts = 4

# The gate is a floor: on these resolutions a reconstruction of higher 1D order than the scheme
# can still dominate the error and steepen the slope (pre-asymptotic), which is not a defect.
tolerance = 0.15

N_LIST = [16, 32, 64, 128]


def config(nx, reconstruction, limiter, mhd_order, diag_dir):
    sim = ph.Simulation(
        # smallest_patch_size=15,
        # largest_patch_size=25,
        time_step=time_step,
        final_time=final_time,
        cells=(2 * nx, nx, nx),
        dl=(3.0 / (2 * nx), 1.5 / nx, 1.5 / nx),
        refinement="tagging",
        max_mhd_level=1,
        max_nbr_levels=1,
        hyper_resistivity=0.0,
        resistivity=0.0,
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
        mhd_order=mhd_order,
        model_options=["MHDModel"],
    )

    eps = 1.0e-6
    background = np.asarray(
        [
            1.0,
            1.0 if mode == "Entropy" else 0.0,
            0.0,
            0.0,
            1.0,
            3.0 / 2.0,
            0.0,
            1.0 / sim.gamma,
        ]
    )
    Rfast_p = (
        1.0
        / (2.0 * np.sqrt(5.0))
        * np.asarray([2.0, 4.0, 2.0, 0.0, 0.0, 4.0, 0.0, 9.0])
    )
    Rfast_m = (
        1.0
        / (2.0 * np.sqrt(5.0))
        * np.asarray([2.0, -4.0, -2.0, 0.0, 0.0, 4.0, 0.0, 9.0])
    )
    Ralven_p = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    Ralven_m = np.asarray([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0])
    Rslow_p = (
        1.0
        / (2.0 * np.sqrt(5.0))
        * np.asarray([4.0, 2.0, 4.0, 0.0, 0.0, -2.0, 0.0, 3.0])
    )
    Rslow_m = (
        1.0
        / (2.0 * np.sqrt(5.0))
        * np.asarray([4.0, -2.0, -4.0, 0.0, 0.0, -2.0, 0.0, 3.0])
    )
    Rentrop = 1.0 / 2.0 * np.asarray([2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    R_mode = {
        "Alfven": Ralven_p,
        "Fast": Rfast_p,
        "Slow": Rslow_p,
        "Entropy": Rentrop,
    }

    R = R_mode[mode]

    sin_a = 2 / 3
    cos_a = np.sqrt(1 - sin_a**2)

    sin_b = 2 / np.sqrt(5)
    cos_b = np.sqrt(1 - sin_b**2)

    e1 = np.array([cos_a * cos_b, cos_a * sin_b, sin_a])
    e2 = np.array([-sin_b, cos_b, 0.0])
    e3 = np.array([-sin_a * cos_b, -sin_a * sin_b, cos_a])

    T = np.vstack([e1, e2, e3]).T

    def rotate(wave_frame):
        return T @ wave_frame

    v_bg = rotate(background[1:4])
    b_bg = rotate(background[4:7])

    Rv = rotate(np.array(R[1:4]))
    Rb = rotate(np.array(R[4:7]))

    e_bg = background[7] / (sim.gamma - 1) + 0.5 * (
        background[0] * np.dot(v_bg, v_bg) + np.dot(b_bg, b_bg)
    )

    def phase(x, y, z):
        return np.cos(2.0 * np.pi * x1(x, y, z))

    def x1(x, y, z):
        return x * cos_a * cos_b + y * cos_a * sin_b + z * sin_a

    def density(x, y, z):
        return background[0] + eps * R[0] * phase(x, y, z)

    def rhovx(x, y, z):
        return background[0] * v_bg[0] + eps * Rv[0] * phase(x, y, z)

    def rhovy(x, y, z):
        return background[0] * v_bg[1] + eps * Rv[1] * phase(x, y, z)

    def rhovz(x, y, z):
        return background[0] * v_bg[2] + eps * Rv[2] * phase(x, y, z)

    def vx(x, y, z):
        return rhovx(x, y, z) / density(x, y, z)

    def vy(x, y, z):
        return rhovy(x, y, z) / density(x, y, z)

    def vz(x, y, z):
        return rhovz(x, y, z) / density(x, y, z)

    def bx(x, y, z):
        return b_bg[0] + eps * Rb[0] * phase(x, y, z)

    def by(x, y, z):
        return b_bg[1] + eps * Rb[1] * phase(x, y, z)

    def bz(x, y, z):
        return b_bg[2] + eps * Rb[2] * phase(x, y, z)

    def E(x, y, z):
        return e_bg + eps * R[7] * phase(x, y, z)

    def p(x, y, z):
        return (
            E(x, y, z)
            - 0.5
            * (
                density(x, y, z)
                * (
                    vx(x, y, z) * vx(x, y, z)
                    + vy(x, y, z) * vy(x, y, z)
                    + vz(x, y, z) * vz(x, y, z)
                )
                + (
                    bx(x, y, z) * bx(x, y, z)
                    + by(x, y, z) * by(x, y, z)
                    + bz(x, y, z) * bz(x, y, z)
                )
            )
        ) * (sim.gamma - 1)

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    # dump at the run's resolved final_time, n * dt, which lands on the period up to round-off
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=[0.0, sim.time_stepper.final_time])

    return sim


# using by error is arbitrary now, add the error for everyone now
def compute_error(run, final_time):
    """Mean |By(final) - By(0)| over the interior cells: the wave returns to its initial state
    after one period, so the difference is the scheme's error."""
    from pyphare.pharesee.hierarchy.hierarchy_utils import single_patch_for_LO

    def by_at(time):
        return (
            single_patch_for_LO(run.GetB(time, all_primal=False).By)
            .levels()[0]
            .patches[0]
            .patch_datas["By"]
        )

    computed, expected = by_at(final_time), by_at(0.0)
    interior = tuple(slice(g, -g) if g else slice(None) for g in computed.ghosts_nbr)
    return np.mean(np.abs(computed.dataset[:][interior] - expected.dataset[:][interior]))


def run_convergence(reconstruction, limiter, mhd_order, expected):
    dx_values, errors, N_values = [], [], []

    # one directory per run, so a run that wrote nothing cannot be measured from another's dump
    profile_diag_dir = f"{diag_dir}_O{mhd_order}_{reconstruction}"

    for N_base in N_LIST:
        Nx, Ny, Nz = 2 * N_base, N_base, N_base
        Dx, Dy, Dz = 3.0 / Nx, 1.5 / Ny, 1.5 / Nz

        run_diag_dir = f"{profile_diag_dir}/N{N_base}"

        ph.global_vars.sim = None
        Simulator(config(N_base, reconstruction, limiter, mhd_order, run_diag_dir)).run().reset()

        run = Run(run_diag_dir)
        error = compute_error(run, final_time)

        dx_values.append(Dx)
        N_values.append(N_base)
        errors.append(error)

    dx_values = np.array(dx_values)
    slope, intercept = np.polyfit(np.log(dx_values), np.log(errors), 1)
    print(
        f"[multidimensional_convergence] MHD{mhd_order} {reconstruction}: N={N_values} "
        f"errors={errors} slope={slope:.3f}"
    )

    # every rank fitted the same slope: only rank 0 writes the plot
    if cpp.mpi_rank() == 0:
        fitted_line = np.exp(intercept) * dx_values**slope
        plt.figure(figsize=(10, 6))
        plt.loglog(dx_values, errors, "o-", label=f"Data (Slope: {slope:.2f})")
        plt.loglog(dx_values, fitted_line, "--", label="Fitted Line")
        plt.xlabel("Δx", fontsize=16)
        plt.ylabel("Error (L1 Norm)", fontsize=16)
        plt.title(f"{mode} - MHD{mhd_order} {reconstruction}", fontsize=20)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(fontsize=20)
        Path(profile_diag_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{profile_diag_dir}/convergence_O{mhd_order}_{reconstruction}.png", dpi=200)
        plt.close()

    floor = expected * (1 - tolerance)
    assert (
        slope >= floor
    ), f"MHD{mhd_order} {reconstruction}: got {slope}, expected >= {floor} (order {expected})"


def main():
    for reconstruction, limiter, mhd_order, expected in CASES:
        run_convergence(reconstruction, limiter, mhd_order, expected)


if __name__ == "__main__":
    main()
