#!/usr/bin/env python3
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import pyphare.pharein as ph
from pyphare import cpp
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator

from tests.simulator import SimulatorTest

# MPI: the analysis is rank-complete, so this runs at the 4 ranks CMakeLists.txt registers.
# Diagnostics land in one shared EM_B.h5 carrying every rank's patches (H5Writer's
# create_data_set_per_mpi collects all ranks' dataset paths into the same file), every
# process reads all of them back (fromh5 iterates a level's patch groups unfiltered), and
# single_patch_for_LO merges every level-0 patch into one whole-domain patch -- so the
# patches[0] used below is the assembled domain, never one rank's subdomain.
# It scans every (reconstruction, mhd_order) case in one run, each compiled as its own
# permutation in res/sim/all.txt.

os.environ["PHARE_SCOPE_TIMING"] = "1"

ph.NO_GUI()

mode_speed = {
    "Alfven": 1.0,
    "Fast": 2.0,
    "Slow": 0.5,
    "Entropy": 1.0,
}

mode = "Alfven"
time_step = 0.0007
final_time = 1.0 / mode_speed[mode]
timestamps = [0.0, final_time]
diag_dir = "phare_outputs/convergence"

# This sweep is the end-to-end statement of the scheme's spatial order on a smooth
# problem, in the dimension where nothing is degenerate. At second order the midpoint
# flux quadrature and the face projections cap every reconstruction at 2 regardless of
# its 1D order -- which is why all four O2 cases below expect 2.0, not the order of
# their own stencil. The fourth-order profile lifts the quadrature and the projections
# together, so its cases must reach 4: that is the claim this file exists to check.
#
# Only WENOZ and MP5 run at fourth order. They are the two reconstructions that take an
# O4 branch (ReconstructionSelector forwards Order == O4 to them alone), and pharein
# enforces the same restriction independently in check_mhd_profile.
#
# Limiters are only valid with Linear.
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

tolerance = 0.15

# Full sweep (default, CI): N=16,32,64,128, full one-period final_time, existing
# gate -- unchanged.
FULL_N_LIST = [16, 32, 64, 128]

# Opt-in local smoke check: measured cost is the reason for its shape. A real
# verifier run of just N=16, one reconstruction, at the FULL one-period
# final_time (final_time=1/mode_speed, ~1400 steps at time_step=0.0007) took
# 35m25s to reach only t=0.91 -- so even N=8/16/32 at full duration are still
# too costly locally, and N=64/128 are worse. The only way to make this cheap
# is to run a HANDFUL OF TIMESTEPS instead of the full period, at a couple of
# small N. That breaks compute_error's own assumption (comparing the
# final_time dump against t=0 assumes the wave has returned to its initial
# state after exactly one period) -- a shortened run has not done that, so a
# slope from it would not measure convergence order at all, just be a
# plausible-looking wrong number. So smoke mode does NOT fit or assert a
# slope; it only runs a few small resolutions for a few steps and checks the
# diagnostics it wrote are finite. It is not, and must never be read as, a
# convergence validation -- that is run_convergence()'s (CI's) job alone.
# Selection: env var PHARE_MHD3D_CONVERGENCE_SMOKE=1 or CLI flag --smoke.
# N list and step count are separately overridable (PHARE_MHD3D_CONVERGENCE_SMOKE_N,
# comma-separated; PHARE_MHD3D_CONVERGENCE_SMOKE_STEPS) since the safe/cheap
# point count and step count are a matter of the machine and patience, not of
# the code -- defaults are deliberately conservative (only the two cheapest,
# already-proven-valid resolutions from the full sweep, and a small step count)
# given no smoke run has actually been measured yet.
DEFAULT_SMOKE_N_LIST = [16, 32]
DEFAULT_SMOKE_STEPS = 5


def _smoke_selected():
    env_val = os.environ.get("PHARE_MHD3D_CONVERGENCE_SMOKE")
    cli_flag = "--smoke" in sys.argv
    if env_val is None:
        return cli_flag
    normalized = env_val.strip().lower()
    if normalized in ("1", "true", "yes"):
        return True
    if normalized in ("0", "false", "no", ""):
        return cli_flag
    raise ValueError(
        "PHARE_MHD3D_CONVERGENCE_SMOKE must be one of "
        "'1'/'true'/'yes' or '0'/'false'/'no' (case-insensitive), "
        f"got {env_val!r}"
    )


def _smoke_n_list():
    env_val = os.environ.get("PHARE_MHD3D_CONVERGENCE_SMOKE_N")
    if env_val is None:
        return DEFAULT_SMOKE_N_LIST
    try:
        n_list = [int(tok.strip()) for tok in env_val.split(",") if tok.strip()]
    except ValueError:
        raise ValueError(
            "PHARE_MHD3D_CONVERGENCE_SMOKE_N must be a comma-separated list of "
            f"positive integers, got {env_val!r}"
        )
    if not n_list or any(n <= 0 for n in n_list):
        raise ValueError(
            "PHARE_MHD3D_CONVERGENCE_SMOKE_N must list at least one positive "
            f"integer, got {env_val!r}"
        )
    return n_list


def _smoke_steps():
    env_val = os.environ.get("PHARE_MHD3D_CONVERGENCE_SMOKE_STEPS")
    if env_val is None:
        return DEFAULT_SMOKE_STEPS
    try:
        steps = int(env_val.strip())
    except ValueError:
        raise ValueError(
            "PHARE_MHD3D_CONVERGENCE_SMOKE_STEPS must be a positive integer, "
            f"got {env_val!r}"
        )
    if steps <= 0:
        raise ValueError(
            "PHARE_MHD3D_CONVERGENCE_SMOKE_STEPS must be a positive integer, "
            f"got {env_val!r}"
        )
    return steps


SMOKE = _smoke_selected()
SMOKE_N_LIST = _smoke_n_list() if SMOKE else None
SMOKE_STEPS = _smoke_steps() if SMOKE else None

# Module-import time: MPI is not yet initialized here (that happens later,
# inside the Simulator lifecycle), so this banner must not query cpp.mpi_rank()
# -- print unconditionally instead. Harmless duplicate output across the 4
# ranks; the later rank0 guard around PNG plotting (inside run_convergence,
# after a Simulator has actually run) is unaffected and stays rank0-only.
if SMOKE:
    print(
        f"[multidimensional_convergence] SMOKE MODE (local, opt-in): "
        f"N={SMOKE_N_LIST}, {SMOKE_STEPS} timestep(s) each, {len(CASES)} "
        "profile cases -- NOT a "
        "convergence validation (short integration, no slope fit/assert). "
        "Checks only that each resolution runs and writes finite "
        "diagnostics. The full sweep (default, no "
        "PHARE_MHD3D_CONVERGENCE_SMOKE/--smoke; N=16,32,64,128, full "
        "one-period integration) remains the sole convergence-order gate."
    )
else:
    print(
        f"[multidimensional_convergence] FULL SWEEP (default, CI): "
        f"N={FULL_N_LIST} for each of the {len(CASES)} profile cases "
        "(4 at second order, 2 at fourth), full one-period integration."
    )


def config(nx, reconstruction, limiter, mhd_order, diag_dir, time_step_nbr=None):
    # time_step_nbr is None (default): full one-period run, exactly as before.
    # time_step_nbr given (smoke only): run that many steps instead of the full
    # period -- check_time() (pharein/simulation.py) accepts exactly two of
    # {final_time, time_step, time_step_nbr}, so this swaps final_time out
    # rather than also passing it alongside time_step_nbr.
    time_kwargs = (
        dict(final_time=final_time)
        if time_step_nbr is None
        else dict(time_step_nbr=time_step_nbr)
    )
    sim = ph.Simulation(
        # smallest_patch_size=15,
        # largest_patch_size=25,
        time_step=time_step,
        **time_kwargs,
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

    # Full run: dump at the module-level [0.0, final_time] as before. Smoke
    # (time_step_nbr given): dump at the run's OWN resolved final_time -- the
    # module-level final_time (full period) would never be reached.
    dump_timestamps = timestamps if time_step_nbr is None else [0.0, sim.final_time]
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=dump_timestamps)

    return sim


# using by error is arbitrary now, add the error for everyone now
def compute_error(run, final_time, Nx, Dx, ghosts=0):
    coords = np.arange(Nx + 2 * ghosts) * Dx + 0.5 * Dx
    from pyphare.pharesee.hierarchy.hierarchy_utils import single_patch_for_LO

    computed_by = (
        single_patch_for_LO(run.GetB(final_time, all_primal=False).By)
        .levels()[0]
        .patches[0]
        .patch_datas["By"]
        .dataset[:]
    )

    expected_by = (
        single_patch_for_LO(run.GetB(0.0, all_primal=False).By)
        .levels()[0]
        .patches[0]
        .patch_datas["By"]
        .dataset[:]
    )

    # expected_by = 1e-6 * np.cos(2 * np.pi * (coords - final_time))
    return np.sum(np.abs(computed_by - expected_by)) / len(computed_by)


def run_convergence(reconstruction, limiter, mhd_order, expected):
    dx_values, errors, N_values = [], [], []

    # One directory per reconstruction and per resolution. Every run in this loop used to write
    # to the single shared diag_dir under "mode": "overwrite", so a resolution that produced no
    # dump would silently be measured against the previous resolution's file -- a plausible-looking
    # slope computed from the wrong data. Separate directories make that impossible.
    profile_diag_dir = f"{diag_dir}_O{mhd_order}_{reconstruction}"

    for N_base in FULL_N_LIST:
        Nx, Ny, Nz = 2 * N_base, N_base, N_base
        Dx, Dy, Dz = 3.0 / Nx, 1.5 / Ny, 1.5 / Nz

        run_diag_dir = f"{profile_diag_dir}/N{N_base}"

        ph.global_vars.sim = None
        started = time.time()
        Simulator(
            config(N_base, reconstruction, limiter, mhd_order, run_diag_dir)
        ).run().reset()

        # The error for this resolution must come from the run just above, never from an absent
        # or left-over dump: without output there is no measurement to make, and reporting one
        # anyway would turn a broken run into a convergence number.
        b_dump = Path(run_diag_dir) / "EM_B.h5"
        if not b_dump.exists():
            raise FileNotFoundError(
                f"{reconstruction} N={N_base}: the simulation wrote no diagnostics to {b_dump}"
            )
        # 1 s of slack absorbs filesystem timestamp granularity; a dump left by an earlier
        # invocation of this script is older than that by orders of magnitude.
        if b_dump.stat().st_mtime < started - 1.0:
            raise RuntimeError(
                f"{reconstruction} N={N_base}: {b_dump} predates this run -- refusing to measure"
                " convergence from a stale dump"
            )

        run = Run(run_diag_dir)
        error = compute_error(run, final_time, Nx, Dx)

        dx_values.append(Dx)
        N_values.append(N_base)
        errors.append(error)

    dx_values = np.array(dx_values)
    slope, intercept = np.polyfit(np.log(dx_values), np.log(errors), 1)

    # Every rank has read the whole domain and fitted the same slope, so the plot is one file
    # four processes would otherwise write at once. Only the writing is rank-guarded: the fit
    # above and the assert below stay on every rank.
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
        plt.savefig(
            f"{profile_diag_dir}/convergence_O{mhd_order}_{reconstruction}.png", dpi=200
        )
        plt.close()

    relative_error = abs(slope - expected) / abs(expected)
    assert (
        relative_error < tolerance
    ), f"MHD{mhd_order} {reconstruction}: got {slope}, expected {expected}"


def run_smoke_check(reconstruction, limiter, mhd_order):
    """Local opt-in smoke check: NOT a convergence measurement. See the
    SMOKE-mode comment block above for why -- short integration invalidates
    compute_error's one-period assumption, so no slope is fit or asserted
    here. This only confirms that each small resolution runs to completion
    and writes finite By diagnostics."""
    from pyphare.pharesee.hierarchy.hierarchy_utils import single_patch_for_LO

    profile_diag_dir = f"{diag_dir}_smoke_O{mhd_order}_{reconstruction}"

    for N_base in SMOKE_N_LIST:
        run_diag_dir = f"{profile_diag_dir}/N{N_base}"

        ph.global_vars.sim = None
        started = time.time()
        Simulator(
            config(
                N_base, reconstruction, limiter, mhd_order, run_diag_dir,
                time_step_nbr=SMOKE_STEPS,
            )
        ).run().reset()

        b_dump = Path(run_diag_dir) / "EM_B.h5"
        if not b_dump.exists():
            raise FileNotFoundError(
                f"[smoke] MHD{mhd_order} {reconstruction} N={N_base}: the simulation "
                f"wrote no diagnostics to {b_dump}"
            )
        if b_dump.stat().st_mtime < started - 1.0:
            raise RuntimeError(
                f"[smoke] MHD{mhd_order} {reconstruction} N={N_base}: {b_dump} predates"
                " this run -- refusing to check a stale dump"
            )

        run = Run(run_diag_dir)
        dump_time = run.times("B")[-1]
        by = (
            single_patch_for_LO(run.GetB(dump_time, all_primal=False).By)
            .levels()[0]
            .patches[0]
            .patch_datas["By"]
            .dataset[:]
        )
        if not np.all(np.isfinite(by)):
            raise AssertionError(
                f"[smoke] MHD{mhd_order} {reconstruction} N={N_base}: "
                f"non-finite By in diagnostics at t={dump_time}"
            )

        if cpp.mpi_rank() == 0:
            print(
                f"[multidimensional_convergence][smoke] MHD{mhd_order} "
                f"{reconstruction} N={N_base}: finite By diagnostics at "
                f"t={dump_time:.4g} (max|By|={np.max(np.abs(by)):.4g}) "
                f"after {SMOKE_STEPS} step(s) -- NOT a convergence slope."
            )


def main():
    for reconstruction, limiter, mhd_order, expected in CASES:
        if SMOKE:
            run_smoke_check(reconstruction, limiter, mhd_order)
        else:
            run_convergence(reconstruction, limiter, mhd_order, expected)


if __name__ == "__main__":
    main()
