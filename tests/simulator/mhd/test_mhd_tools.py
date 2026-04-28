import os

import numpy as np
import pyphare.pharein as ph
from pyphare.core import box as boxm
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator


def mhd_combination(
    mhd_timestepper,
    reconstruction,
    limiter,
    riemann,
    *,
    hall=True,
    res=False,
    hyper_res=True,
):
    return {
        "mhd_timestepper": mhd_timestepper,
        "reconstruction": reconstruction,
        "limiter": limiter,
        "riemann": riemann,
        "hall": hall,
        "res": res,
        "hyper_res": hyper_res,
    }


DEFAULT_COMBINATION = mhd_combination("SSPRK4_5", "WENOZ", "None", "HLL")
MHD_COMBINATIONS = (DEFAULT_COMBINATION,)


def combination_name(combination):
    return (
        f"{combination['mhd_timestepper']}_{combination['reconstruction']}_"
        f"{combination['limiter']}_{combination['riemann']}"
    )


def _domain_integral(hier, field_name, time=None):
    """Integrate field * cell_volume over domain, AMR-aware.

    Iterates level by level; for coarser levels only sums cells not covered
    by the next finer level (avoids double-counting in AMR).
    """
    total = 0.0
    finest = hier.finest_level(time)
    ratio = hier.refinement_ratio

    for ilvl in range(finest + 1):
        level = hier.level(ilvl, time)
        covered = (
            [boxm.coarsen(fp.box, ratio) for fp in hier.level(ilvl + 1, time).patches]
            if ilvl < finest
            else []
        )

        for patch in level.patches:
            pdata = patch.patch_datas[field_name]
            ng = pdata.ghosts_nbr
            cell_vol = np.prod(pdata.dl)

            uncovered = [patch.box]
            for cbox in covered:
                new_unc = []
                for ubox in uncovered:
                    new_unc.extend(ubox - cbox)
                uncovered = new_unc

            for ubox in uncovered:
                lo = ubox.lower - patch.box.lower
                hi = ubox.upper - patch.box.lower
                sl = tuple(
                    slice(int(ng[d] + lo[d]), int(ng[d] + hi[d] + 1))
                    for d in range(patch.box.ndim)
                )
                total += np.sum(pdata.dataset[sl]) * cell_vol

    return total


def check_mhd_conservation(
    test_case,
    *,
    case_name,
    config,
    combination,
    final_time,
    mass_rtol=1e-12,
    energy_rtol=1e-12,
    divB_atol=1e-10,
    check_divB=True,
):
    """Check mass, total energy, and divB=0 for a periodic MHD case.

    Plasma momentum is NOT checked: J×B exchanges momentum between plasma and EM field.
    Mass and total energy are conserved to machine precision for ideal/Hall MHD with
    periodic BCs (Hall and viscous terms are energy-conservative with divergence form).
    divB=0 is checked via finite-difference post-processing (not available in 1D).
    """
    ph.global_vars.sim = None
    combo_name = combination_name(combination)
    simulation = config(
        combination=combination,
        diag_dir=f"phare_outputs/simulator/mhd/{case_name}/conservation/{combo_name}",
    )

    unique_dir = f"{simulation.diag_options['options']['dir']}/{test_case.unique_diag_dir(simulation)}"
    os.makedirs(unique_dir, exist_ok=True)
    simulation.diag_options["options"]["dir"] = unique_dir
    test_case.register_diag_dir_for_cleanup(unique_dir)

    Simulator(simulation).run().reset()

    run = Run(unique_dir)
    t0 = 0.0

    mass_0 = _domain_integral(run.GetMHDrho(t0, all_primal=False), "mhdRho", t0)
    mass_f = _domain_integral(run.GetMHDrho(final_time, all_primal=False), "mhdRho", final_time)
    with test_case.subTest("mass"):
        rel_err = abs(mass_f - mass_0) / mass_0
        if rel_err >= mass_rtol:
            test_case.fail(
                f"{case_name}/{combo_name}: mass rel err {rel_err:.2e} >= {mass_rtol:.2e} "
                f"(∫ρ: {mass_0:.6e} → {mass_f:.6e})"
            )

    energy_0 = _domain_integral(run.GetMHDEtot(t0, all_primal=False), "mhdEtot", t0)
    energy_f = _domain_integral(run.GetMHDEtot(final_time, all_primal=False), "mhdEtot", final_time)
    with test_case.subTest("energy"):
        rel_err = abs(energy_f - energy_0) / energy_0
        if rel_err >= energy_rtol:
            test_case.fail(
                f"{case_name}/{combo_name}: energy rel err {rel_err:.2e} >= {energy_rtol:.2e} "
                f"(∫E: {energy_0:.6e} → {energy_f:.6e})"
            )

    if check_divB:
        divB_hier = run.GetDivB(final_time)
        max_divB = 0.0
        for ilvl in range(divB_hier.finest_level(final_time) + 1):
            for patch in divB_hier.level(ilvl, final_time).patches:
                pdata = patch.patch_datas["value"]
                ng = pdata.ghosts_nbr
                sl = tuple(slice(int(ng[d]), -int(ng[d])) for d in range(patch.box.ndim))
                max_divB = max(max_divB, float(np.max(np.abs(pdata.dataset[sl]))))
        with test_case.subTest("divB"):
            if max_divB >= divB_atol:
                test_case.fail(
                    f"{case_name}/{combo_name}: max|divB|={max_divB:.2e} >= {divB_atol:.2e}"
                )
