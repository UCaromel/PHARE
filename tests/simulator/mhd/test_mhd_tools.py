import os

import pyphare.pharein as ph
from pyphare.pharesee.hierarchy import hierarchy_utils as hootils
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


def compare_case_to_reference(
    test_case,
    *,
    case_name,
    reference_root,
    config,
    combination,
    final_time,
    rtol=1e-14,
    atol=1e-16,
):
    ph.global_vars.sim = None
    combo_name = combination_name(combination)
    simulation = config(
        combination=combination,
        diag_dir=f"phare_outputs/simulator/mhd/{case_name}/{combo_name}",
    )

    unique_dir = f"{simulation.diag_options['options']['dir']}/{test_case.unique_diag_dir(simulation)}"
    os.makedirs(unique_dir, exist_ok=True)
    simulation.diag_options["options"]["dir"] = unique_dir
    test_case.register_diag_dir_for_cleanup(unique_dir)

    Simulator(simulation).run().reset()

    combo_reference_root = reference_root / combo_name
    case_reference_root = combo_reference_root if combo_reference_root.exists() else reference_root
    reference = Run(str(case_reference_root))
    candidate = Run(simulation.diag_options["options"]["dir"])

    quantities = {
        "B": lambda run: run.GetB(final_time),
        "rho": lambda run: run.GetMHDrho(final_time),
        "V": lambda run: run.GetMHDV(final_time),
        "P": lambda run: run.GetMHDP(final_time),
    }

    for quantity, getter in quantities.items():
        with test_case.subTest(quantity=quantity):
            eqr = hootils.hierarchy_compare(
                getter(candidate), getter(reference), atol=atol, rtol=rtol
            )
            test_case.assertTrue(bool(eqr), f"{case_name}/{combo_name} {quantity}: {eqr}")
