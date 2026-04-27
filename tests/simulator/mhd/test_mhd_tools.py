import os

import numpy as np
import pyphare.pharein as ph
from pyphare import cpp
from pyphare.pharesee.hierarchy import hierarchy_utils as hootils
from pyphare.pharesee.hierarchy.hierarchy_utils import flat_finest_field
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
    quantities=None,
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
    base_reference_root = combo_reference_root if combo_reference_root.exists() else reference_root
    mpi_n = cpp.mpi_size()
    mpi_specific = base_reference_root / f"mpi_{mpi_n}"
    case_reference_root = mpi_specific if mpi_specific.exists() else base_reference_root
    reference = Run(str(case_reference_root))
    candidate = Run(simulation.diag_options["options"]["dir"])

    if quantities is None:
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


def _flat_at_intersection(hier_a, hier_b, field_name):
    """Return field values from both hierarchies at their common physical coordinates.

    flat_finest_field keeps 1 ghost per patch side, so different proc counts yield
    different numbers of points at patch boundaries. Taking the intersection of
    coordinates discards those boundary-ghost duplicates and compares only the
    common (interior) points.
    """
    data_a, coords_a = flat_finest_field(hier_a, field_name)
    data_b, coords_b = flat_finest_field(hier_b, field_name)

    # Round to avoid floating-point noise in grid coordinate comparison.
    # Use dict keyed by coordinate to deduplicate: flat_finest_field keeps 1 ghost
    # per patch side, so the same physical coordinate can appear twice at patch
    # boundaries. Ghost copies have consistent values, so last-wins dedup is safe.
    decimals = 10
    if hier_a.ndim == 1:
        ca = np.round(coords_a, decimals)
        cb = np.round(coords_b, decimals)
        ca_dict = dict(zip(ca.tolist(), data_a))
        cb_dict = dict(zip(cb.tolist(), data_b))
    else:
        # 2D/3D: coords shape (N, ndim) — encode each point as a rounded tuple
        ca = np.round(coords_a, decimals)
        cb = np.round(coords_b, decimals)
        ca_dict = dict(zip(map(tuple, ca), data_a))
        cb_dict = dict(zip(map(tuple, cb), data_b))

    common = sorted(set(ca_dict.keys()) & set(cb_dict.keys()))
    vals_a = np.array([ca_dict[k] for k in common])
    vals_b = np.array([cb_dict[k] for k in common])

    return vals_a, vals_b


def compare_case_to_reference_flat(
    test_case,
    *,
    case_name,
    reference_root,
    config,
    combination,
    final_time,
    rtol=1e-14,
    atol=1e-16,
    quantities=None,
):
    """Proc-count-agnostic comparison via flat_finest_field.

    Uses single serial golden dataset; comparison works for any MPI count and all dimensions.
    """
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

    if quantities is None:
        quantities = {
            "B": lambda run: run.GetB(final_time, all_primal=False),
            "rho": lambda run: run.GetMHDrho(final_time, all_primal=False),
            "V": lambda run: run.GetMHDV(final_time, all_primal=False),
            "P": lambda run: run.GetMHDP(final_time, all_primal=False),
        }

    for quantity, getter in quantities.items():
        hier_cand = getter(candidate)
        hier_ref = getter(reference)
        for field_name in hier_cand.quantities():
            with test_case.subTest(quantity=quantity, field=field_name):
                vals_cand, vals_ref = _flat_at_intersection(hier_cand, hier_ref, field_name)
                try:
                    np.testing.assert_allclose(vals_cand, vals_ref, rtol=rtol, atol=atol)
                except AssertionError as e:
                    test_case.fail(f"{case_name}/{combo_name} {quantity}/{field_name}: {e}")
