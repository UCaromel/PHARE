#
# Shared infrastructure for CT reflux unit tests.
#
# Two-sim comparison: run AMR+reflux and flat (no refinement) with the same IC/dt/domain,
# compare coarse-level B. Any geometry bug surfaces as a per-cell mismatch.

import os
import numpy as np
from dataclasses import dataclass

import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator
from pyphare.pharesee.hierarchy import hierarchy_from


# ---------------------------------------------------------------------------
# Placement vs divB gating
# ---------------------------------------------------------------------------
# The production gate is discrete domain divB (assert_divB_zero /
# assert_domain_divB_zero) — it is what we ship on and runs unconditionally.
#
# The *placement* checks (assert_no_spurious / assert_correction_fired) compare
# B_amr vs B_flat outside the CF+covered band.
#
# UN-PARKED 2026-06-06: these previously appeared to expose a high-side reflux
# over-reach (corrections 2-5 cells past the high CF edge). That was a TEST
# ARTIFACT: GridLayout.nbrGhosts returned 2 (model default) while MHD/WENOZ field
# data carries 6 ghosts, so select/AMRToLocal windows were misaligned and ghost
# values leaked into the apparent "interior" comparison. With nbrGhosts=6
# (gridlayout.py session hack), REFLUX_STRICT=1 2D MHD goes fully green → the
# placement checks are real and now run BY DEFAULT.
#
# Disable with REFLUX_STRICT=0 to gate on divB only (e.g. while nbrGhosts is not 6).
_STRICT_NOTICE_EMITTED = False


def reflux_strict_placement():
    return os.environ.get("REFLUX_STRICT", "1").strip().lower() not in ("", "0", "false", "no")


def _skip_placement(check_name):
    """Return True (skip) when strict placement is off; emit a one-time notice."""
    if reflux_strict_placement():
        return False
    global _STRICT_NOTICE_EMITTED
    if not _STRICT_NOTICE_EMITTED:
        import sys
        print(
            "[reflux tests] placement checks (assert_no_spurious / "
            "assert_correction_fired) SKIPPED — divB-only gate. "
            "Set REFLUX_STRICT=1 to enable (known pre-existing high-side over-reach).",
            file=sys.stderr,
        )
        _STRICT_NOTICE_EMITTED = True
    return True


@dataclass
class RefluxIC:
    """B and v callables for one reflux test case. Each callable: *xyz -> float|ndarray."""
    bx: object
    by: object
    bz: object
    vx: object
    vy: object
    vz: object


def run_reflux_sim(solver, ndim, refinement_boxes, ic, cells, dl, dt, outdir, nbr_part_per_cell=100,
                   write_timestamps=None, smallest_patch_size=None, largest_patch_size=None):
    """Reset global state, build sim+model+diag, run one coarse step, return B hierarchy.

    smallest_patch_size: if None, auto-computed; pass an int to force the patch
    decomposition (e.g. smallest==largest==20 to place a patch seam at a chosen cell).
    """
    from pyphare.pharein.simulation import check_patch_size

    ph.global_vars.sim = None

    if largest_patch_size is None:
        # cells//2 generalizes the prior hardcoded 20 (2D CELLS=40 → 20, identical).
        largest_patch_size = cells // 2

    if smallest_patch_size is None:
        # nbrGhosts=6 hack (gridlayout.py): check_patch_size sets small_invalid=6 and the
        # auto min_per_interp[0]=6, so the default would raise on 6<=6. Force >=7 (localized
        # harness relax). cells>=20 > 7 OK; all refinement boxes are >=5 coarse cells.
        _, smallest_patch_size = check_patch_size(
            ndim, interp_order=1, cells=[cells] * ndim, smallest_patch_size=7
        )

    os.makedirs(outdir, exist_ok=True)

    common = dict(
        interp_order=1,
        smallest_patch_size=smallest_patch_size,
        largest_patch_size=largest_patch_size,
        time_step_nbr=1,
        time_step=dt,
        boundary_types=["periodic"] * ndim,
        cells=[cells] * ndim,
        dl=[dl] * ndim,
        refinement_boxes=refinement_boxes,
        diag_options={"format": "phareh5", "options": {"dir": outdir, "mode": "overwrite"}},
        strict=True,
        # 0: hand-controlled periodic boxes are valid flush-to-edge; check_refinement_boxes
        # is not periodic-aware and would spuriously reject them with a nonzero buffer.
        nesting_buffer=0,
    )

    if solver == "MHD":
        max_nbr_levels = len(refinement_boxes) + 1
        sim = ph.Simulation(
            **common,
            model_options=["MHDModel"],
            max_mhd_level=max_nbr_levels,
            reconstruction="WENOZ",
            limiter="None",
            riemann="Rusanov",
            mhd_timestepper="TVDRK3",
            gamma=5.0 / 3.0,
        )

        def rho(*xyz):
            return np.ones_like(xyz[0]) + 0.0 * xyz[0]

        def p(*xyz):
            return np.ones_like(xyz[0]) + 0.0 * xyz[0]

        ph.MHDModel(density=rho, vx=ic.vx, vy=ic.vy, vz=ic.vz, bx=ic.bx, by=ic.by, bz=ic.bz, p=p)

    elif solver == "Hybrid":
        sim = ph.Simulation(**common)

        def rho(*xyz):
            return np.ones_like(xyz[0]) + 0.0 * xyz[0]

        def vth(*xyz):
            return np.full_like(xyz[0], 0.01)

        protons = {
            "charge": 1,
            "density": rho,
            "vbulkx": ic.vx,
            "vbulky": ic.vy,
            "vbulkz": ic.vz,
            "vthx": vth,
            "vthy": vth,
            "vthz": vth,
            "nbr_part_per_cell": nbr_part_per_cell,
            "init": {"seed": 1337},
        }
        ph.MaxwellianFluidModel(bx=ic.bx, by=ic.by, bz=ic.bz, protons=protons)
        ph.ElectronModel(closure="isothermal", Te=0.12)

    else:
        raise ValueError(f"Unknown solver: {solver!r}. Use 'MHD' or 'Hybrid'.")

    if write_timestamps is None:
        write_timestamps = np.array([dt])
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=np.asarray(write_timestamps))

    Simulator(sim).run()

    return hierarchy_from(h5_filename=outdir + "/EM_B.h5")


def run_reflux_pair(solver, ndim, refinement_boxes, ic, cells, dl, dt, label):
    """Run AMR+reflux and flat reference. Returns (hier_amr, hier_flat)."""
    import tempfile

    base = os.path.join(tempfile.gettempdir(), f"phare_reflux_{label}")
    hier_amr = run_reflux_sim(solver, ndim, refinement_boxes, ic, cells, dl, dt, base + "_amr")
    hier_flat = run_reflux_sim(solver, ndim, {}, ic, cells, dl, dt, base + "_flat")
    return hier_amr, hier_flat


def coarse_field_array(hier, component):
    """Assemble ghost-stripped global array for `component` at coarse level 0."""
    level = hier.level(0)
    patches = level.patches
    if not patches:
        raise ValueError("No patches at level 0")

    pd0 = patches[0].patch_datas[component]
    ndim = pd0.ndim
    centerings = pd0.centerings

    # Infer domain extent from union of all patch boxes
    upper_cells = np.zeros(ndim, dtype=int)
    for patch in patches:
        upper_cells = np.maximum(upper_cells, np.array(patch.box.upper))
    domain_cells = upper_cells + 1  # number of cells per direction

    shape = []
    for i, c in enumerate(centerings):
        shape.append(int(domain_cells[i]) + (1 if c == "primal" else 0))

    global_arr = np.zeros(shape)

    for patch in patches:
        pd = patch.patch_datas[component]
        interior = pd.select(patch.box)
        slices = []
        for i, c in enumerate(centerings):
            lo = int(patch.box.lower[i])
            n = int(patch.box.shape[i]) + (1 if c == "primal" else 0)
            slices.append(slice(lo, lo + n))
        global_arr[tuple(slices)] = interior

    return global_arr


# ---------------------------------------------------------------------------
# Cell-set helpers
# ---------------------------------------------------------------------------

def _get_boxes(refinement_boxes, level):
    """Extract list of Box objects from refinement_boxes at `level` (e.g. 'L0')."""
    val = refinement_boxes.get(level, None)
    if val is None and isinstance(level, str) and level.startswith("L"):
        val = refinement_boxes.get(int(level[1:]), None)
    if val is None:
        return []
    if isinstance(val, dict):
        from pyphare.core.box import Box
        boxes = []
        for coords in val.values():
            if hasattr(coords, "lower"):
                boxes.append(coords)
            else:
                lo, hi = coords
                boxes.append(Box(lo, hi))
        return boxes
    if isinstance(val, list):
        return val
    return [val]


def _expand_periodic(lower, upper, cells, grow):
    """Return set of cell tuples in grown box [lower-grow, upper+grow], periodic."""
    ndim = len(lower)
    ranges = []
    for i in range(ndim):
        lo = int(lower[i]) - grow
        hi = int(upper[i]) + grow
        ranges.append([j % cells for j in range(lo, hi + 1)])

    result = set()
    if ndim == 1:
        for i in ranges[0]:
            result.add(i)
    elif ndim == 2:
        for i in ranges[0]:
            for j in ranges[1]:
                result.add((i, j))
    elif ndim == 3:
        for i in ranges[0]:
            for j in ranges[1]:
                for k in ranges[2]:
                    result.add((i, j, k))
    return result


def covered_cell_set(refinement_boxes, domain_cells, ndim, level="L0", ratio=2):
    """Cell-index tuples (coarse) covered by fine patches.
    refinement_boxes are in coarse index space (PHARE convention).
    """
    covered = set()
    for box in _get_boxes(refinement_boxes, level):
        lower = np.array(box.lower)
        upper = np.array(box.upper)
        covered |= _expand_periodic(lower, upper, domain_cells, 0)
    return covered


def cf_cell_set(refinement_boxes, domain_cells, ndim, level="L0", ratio=2):
    """1-cell shell around covered cells (cells with at least one CF boundary face), periodic.
    refinement_boxes are in coarse index space (PHARE convention).
    """
    cf = set()
    for box in _get_boxes(refinement_boxes, level):
        lower = np.array(box.lower)
        upper = np.array(box.upper)
        shell = _expand_periodic(lower, upper, domain_cells, 1)
        interior = _expand_periodic(lower, upper, domain_cells, 0)
        cf |= shell - interior
    return cf


# ---------------------------------------------------------------------------
# Suspect mask helpers (face-level, for face-centered components)
# ---------------------------------------------------------------------------

def _suspect_mask(suspect_cells, centerings, domain_cells, ndim):
    """Build boolean mask over the global field array.

    For dual directions, a cell is suspect if it's in suspect_cells.
    For primal directions, a face is suspect if either adjacent cell is suspect.
    Handles periodic wrap: face 0 also borrows from cell domain_cells-1.
    """
    shape = [domain_cells + (1 if c == "primal" else 0) for c in centerings]
    mask = np.zeros(shape, dtype=bool)

    # Build cell-indexed suspect mask  shape=(domain_cells,)*ndim
    cell_mask = np.zeros([domain_cells] * ndim, dtype=bool)
    for idx in suspect_cells:
        if ndim == 1:
            cell_mask[int(idx) % domain_cells] = True
        elif ndim == 2:
            cell_mask[idx[0] % domain_cells, idx[1] % domain_cells] = True
        elif ndim == 3:
            cell_mask[idx[0] % domain_cells, idx[1] % domain_cells, idx[2] % domain_cells] = True

    def _expand_axis(m_out, m_cell, ax):
        """Expand cell mask onto face mask along `ax` (primal direction).

        Face ix on axis ax is suspect if cell ix or cell ix-1 is suspect.
        We address via slices rather than loops for arbitrary ndim.
        """
        n = m_cell.shape[ax]
        # Build index tuples for slicing along `ax`
        def _sl(start, stop):
            s = [slice(None)] * m_out.ndim
            s[ax] = slice(start, stop)
            return tuple(s)

        def _csl(start, stop):
            s = [slice(None)] * m_cell.ndim
            s[ax] = slice(start, stop)
            return tuple(s)

        # face ix from cell ix (right neighbor): face 0..n-1 from cell 0..n-1
        m_out[_sl(0, n)] |= m_cell[_csl(0, n)]
        # face ix from cell ix-1 (left neighbor): face 1..n from cell 0..n-1
        m_out[_sl(1, n + 1)] |= m_cell[_csl(0, n)]
        # periodic wrap: face 0 from cell n-1, face n from cell 0
        m_out[_sl(0, 1)] |= m_cell[_csl(n - 1, n)]
        m_out[_sl(n, n + 1)] |= m_cell[_csl(0, 1)]

    for ax, c in enumerate(centerings):
        if c == "primal":
            _expand_axis(mask, cell_mask, ax)

    # Fill dual directions directly from cell mask, but only if nothing was written yet
    # by a primal expansion. For a pure-dual component (e.g. 2D Bz), fill entire mask.
    if all(c == "dual" for c in centerings):
        mask[:] = cell_mask
    elif any(c == "dual" for c in centerings):
        # For mixed centering: dual axes need cell mask smeared through primal expansions.
        # The primal _expand_axis calls already cover all faces in the primal direction.
        # For cells that are purely dual, we OR in the cell mask directly.
        # (This is handled implicitly since dual axes have shape==domain_cells, and
        # the primal expansion writes into full slices along dual axes.)
        pass

    return mask


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_no_spurious(hier_amr, hier_flat, cf_cells, covered_cells, components=("Bx", "By", "Bz"), atol=1e-13):
    """Assert B_AMR == B_flat at every coarse face/cell not in cf or covered region."""
    if _skip_placement("assert_no_spurious"):
        return
    suspect = cf_cells | covered_cells

    for comp in components:
        arr_amr = coarse_field_array(hier_amr, comp)
        arr_flat = coarse_field_array(hier_flat, comp)

        pd0 = hier_amr.level(0).patches[0].patch_datas[comp]
        centerings = pd0.centerings
        ndim = pd0.ndim

        # domain_cells from array shape: dual dim → shape[i], primal dim → shape[i]-1
        domain_cells_arr = [arr_amr.shape[i] - (1 if c == "primal" else 0) for i, c in enumerate(centerings)]
        domain_cells = domain_cells_arr[0]  # uniform

        mask = _suspect_mask(suspect, centerings, domain_cells, ndim)
        diff = np.abs(arr_amr - arr_flat)
        bad = diff[~mask]
        if np.any(bad > atol):
            idx = np.argmax(diff * (~mask))
            raise AssertionError(
                f"{comp}: spurious correction outside CF/covered region. "
                f"Max diff = {bad.max():.3e} (atol={atol:.1e})"
            )


def assert_correction_fired(hier_amr, hier_flat, cf_cells, components=("Bx", "By", "Bz"), min_delta=1e-10):
    """Assert at least one CF cell/face has |B_AMR - B_flat| > min_delta."""
    if _skip_placement("assert_correction_fired"):
        return
    found_any = False

    for comp in components:
        arr_amr = coarse_field_array(hier_amr, comp)
        arr_flat = coarse_field_array(hier_flat, comp)

        pd0 = hier_amr.level(0).patches[0].patch_datas[comp]
        centerings = pd0.centerings
        ndim = pd0.ndim
        domain_cells = arr_amr.shape[0] - (1 if centerings[0] == "primal" else 0)

        mask = _suspect_mask(cf_cells, centerings, domain_cells, ndim)
        diff = np.abs(arr_amr - arr_flat)
        if np.any(diff[mask] > min_delta):
            found_any = True
            break

    if not found_any:
        max_diff = max(
            np.abs(coarse_field_array(hier_amr, c) - coarse_field_array(hier_flat, c)).max()
            for c in components
        )
        raise AssertionError(
            f"No reflux correction detected in CF region for components {components}. "
            f"Max diff anywhere = {max_diff:.3e} (min_delta={min_delta:.1e}). "
            f"IC may be too trivial or reflux not triggered."
        )


def assert_domain_divB_zero(B_hier, atol=1e-12):
    """Assert discrete divB ~ 0 in every DOMAIN cell of every level.

    Uses the validated production divB (`_divB2D`) on the full ghost-inclusive
    Bx/By arrays, then strips to the domain via `ghost_box` vs `box` (data-driven,
    independent of GridLayout.nbrGhosts / select). Ghost cells are intentionally
    excluded — the coarse-fine ghost seam is a separate, known concern; the domain
    is the invariant we gate on.

    ndim-dispatch: the 2D branch is byte-identical to the validated original; the 3D
    branch uses the same per-patch + ghost-strip + all-levels mechanism with the
    3-axis divergence formula proven in `assert_divB_zero`. Calibrate before trusting
    (`test_divB_gate_calibration`).
    """
    first_lvl = sorted(B_hier.levels().keys())[0]
    ndim = B_hier.level(first_lvl).patches[0].patch_datas["Bx"].ndim

    if ndim == 2:
        from pyphare.pharesee.run.utils import _divB2D

        for ilvl in sorted(B_hier.levels().keys()):
            worst, worst_loc = 0.0, None
            for p in B_hier.level(ilvl).patches:
                bx, by = p.patch_datas["Bx"], p.patch_datas["By"]
                divb = np.abs(_divB2D(bx.dataset[:], by.dataset[:], bx.x, by.y))  # (nx+2g, ny+2g) dual
                # divB dual-cell (i,j) is indexed from Bx.ghost_box.lower; strip to domain box
                off = np.array(p.box.lower) - np.array(bx.ghost_box.lower)
                nx, ny = int(p.box.shape[0]), int(p.box.shape[1])
                dom = divb[off[0]:off[0] + nx, off[1]:off[1] + ny]
                if dom.size and dom.max() > worst:
                    worst = dom.max()
                    worst_loc = (ilvl, tuple(int(p.box.lower[k]) for k in range(2)))
            assert worst < atol, (
                f"domain divB not zero at level {ilvl}: max |divB| = {worst:.3e} "
                f"(atol={atol:.1e}), patch {worst_loc}"
            )

    elif ndim == 3:
        for ilvl in sorted(B_hier.levels().keys()):
            worst, worst_loc = 0.0, None
            for p in B_hier.level(ilvl).patches:
                bx, by, bz = p.patch_datas["Bx"], p.patch_datas["By"], p.patch_datas["Bz"]
                divb = np.abs(
                    np.diff(bx.dataset[:], axis=0) / bx.dl[0]
                    + np.diff(by.dataset[:], axis=1) / by.dl[1]
                    + np.diff(bz.dataset[:], axis=2) / bz.dl[2]
                )  # (nxd, nyd, nzd), indexed from Bx.ghost_box.lower
                off = np.array(p.box.lower) - np.array(bx.ghost_box.lower)
                nx, ny, nz = (int(p.box.shape[k]) for k in range(3))
                dom = divb[off[0]:off[0] + nx, off[1]:off[1] + ny, off[2]:off[2] + nz]
                if dom.size and dom.max() > worst:
                    worst = dom.max()
                    worst_loc = (ilvl, tuple(int(p.box.lower[k]) for k in range(3)))
            assert worst < atol, (
                f"domain divB not zero at level {ilvl}: max |divB| = {worst:.3e} "
                f"(atol={atol:.1e}), patch {worst_loc}"
            )

    else:
        raise ValueError(f"assert_domain_divB_zero: unsupported ndim={ndim}")


def assert_shared_process_seam(B_hier, base_cells, ndim=2):
    """Guard: assert a patch (process) boundary exists at the domain center on EVERY
    level — i.e. the pathological triple-coincident seam is actually realized, not
    silently collapsed by the box generator into a single patch."""
    for ilvl in sorted(B_hier.levels().keys()):
        center = (base_cells // 2) * (2 ** ilvl)
        lowers = {tuple(int(p.box.lower[k]) for k in range(ndim)) for p in B_hier.level(ilvl).patches}
        assert (center,) * ndim in lowers, (
            f"level {ilvl}: no patch with lower-corner at center {(center,) * ndim} "
            f"(patch lowers: {sorted(lowers)}) — shared process seam not formed; "
            f"check forced patch size"
        )


def assert_divB_zero(hier, level=0, atol=1e-13):
    """Discrete divB = 0 at every interior cell of the given level."""
    lvl = hier.level(level)
    patches = lvl.patches
    if not patches:
        return

    pd0 = patches[0].patch_datas.get("Bx", None)
    if pd0 is None:
        return
    ndim = pd0.ndim

    if ndim == 1:
        # divB = (Bx[i+1] - Bx[i]) / dx
        bx = coarse_field_array(hier, "Bx") if level == 0 else _level_field_array(hier, level, "Bx")
        dl = pd0.dl[0]
        divb = np.diff(bx) / dl
        max_div = np.abs(divb).max()
        assert max_div < atol, f"divB not zero at level {level}: max |divB| = {max_div:.3e}"

    elif ndim == 2:
        bx_arr = coarse_field_array(hier, "Bx") if level == 0 else _level_field_array(hier, level, "Bx")
        by_arr = coarse_field_array(hier, "By") if level == 0 else _level_field_array(hier, level, "By")
        dx = pd0.dl[0]
        dy_pd = patches[0].patch_datas["By"].dl[1]
        # divB[i,j] = (Bx[i+1,j] - Bx[i,j])/dx + (By[i,j+1] - By[i,j])/dy
        divb = np.diff(bx_arr, axis=0) / dx + np.diff(by_arr, axis=1) / dy_pd
        max_div = np.abs(divb).max()
        assert max_div < atol, f"divB not zero at level {level}: max |divB| = {max_div:.3e}"

    elif ndim == 3:
        bx_arr = coarse_field_array(hier, "Bx") if level == 0 else _level_field_array(hier, level, "Bx")
        by_arr = coarse_field_array(hier, "By") if level == 0 else _level_field_array(hier, level, "By")
        bz_arr = coarse_field_array(hier, "Bz") if level == 0 else _level_field_array(hier, level, "Bz")
        dx = pd0.dl[0]
        dy_pd = patches[0].patch_datas["By"].dl[1]
        dz_pd = patches[0].patch_datas["Bz"].dl[2]
        divb = (
            np.diff(bx_arr, axis=0) / dx
            + np.diff(by_arr, axis=1) / dy_pd
            + np.diff(bz_arr, axis=2) / dz_pd
        )
        max_div = np.abs(divb).max()
        assert max_div < atol, f"divB not zero at level {level}: max |divB| = {max_div:.3e}"


def _level_field_array(hier, level, component):
    """Assemble ghost-stripped global array for `component` at arbitrary level."""
    lvl = hier.level(level)
    patches = lvl.patches
    if not patches:
        raise ValueError(f"No patches at level {level}")

    pd0 = patches[0].patch_datas[component]
    ndim = pd0.ndim
    centerings = pd0.centerings

    upper_cells = np.zeros(ndim, dtype=int)
    for patch in patches:
        upper_cells = np.maximum(upper_cells, np.array(patch.box.upper))
    domain_cells = upper_cells + 1

    shape = [int(domain_cells[i]) + (1 if centerings[i] == "primal" else 0) for i in range(ndim)]
    global_arr = np.zeros(shape)

    for patch in patches:
        pd = patch.patch_datas[component]
        interior = pd.select(patch.box)
        slices = []
        for i, c in enumerate(centerings):
            lo = int(patch.box.lower[i])
            n = int(patch.box.shape[i]) + (1 if c == "primal" else 0)
            slices.append(slice(lo, lo + n))
        global_arr[tuple(slices)] = interior

    return global_arr


# ---------------------------------------------------------------------------
# IC factories
# ---------------------------------------------------------------------------

def ct_ic_2d(Lx, Ly, B0=0.1, v0=0.05):
    """Bx=B0*sin(2πy/Ly), By=B0*sin(2πx/Lx), Bz=0. divB=0, non-trivial Ez at both CF directions."""

    def bx(*xyz):
        return B0 * np.sin(2 * np.pi * xyz[1] / Ly)

    def by(*xyz):
        return B0 * np.sin(2 * np.pi * xyz[0] / Lx)

    def bz(*xyz):
        return np.zeros_like(xyz[0]) + 0.0

    def vx(*xyz):
        return np.zeros_like(xyz[0]) + v0

    def vy(*xyz):
        return np.zeros_like(xyz[0]) + v0

    def vz(*xyz):
        return np.zeros_like(xyz[0]) + 0.0

    return RefluxIC(bx=bx, by=by, bz=bz, vx=vx, vy=vy, vz=vz)


def hydro_ic_2d(Lx, Ly, B0=0.1, v0=0.05):
    """Bx=By=0, Bz=B0*sin(2πx/Lx)*sin(2πy/Ly), vx=vy=v0. Hydro-like (Bz cell-centered) path."""

    def bx(*xyz):
        return np.zeros_like(xyz[0]) + 0.0

    def by(*xyz):
        return np.zeros_like(xyz[0]) + 0.0

    def bz(*xyz):
        return B0 * np.sin(2 * np.pi * xyz[0] / Lx) * np.sin(2 * np.pi * xyz[1] / Ly)

    def vx(*xyz):
        return np.zeros_like(xyz[0]) + v0

    def vy(*xyz):
        return np.zeros_like(xyz[0]) + v0

    def vz(*xyz):
        return np.zeros_like(xyz[0]) + 0.0

    return RefluxIC(bx=bx, by=by, bz=bz, vx=vx, vy=vy, vz=vz)


def hydro_ic_1d_by(Lx, B0=0.1, v0=0.05):
    """Bx=B0 (const), By=B0*sin(2πx/Lx), Bz=0, vx=v0. By advected by Ez."""

    def bx(*xyz):
        return np.zeros_like(xyz[0]) + B0

    def by(*xyz):
        return B0 * np.sin(2 * np.pi * xyz[0] / Lx)

    def bz(*xyz):
        return np.zeros_like(xyz[0]) + 0.0

    def vx(*xyz):
        return np.zeros_like(xyz[0]) + v0

    def vy(*xyz):
        return np.zeros_like(xyz[0]) + 0.0

    def vz(*xyz):
        return np.zeros_like(xyz[0]) + 0.0

    return RefluxIC(bx=bx, by=by, bz=bz, vx=vx, vy=vy, vz=vz)


def hydro_ic_1d_bz(Lx, B0=0.1, v0=0.05):
    """Bx=B0 (const), By=0, Bz=B0*sin(2πx/Lx), vx=v0. Bz advected by Ey."""

    def bx(*xyz):
        return np.zeros_like(xyz[0]) + B0

    def by(*xyz):
        return np.zeros_like(xyz[0]) + 0.0

    def bz(*xyz):
        return B0 * np.sin(2 * np.pi * xyz[0] / Lx)

    def vx(*xyz):
        return np.zeros_like(xyz[0]) + v0

    def vy(*xyz):
        return np.zeros_like(xyz[0]) + 0.0

    def vz(*xyz):
        return np.zeros_like(xyz[0]) + 0.0

    return RefluxIC(bx=bx, by=by, bz=bz, vx=vx, vy=vy, vz=vz)


def ct_ic_3d(Lx, Ly, Lz, B0=0.1, v0=0.05):
    """3D CT IC: divB=0 by construction, non-trivial E on all 3 face orientations."""

    def bx(*xyz):
        return B0 * np.sin(2 * np.pi * xyz[1] / Ly) * np.sin(2 * np.pi * xyz[2] / Lz)

    def by(*xyz):
        return B0 * np.sin(2 * np.pi * xyz[0] / Lx) * np.sin(2 * np.pi * xyz[2] / Lz)

    def bz(*xyz):
        return B0 * np.sin(2 * np.pi * xyz[0] / Lx) * np.sin(2 * np.pi * xyz[1] / Ly)

    def vx(*xyz):
        return np.zeros_like(xyz[0]) + v0

    def vy(*xyz):
        return np.zeros_like(xyz[0]) + v0

    def vz(*xyz):
        return np.zeros_like(xyz[0]) + v0

    return RefluxIC(bx=bx, by=by, bz=bz, vx=vx, vy=vy, vz=vz)
