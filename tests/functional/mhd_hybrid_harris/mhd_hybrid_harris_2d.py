#!/usr/bin/env python3
"""
Smoke test: 2D MHD-Hybrid coupling, Harris current sheet.

MHD on level 0, Hybrid (PIC) on level 1.
Verifies initialization + a handful of timesteps run without crash or NaN.
"""

import numpy as np
from pathlib import Path

from pyphare import cpp
import pyphare.pharein as ph
from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator, startMPI

from tests.simulator import SimulatorTest

ph.NO_GUI()

cells = (96, 48)
time_step = 0.005
final_time = 0.15
timestamps = [0, final_time]
# B is dumped every coarse step so the first L1 regrid can be bracketed (Step 1).
B_TIMESTAMPS = [i * time_step for i in range(int(final_time / time_step) + 1)]
# divB gate tolerance (Step 6). Baseline mode: reported, NOT asserted.
# Diagnostics are dumped float32: the post-hoc divB floor is ~B*eps32/dx ~ 6e-7
# (measured 5.9e-7 on L1), so the gate sits ~10x above the dump-precision floor
# and ~5 orders below the two-step-regrid violation (1.7e-1).
DIVB_TOL = 5e-6

# island perturbation amplitude; 0 => discretely divergence-free init,
# required for the absolute DIVB_TOL gate (point-sampled perturbation
# carries an O(dx^2 * dB) divB floor ~1.4e-3 that CT preserves)
DB_PERT = 0.0
diag_dir = "phare_outputs/mhd_hybrid_harris_2d"

hall = True
res = False
hyper_res = True

def config():
    L = 0.5

    sim = ph.Simulation(
        time_step=time_step,
        final_time=final_time,
        cells=cells,
        dl=(0.40, 0.40),
        refinement="tagging",
        max_mhd_level=1,
        max_nbr_levels=2,
        # break the single giant L1 patch into several so tag fluctuations
        # produce layout-changing regrids within a few coarse steps
        largest_patch_size=(24, 24),
        interp_order=2,
        hyper_resistivity=0.002,
        resistivity=0.0,
        diag_options={
            "format": "phareh5",
            "options": {"dir": diag_dir, "mode": "overwrite"},
        },
        strict=True,
        nesting_buffer=1,
        hyper_mode="spatial",
        eta=0.0,
        nu=0.02,
        gamma=5.0 / 3.0,
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="TVDRK3",
        hall=hall,
        res=res,
        hyper_res=hyper_res,
        model_options=["MHDModel", "HybridModel"],
    )

    def S(y, y0, l):
        return 0.5 * (1.0 + np.tanh((y - y0) / l))

    def density(x, y):
        Ly = sim.simulation_domain()[1]
        return (
            0.4
            + 1.0 / np.cosh((y - Ly * 0.3) / L) ** 2
            + 1.0 / np.cosh((y - Ly * 0.7) / L) ** 2
        )

    def vx(x, y):
        return 0.0

    def vy(x, y):
        return 0.0

    def vz(x, y):
        return 0.0

    def bx(x, y):
        Lx = sim.simulation_domain()[0]
        Ly = sim.simulation_domain()[1]
        sigma = 1.0
        dB = DB_PERT

        x0 = x - 0.5 * Lx
        y1 = y - 0.3 * Ly
        y2 = y - 0.7 * Ly

        dBx1 = -2 * dB * y1 * np.exp(-(x0**2 + y1**2) / sigma**2)
        dBx2 = 2 * dB * y2 * np.exp(-(x0**2 + y2**2) / sigma**2)

        v1 = -1.0
        v2 = 1.0
        return v1 + (v2 - v1) * (S(y, Ly * 0.3, L) - S(y, Ly * 0.7, L)) + dBx1 + dBx2

    def by(x, y):
        Lx = sim.simulation_domain()[0]
        Ly = sim.simulation_domain()[1]
        sigma = 1.0
        dB = DB_PERT

        x0 = x - 0.5 * Lx
        y1 = y - 0.3 * Ly
        y2 = y - 0.7 * Ly

        dBy1 = 2 * dB * x0 * np.exp(-(x0**2 + y1**2) / sigma**2)
        dBy2 = -2 * dB * x0 * np.exp(-(x0**2 + y2**2) / sigma**2)

        return dBy1 + dBy2

    def bz(x, y):
        return 0.0

    def p(x, y):
        return 1.0 - (bx(x, y) ** 2 + by(x, y) ** 2) / 2.0

    # MHDModel carries the proton population spec for the Hybrid level.
    # Particle density/velocities/temperatures are placeholders here;
    # actual injection uses MHD state at runtime (postprocessRefine).
    ph.MHDModel(
        density=density,
        vx=vx,
        vy=vy,
        vz=vz,
        bx=bx,
        by=by,
        bz=bz,
        p=p,
        protons={
            "charge": 1,
            "mass": 1,
            "nbr_part_per_cell": 10,
            "init": {"seed": 1333},
        },
    )

    ph.ElectronModel(closure="isothermal", Te=0.0)

    ph.ElectromagDiagnostics(quantity="B", write_timestamps=B_TIMESTAMPS)
    ph.ElectromagDiagnostics(quantity="E", write_timestamps=timestamps)

    for quantity in ["rho", "V", "P"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=timestamps)

    ph.FluidDiagnostics(quantity="mass_density", write_timestamps=timestamps)
    ph.FluidDiagnostics(quantity="bulkVelocity", write_timestamps=timestamps)

    return sim


# ---------------------------------------------------------------------------
# divB validation: first-L1-regrid detector + per-level divB report.
# gate=False: baseline mode, values reported only; gate=True: asserted vs DIVB_TOL.
# ---------------------------------------------------------------------------


def _l1_box_sets(run, times):
    """{t: sorted [(lower, upper), ...] of L1 patch boxes} for each dump time."""
    box_sets = {}
    for t in times:
        hier = run.GetB(t, all_primal=False)
        lvls = hier.levels(t)
        boxes = []
        if 1 in lvls:
            boxes = sorted(
                (
                    tuple(int(i) for i in p.box.lower),
                    tuple(int(i) for i in p.box.upper),
                )
                for p in lvls[1].patches
            )
        box_sets[t] = boxes
    return box_sets


def find_first_l1_regrid(run, times):
    """First consecutive (t_before, t_after) where the L1 patch-box set changes."""
    box_sets = _l1_box_sets(run, times)
    for t0, t1 in zip(times[:-1], times[1:]):
        if box_sets[t0] != box_sets[t1]:
            return t0, t1
    raise RuntimeError(
        f"no L1 patch-layout change found across {len(times)} B dumps over "
        f"t=[{times[0]}, {times[-1]}] — extend final_time to capture the first regrid"
    )


def divb_linf_per_level(run, t):
    """Interior-only Linf(divB) per level at time t."""
    scalar = run.GetDivB(t)
    linf = {}
    for ilvl, lvl in scalar.levels(t).items():
        vmax = 0.0
        for patch in lvl.patches:
            if not patch.patch_datas:
                continue
            pd = patch.patch_datas["value"]
            data = pd[patch.box] if np.any(pd.ghosts_nbr) else pd.dataset[:]
            vals = np.abs(np.asarray(data))
            if np.isnan(vals).any():
                print(
                    f"[divB] WARNING: NaN in interior divB "
                    f"L{ilvl} patch {patch.id} t={t:.3f}"
                )
            vmax = max(vmax, float(np.nanmax(vals)))
        linf[ilvl] = vmax
    return linf


def plot_divb_maps(run, t, plot_dir, tag):
    """All-levels + per-level divB maps at time t, symmetric color scale."""
    scalar = run.GetDivB(t)
    v = max(divb_linf_per_level(run, t).values())
    v = v if v > 0 else 1e-16
    scalar.plot(
        time=t,
        filename=f"{plot_dir}/divb_{tag}_t{t:.3f}.png",
        plot_patches=True,
        vmin=-v,
        vmax=+v,
    )
    for ilvl in sorted(scalar.levels(t).keys()):
        scalar.plot(
            time=t,
            levels=[ilvl],
            filename=f"{plot_dir}/divb_{tag}_L{ilvl}_t{t:.3f}.png",
            plot_patches=True,
            vmin=-v,
            vmax=+v,
        )


def plot_divb_timeseries(run, times, regrid_t, plot_dir, tag):
    """Linf(divB) vs time, one line per level; regrid marker + DIVB_TOL line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = {t: divb_linf_per_level(run, t) for t in times}
    levels = sorted({ilvl for per in table.values() for ilvl in per})
    fig, ax = plt.subplots()
    for ilvl in levels:
        ts = [t for t in times if ilvl in table[t]]
        ax.plot(ts, [table[t][ilvl] for t in ts], marker="o", label=f"L{ilvl}")
    ax.axvline(regrid_t, color="k", ls="--", label="first L1 regrid")
    ax.axhline(DIVB_TOL, color="r", ls=":", label=f"DIVB_TOL={DIVB_TOL:g}")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("Linf(divB)")
    ax.set_title(f"divB Linf per level ({tag})")
    ax.legend()
    fig.savefig(f"{plot_dir}/divb_linf_{tag}.png", dpi=200)
    plt.close(fig)


def divb_report(run, times, plot_dir, tag, gate):
    """divB report + plots. gate=False: baseline (print only); gate=True: assert."""
    t_before, t_after = find_first_l1_regrid(run, times)
    print(f"[divB] first L1 regrid: t_before={t_before:.3f} t_after={t_after:.3f}")

    table = {t: divb_linf_per_level(run, t) for t in times}
    levels = sorted({ilvl for per in table.values() for ilvl in per})
    print("[divB] time      " + "  ".join(f"L{ilvl} Linf(divB)" for ilvl in levels))
    for t in times:
        row = "  ".join(
            f"{table[t][ilvl]:>13.6e}" if ilvl in table[t] else f"{'-':>13}"
            for ilvl in levels
        )
        print(f"[divB] {t:8.3f}  {row}")
    for t, label in ((t_before, "before"), (t_after, "after")):
        vals = ", ".join(f"L{ilvl}={v:.6e}" for ilvl, v in sorted(table[t].items()))
        print(f"[divB] t_{label}={t:.3f}: {vals}")

    plot_divb_maps(run, t_before, plot_dir, tag)
    plot_divb_maps(run, t_after, plot_dir, tag)
    plot_divb_timeseries(run, times, t_after, plot_dir, tag)

    if gate:
        for t in (t_before, t_after):
            for ilvl, v in table[t].items():
                assert v <= DIVB_TOL, (
                    f"divB gate: L{ilvl} Linf(divB)={v:.6e} > {DIVB_TOL} at t={t:.3f}"
                )
    return t_before, t_after, table


class MHDHybridHarrisTest(SimulatorTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulator = None

    def tearDown(self):
        super().tearDown()
        if self.simulator is not None:
            self.simulator.reset()
        self.simulator = None
        ph.global_vars.sim = None

    def test_run(self):
        self.register_diag_dir_for_cleanup(diag_dir)
        Simulator(config()).run().reset()
        if cpp.mpi_rank() == 0:
            run = Run(diag_dir)

            # Verify the Hybrid level was actually created and has B data.
            # If tagging never fires, coupling code is untested — fail loudly.
            b_hier = run.GetB(final_time)
            levels = b_hier.levels(final_time)
            assert 1 in levels, (
                "Hybrid level 1 not found in B diagnostics — "
                "MHD tagger may not have fired on the initial Harris profile"
            )
            # Shared EM_B.h5: MHD manager owns level 0, hybrid manager level 1.
            assert 0 in levels, "MHD level 0 missing from shared EM_B.h5"

            def assert_no_nan(hier, tag):
                for ilvl, lvl in hier.levels(final_time).items():
                    for patch in lvl.patches:
                        for name, pd in patch.patch_datas.items():
                            assert not np.isnan(pd.dataset[:]).any(), (
                                f"NaN in {tag} L{ilvl} {name}"
                            )

            assert_no_nan(b_hier, "B")

            # Coupled getters stitch MHD-owned and hybrid-owned levels.
            rho = run.GetCoupledDensity(final_time)
            assert sorted(rho.levels(final_time).keys()) == [0, 1]
            assert_no_nan(rho, "coupled rho")

            v = run.GetCoupledV(final_time)
            assert sorted(v.levels(final_time).keys()) == [0, 1]
            assert_no_nan(v, "coupled V")

            # Exercise the viz path end to end (interim physics, values not asserted).
            plot_dir = Path(diag_dir) / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            rho.plot(time=final_time, filename=str(plot_dir / "coupled_rho.png"))
            for c in b_hier.quantities():
                b_hier.plot(
                    qty=c, time=final_time, filename=str(plot_dir / f"B{c}.png")
                )

            # Step 6: divB gate across the first L1 regrid (shared-id mechanism).
            # Baseline (two-step regrid) archived in plans dir: baseline-noperturb/.
            divb_report(run, B_TIMESTAMPS, str(plot_dir), tag="shared", gate=True)
        cpp.mpi_barrier()
        return self


if __name__ == "__main__":
    startMPI()
    MHDHybridHarrisTest().test_run().tearDown()
