"""Base class for AMR space-time convergence tests.

Protocol (classical AMR convergence, McCorquodale & Colella style):
- smooth exact solution whose state at t=final_time equals the initial
  condition (one wave period / traversal);
- STATIC refinement box fixed in physical space (central quarter of the
  domain), scaled with N so the coarse-fine boundary sits at the same physical
  location at every resolution, with the wave crossing it continuously;
- composite error at t=final_time against the t=0 dump of the same run
  (compute_errors module);
- spatial check: sweep N at fixed CFL number sigma; measured order = slope of
  log eps vs log N, asserted within a band;
- temporal check: at fixed N, sweep sigma and require the AMR/uniform error
  RATIO to stay near-constant. The uniform control carries the scheme's own
  O(dt^q) truncation error (visibly so: TVDRK2 at N=64 drifts ~15% over
  sigma in [0.4, 0.9]), so raw flatness is the wrong ask; dividing by the
  control removes that common mode, and any residual sigma-dependence of the
  ratio is coarse-fine specific (time interpolation of ghosts, refluxing).
  Calibration 2026-07-09 (alfven, N=64): ratio flat to 0.3% while both raw
  errors drift ~15%.

Subclasses provide the problem physics: see the "subclass contract" attributes
and methods below.
"""

import numpy as np

from pyphare.pharesee.run import Run
from pyphare.simulator.simulator import Simulator

from tests.simulator import SimulatorTest
from tests.simulator.amr_convergence import compute_errors


class ConvergenceTestBase(SimulatorTest):
    # ---- subclass contract -------------------------------------------------
    # name: str                      -> diag dirs under phare_outputs/<name>_amr_convergence
    # final_time: float              -> exact-return time of the wave
    # MAX_AMR_SIGMA_DRIFT: float     -> gate on AMR/uniform ratio drift (check_sigma_sweep)
    # def cfl_dt(self, N)            -> per-unit-sigma stable step
    # def amr_simulation(self, mhd_order, N, n) -> ph.Simulation via self.simulation()
    # def uniform_simulation(self, mhd_order, N, n) -> matching single-level control
    # def add_model_and_diags(self)  -> MHDModel closures + conserved-set diagnostics

    def n_steps(self, N, sigma):
        """Steps to final_time for dt = sigma*cfl_dt(N), snapped so that
        dt = final_time/n (<= target) and final_time is hit exactly."""
        return int(np.ceil(self.final_time / (sigma * self.cfl_dt(N))))

    def fine_box(self, N):
        # central quarter of the domain in cell coordinates: [N/4, 3N/4 - 1]
        lo, hi = N // 4, 3 * N // 4 - 1
        return [[lo, lo], [hi, hi]]

    @staticmethod
    def _physical_divb_growth(run, final_time):
        """Per-level, non-merged divB growth restricted to the physical (ghost-free)
        region, t=0 vs t=final_time. _compute_divB builds divB from the full B
        datasets (ghosts included) and drops ghosts_nbr, so dataset[:] still spans
        the ghost band -- strip B's own ghost margin (same recipe as
        tests/functional/refinement/divb_refinement.py's max_divb_per_level) rather
        than trust divB's own (unset) ghosts_nbr. The fine-level coarse-fine ghost
        fill is a known, order-independent divB hot-spot, not a physical interior
        violation, which is why merged=True (ghosts included via the interpolator)
        measured growth ~1e-4: it was seeing exactly that ghost band.
        """
        b0_levels = run.GetB(0.0, all_primal=False).levels(0.0)
        bx0 = next(iter(b0_levels.values())).patches[0].patch_datas["Bx"]
        ng = int(bx0.ghosts_nbr[0])

        def snapshot(time):
            out = {}
            for ilvl, level in run.GetDivB(time).levels(time).items():
                for patch in level.patches:
                    key = (ilvl, tuple(patch.box.lower), tuple(patch.box.upper))
                    arr = patch.patch_datas["value"].dataset[:]
                    if all(s > 2 * ng for s in arr.shape):
                        arr = arr[tuple(slice(ng, -ng) for _ in arr.shape)]
                    out[key] = arr
            return out

        initial, final = snapshot(0.0), snapshot(final_time)
        if initial.keys() != final.keys():
            raise RuntimeError(
                "divB patch layout changed between t=0 and t=final_time: "
                f"{sorted(initial.keys() ^ final.keys())}"
            )
        growth = 0.0
        for key in initial:
            if initial[key].shape != final[key].shape:
                raise RuntimeError(
                    f"divB shape mismatch at {key}: "
                    f"{initial[key].shape} != {final[key].shape}"
                )
            growth = max(growth, float(np.max(np.abs(final[key] - initial[key]))))
        return growth

    def run_amr_case(self, mhd_order, N, n):
        """One AMR run; (per_level eps, composite eps), None under PHARE_DRY_RUN."""
        sim = self.amr_simulation(mhd_order, N, n)
        self.add_model_and_diags()
        Simulator(sim).run().reset()
        if sim.dry_run:  # setup only: nothing advanced, no diagnostics to read back
            return None
        run = Run(sim.diag_options["options"]["dir"])
        divb_growth = self._physical_divb_growth(run, self.final_time)
        self.assertLessEqual(
            divb_growth, 1e-11,
            f"{self.name} MHD{mhd_order}: physical divB growth "
            f"{divb_growth:.3e} exceeds 1e-11",
        )
        return compute_errors.composite_errors(run, self.final_time, self.fine_box(N))

    def run_uniform_case(self, mhd_order, N, n):
        """One single-level control run; eps, None under PHARE_DRY_RUN."""
        sim = self.uniform_simulation(mhd_order, N, n)
        self.add_model_and_diags()
        Simulator(sim).run().reset()
        if sim.dry_run:  # setup only: nothing advanced, no diagnostics to read back
            return None
        run = Run(sim.diag_options["options"]["dir"])
        return compute_errors.uniform_error(run, self.final_time)

    def check_spatial_order(self, mhd_order, Ns, sigma, band):
        """N-sweep at fixed sigma; assert the composite convergence order."""
        print(
            f"\n== {self.name}: spatial convergence, order={mhd_order}, "
            f"sigma={sigma} =="
        )
        rows = []
        for N in Ns:
            n = self.n_steps(N, sigma)
            case = self.run_amr_case(mhd_order, N, n)
            if case is None:  # dry run: keep sweeping so every deck is built
                continue
            per_level, composite = case
            rows.append((N, composite, per_level))
            lvls = "  ".join(
                f"L{level}={err:.3e}" for level, err in sorted(per_level.items())
            )
            print(
                f"N={N:4d}  dt={self.final_time/n:.3e} (n={n})  "
                f"composite={composite:.3e}  {lvls}"
            )
        if not rows:  # dry run: decks constructed, no errors to fit an order to
            print("  dry run: setup only, convergence order not checked")
            return
        for (Na, ea, _), (Nb, eb, _) in zip(rows, rows[1:]):
            print(f"  segment N={Na}->{Nb}: order {np.log(ea/eb)/np.log(Nb/Na):.2f}")
        errs = [e for _, e, _ in rows]
        slope = -np.polyfit(np.log(Ns), np.log(errs), 1)[0]
        print(f"  measured order (all N): {slope:.2f}   expected band {band}")
        self.assertTrue(
            band[0] <= slope <= band[1],
            f"{self.name} order={mhd_order}: measured spatial order {slope:.2f} "
            f"outside {band}; errors {list(zip(Ns, errs))}",
        )

    def check_sigma_sweep(self, mhd_order, N, sigmas):
        """Fixed-N sigma sweep: the AMR/uniform error ratio must stay flat vs
        sigma. drift = (max - min)/min over the sweep. Raw drifts are printed
        for information; they include the scheme's own O(dt^q) truncation
        error and are not gated."""
        by_n = {}  # snapped n -> sigma (dedupe: two sigmas can give the same n)
        for sigma in sorted(sigmas):
            by_n.setdefault(self.n_steps(N, sigma), sigma)
        self.assertGreaterEqual(
            len(by_n), 3,
            f"sigma sweep needs >=3 distinct step counts at N={N}, "
            f"got n={sorted(by_n)} from sigmas={sorted(sigmas)}",
        )

        print(f"\n== {self.name}: sigma sweep, order={mhd_order}, N={N} ==")
        uni, amr_comp, amr_fine = [], [], []
        for n, sigma in sorted(by_n.items(), reverse=True):  # small dt -> large dt
            case = self.run_amr_case(mhd_order, N, n)
            eps_uni = self.run_uniform_case(mhd_order, N, n)
            if case is None:  # dry run: keep sweeping so every deck is built
                continue
            per_level, composite = case
            uni.append(eps_uni)
            amr_comp.append(composite)
            amr_fine.append(per_level[1])
            print(
                f"sigma={sigma:5.3f}  dt={self.final_time/n:.3e} (n={n})  "
                f"uniform={eps_uni:.6e}  composite={composite:.6e}  "
                f"L1(fine)={per_level[1]:.6e}"
            )

        if not uni:  # dry run: decks constructed, no errors to measure drift on
            print("  dry run: setup only, sigma drift not checked")
            return

        def drift(errs):
            return (max(errs) - min(errs)) / min(errs)

        # raw drifts: informational only -- they carry the scheme's own
        # O(dt^q) truncation error, common to AMR and uniform.
        print(
            f"  raw drift (not gated): uniform={drift(uni):.3f}  "
            f"composite={drift(amr_comp):.3f}  L1(fine)={drift(amr_fine):.3f}"
        )
        ratios = {
            "composite/uniform": [a / u for a, u in zip(amr_comp, uni)],
            "L1(fine)/uniform": [a / u for a, u in zip(amr_fine, uni)],
        }
        for label, vals in ratios.items():
            d = drift(vals)
            print(f"  ratio {label}: {['%.4f' % v for v in vals]}  drift={d:.4f}")
            self.assertLessEqual(
                d, self.MAX_AMR_SIGMA_DRIFT,
                f"{self.name} order={mhd_order} N={N}: {label} error ratio drifts "
                f"{d:.3f} > {self.MAX_AMR_SIGMA_DRIFT} over the sigma sweep -- "
                f"sigma-dependent AMR-specific error (coarse-fine time "
                f"interpolation defect); ratios {vals}",
            )
