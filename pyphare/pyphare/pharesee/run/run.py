#
#
#

import os
import glob
import numpy as np
from pathlib import Path


from pyphare.pharesee.hierarchy import all_times_from
from pyphare.pharesee.hierarchy import default_time_from
from pyphare.pharesee.hierarchy import hierarchy_from
from pyphare.pharesee.hierarchy import hierarchy_compute as hc
from pyphare.pharesee.hierarchy import ScalarField, VectorField

from pyphare.core import phare_utilities as phut
from pyphare.pharesee.hierarchy.hierarchy_utils import compute_hier_from
from pyphare.pharesee.hierarchy.hierarchy_utils import flat_finest_field

from pyphare.logger import getLogger

from .man import RunMan

from .utils import (
    _compute_current,
    _compute_divB,
    _compute_pop_pressure,
    _compute_pressure,
    _compute_to_primal,
    _get_rank,
    make_interpolator,
)

logger = getLogger(__name__)


class Run:
    def __init__(self, path, default_time=None):
        self.path = path
        self.default_time_ = default_time
        self.available_diags = self._available_diags()

    def GetTags(self, time, merged=False, **kwargs):
        hier = self._get_hierarchy(time, "tags.h5")
        return self._get(hier, time, merged, "nearest")

    def GetB(self, time, merged=False, interp="nearest", all_primal=True, **kwargs):
        if merged:
            all_primal = False
        hier = self._get_hier_for(time, "EM_B", **kwargs)
        if not all_primal:
            return self._get(hier, time, merged, interp)

        return VectorField(compute_hier_from(_compute_to_primal, hier))

    def GetE(self, time, merged=False, interp="nearest", all_primal=True, **kwargs):
        if merged:
            all_primal = False
        hier = self._get_hier_for(time, "EM_E", **kwargs)
        if not all_primal:
            return self._get(hier, time, merged, interp)

        return VectorField(compute_hier_from(_compute_to_primal, hier))

    def GetMassDensity(self, time, merged=False, interp="nearest", **kwargs):
        hier = self._get_hier_for(time, "ions_mass_density", **kwargs)
        return ScalarField(self._get(hier, time, merged, interp))

    def GetNi(self, time, merged=False, interp="nearest", **kwargs):
        hier = self._get_hier_for(time, "ions_charge_density", **kwargs)
        return ScalarField(self._get(hier, time, merged, interp, drop_ghosts=True))

    def GetN(self, time, pop_name, merged=False, interp="nearest", **kwargs):
        hier = self._get_hier_for(time, f"ions_pop_{pop_name}_density", **kwargs)
        return ScalarField(self._get(hier, time, merged, interp))

    def GetVi(self, time, merged=False, interp="nearest", **kwargs):
        hier = self._get_hier_for(time, "ions_bulkVelocity", **kwargs)
        return VectorField(self._get(hier, time, merged, interp, drop_ghosts=True))
        return RunMan(self).GetVi(time, merged, interp, **kwargs)

    def GetFlux(self, time, pop_name, merged=False, interp="nearest", **kwargs):
        hier = self._get_hier_for(time, f"ions_pop_{pop_name}_flux", **kwargs)
        return VectorField(self._get(hier, time, merged, interp))

    def GetPressure(self, time, pop_name, merged=False, interp="nearest", **kwargs):
        # return RunMan(self).GetPressure(time, pop_name, merged, interp, **kwargs)
        M = self._get_hierarchy(
            time, f"ions_pop_{pop_name}_momentum_tensor.h5", **kwargs
        )
        V = self.GetFlux(time, pop_name, **kwargs)
        N = self.GetN(time, pop_name, **kwargs)
        P = compute_hier_from(
            _compute_pop_pressure,
            (M, V, N),
            popname=pop_name,
            mass=self.GetMass(pop_name, **kwargs),
        )
        return self._get(P, time, merged, interp)  # should later be a TensorField
        return RunMan(self).GetPressure(time, pop_name, merged, interp, **kwargs)

    def GetPi(self, time, merged=False, interp="nearest", **kwargs):
        M = self._get_hierarchy(time, "ions_momentum_tensor.h5", **kwargs)
        massDensity = self.GetMassDensity(time, **kwargs)
        Vi = self._get_hier_for(time, "ions_bulkVelocity", **kwargs)
        Pi = compute_hier_from(_compute_pressure, (M, massDensity, Vi))
        return self._get(Pi, time, merged, interp)  # should later be a TensorField

    def GetPe(self, time, merged=False, interp="nearest", all_primal=True):
        hier = self._get_hier_for(time, "ions_charge_density")
        Te = hier.sim.electrons.closure.Te
        if not all_primal:
            return Te * self._get(hier, time, merged, interp)
        h = compute_hier_from(hc.drop_ghosts, hier)
        return ScalarField(h) * Te

    def GetJ(self, time, merged=False, interp="nearest", all_primal=True, **kwargs):
        if merged:
            all_primal = False
        B = self.GetB(time, all_primal=False, **kwargs)
        J = compute_hier_from(_compute_current, B)
        if not all_primal:
            return self._get(J, time, merged, interp)
        J = hc.rename(compute_hier_from(_compute_to_primal, J), ["Jx", "Jy", "Jz"])
        return VectorField(J)

    def GetDivB(self, time, merged=False, interp="nearest", **kwargs):
        B = self.GetB(time, all_primal=False, **kwargs)
        db = compute_hier_from(_compute_divB, B)
        return ScalarField(self._get(db, time, merged, interp))

    def GetMHDrho(
        self, time, merged=False, interp="nearest", all_primal=True, **kwargs
    ):
        if merged:
            all_primal = False
        hier = self._get_hier_for(time, "mhd_rho", **kwargs)
        if not all_primal:
            return self._get(hier, time, merged, interp)

        h = compute_hier_from(_compute_to_primal, hier, value="mhdRho")
        return ScalarField(h)

    def GetMHDV(self, time, merged=False, interp="nearest", all_primal=True, **kwargs):
        if merged:
            all_primal = False
        hier = self._get_hier_for(time, "mhd_V", **kwargs)
        if not all_primal:
            return self._get(hier, time, merged, interp)

        h = compute_hier_from(_compute_to_primal, hier, x="mhdVx", y="mhdVy", z="mhdVz")
        return VectorField(h)

    def GetMHDP(self, time, merged=False, interp="nearest", all_primal=True, **kwargs):
        if merged:
            all_primal = False
        hier = self._get_hier_for(time, "mhd_P", **kwargs)
        if not all_primal:
            return self._get(hier, time, merged, interp)

        h = compute_hier_from(_compute_to_primal, hier, value="mhdP")
        return ScalarField(h)

    def GetMHDrhoV(
        self, time, merged=False, interp="nearest", all_primal=True, **kwargs
    ):
        if merged:
            all_primal = False
        hier = self._get_hier_for(time, "mhd_rhoV", **kwargs)
        if not all_primal:
            return self._get(hier, time, merged, interp)

        h = compute_hier_from(
            _compute_to_primal, hier, x="mhdRhoVx", y="mhdRhoVy", z="mhdRhoVz"
        )
        return VectorField(h)

    def GetMHDEtot(
        self, time, merged=False, interp="nearest", all_primal=True, **kwargs
    ):
        if merged:
            all_primal = False
        hier = self._get_hier_for(time, "mhd_Etot", **kwargs)
        if not all_primal:
            return self._get(hier, time, merged, interp)

        h = compute_hier_from(_compute_to_primal, hier, value="mhdEtot")
        return ScalarField(h)

    def _az_at_point(self, bx_i, by_i, x, y, x_ref=0.0, y_ref=0.0, steps=200):
        """L-shaped line integral from (x_ref, y_ref) to (x, y) using loaded interpolators."""
        x_path = np.linspace(x_ref, x, steps)
        az_x = -np.trapz(np.ravel(by_i(x_path, np.full(steps, y_ref))), x_path)
        y_path = np.linspace(y_ref, y, steps)
        az_y = np.trapz(np.ravel(bx_i(np.full(steps, x), y_path)), y_path)
        return az_x + az_y

    def GetMagneticFlux(self, time, interp="nearest", target_pos=None, xn=None, yn=None):
        """
        Computes Az.
        If target_pos=(x, y) is provided, uses a fast 1D line integral.
        Otherwise, performs a 2D integration (slower).
        """
        merged_B = self.GetB(time, merged=True, interp=interp)
        bx_interp = merged_B["Bx"][0]
        by_interp = merged_B["By"][0]

        if target_pos is not None:
            tx, ty = target_pos
            return self._az_at_point(bx_interp, by_interp, tx, ty)

        # Full 2D map
        if xn is None or yn is None:
            domain = self.GetDomainSize()
            dl = self.GetDl(level="finest", time=time)
            xn = np.arange(0, domain[0] + dl[0], dl[0])
            yn = np.arange(0, domain[1] + dl[1], dl[1])

        Xn, Yn = np.meshgrid(xn, yn, indexing="ij")
        bx = bx_interp(Xn, Yn)
        by = by_interp(Xn, Yn)

        Az_x0 = -np.cumsum(by[:, 0]) * (xn[1] - xn[0])
        Az = np.cumsum(bx, axis=1) * (yn[1] - yn[0])
        Az += Az_x0[:, np.newaxis]

        return Az, (xn, yn)

    @staticmethod
    def _crop_indices(xn, yn, search_box):
        """Return (ix0, ix1, iy0, iy1) slice indices for search_box; None = full domain."""
        if search_box is None:
            return 0, len(xn), 0, len(yn)
        x_min, x_max, y_min, y_max = search_box
        ix0, ix1 = np.searchsorted(xn, [x_min, x_max])
        iy0, iy1 = np.searchsorted(yn, [y_min, y_max])
        return int(ix0), int(ix1), int(iy0), int(iy1)

    def _find_xpoint_min_az(self, bx_i, by_i, xn, yn, search_box=None, n_eval=400):
        """
        Finds an X-point as a saddle of Az (det_H < 0) within the search box.

        B is evaluated on a fine independent grid (n_eval x n_eval) within the
        padded search box. This ensures the Hessian is computed at a resolution that
        properly samples fine AMR data — the base-level grid from make_interpolator
        is too coarse to give accurate det_H signs near plasmoid O-points.

        Among saddle candidates, selects the global min-|B|².

        Returns (x_xp, y_xp) or None if no saddle point exists in the search region.
        """
        if search_box is None:
            sx0, sx1, sy0, sy1 = xn[0], xn[-1], yn[0], yn[-1]
        else:
            sx0, sx1, sy0, sy1 = search_box

        # Pad by 10% of box size so crop-interior points use central differences
        pad = 0.1
        dx_box = sx1 - sx0
        dy_box = sy1 - sy0
        x0p = max(xn[0], sx0 - pad * dx_box)
        x1p = min(xn[-1], sx1 + pad * dx_box)
        y0p = max(yn[0], sy0 - pad * dy_box)
        y1p = min(yn[-1], sy1 + pad * dy_box)

        xn_eval = np.linspace(x0p, x1p, n_eval)
        yn_eval = np.linspace(y0p, y1p, n_eval)
        Xe, Ye = np.meshgrid(xn_eval, yn_eval, indexing="ij")

        bx = bx_i(Xe, Ye)
        by = by_i(Xe, Ye)

        dbx_dx = np.gradient(bx, xn_eval, axis=0)
        dbx_dy = np.gradient(bx, yn_eval, axis=1)
        dby_dx = np.gradient(by, xn_eval, axis=0)
        det_H = (-dby_dx * dbx_dy) - (dbx_dx**2)

        # Restrict to unpadded search region
        ix0, ix1 = np.searchsorted(xn_eval, [sx0, sx1])
        iy0, iy1 = np.searchsorted(yn_eval, [sy0, sy1])
        xn_c = xn_eval[int(ix0):int(ix1)]
        yn_c = yn_eval[int(iy0):int(iy1)]
        det_c = det_H[int(ix0):int(ix1), int(iy0):int(iy1)]
        bx_c = bx[int(ix0):int(ix1), int(iy0):int(iy1)]
        by_c = by[int(ix0):int(ix1), int(iy0):int(iy1)]

        saddle_mask = det_c < 0
        if not np.any(saddle_mask):
            return None

        Xc, Yc = np.meshgrid(xn_c, yn_c, indexing="ij")

        b2_c = bx_c**2 + by_c**2
        masked = np.where(saddle_mask, b2_c, np.inf)

        idx = np.unravel_index(np.argmin(masked), masked.shape)
        return Xc[idx], Yc[idx]


    def GetReconnectionRate(self, times, interp="nearest", search_box=None):
        """
        Computes reconnection rate dAz/dt at an X-point over time.

        At each timestep, picks the saddle (det_H < 0) with global min-|B|² in
        the search box. No tracking between timesteps.

        :param times: sequence of simulation times
        :param interp: interpolation method ('nearest' or 'bilinear')
        :param search_box: (x_min, x_max, y_min, y_max) in physical coordinates;
                           None searches the full domain
        :returns: (t_mid, rates, flux_at_xpoint, xpoint_trajectory)
        """
        flux_at_xpoint = []
        xpoint_trajectory = []

        for t in times:
            merged_B = self.GetB(t, merged=True, interp=interp)
            bx_i, by_i = merged_B["Bx"][0], merged_B["By"][0]
            xn, yn = merged_B["Bx"][1]

            result = self._find_xpoint_min_az(bx_i, by_i, xn, yn, search_box)
            if result is None:
                raise RuntimeError(f"No saddle point found at t={t:.4f}")
            x_xp, y_xp = result

            az_val = self._az_at_point(bx_i, by_i, x_xp, y_xp)
            xpoint_trajectory.append([x_xp, y_xp])
            flux_at_xpoint.append(az_val)

        flux_at_xpoint = np.array(flux_at_xpoint)
        rates = np.diff(flux_at_xpoint) / np.diff(times)
        t_mid = (times[:-1] + times[1:]) / 2

        return t_mid, rates, flux_at_xpoint, np.array(xpoint_trajectory)

    def GetRanks(self, time, merged=False, interp="nearest", **kwargs):
        """
        returns a hierarchy of MPI ranks
        takes the information from magnetic field diagnostics arbitrarily
        this fails if the magnetic field is not written and could at some point
        be replace by a search of any available diag at the requested time.
        """
        B = self.GetB(time, all_primal=False, **kwargs)
        ranks = compute_hier_from(_get_rank, B)
        return ScalarField(self._get(ranks, time, merged, interp))

    def GetParticles(self, time, pop_name, hier=None, **kwargs):
        def filename(name):
            return f"ions_pop_{name}_domain.h5"

        if isinstance(pop_name, (list, tuple)):
            for pop in pop_name:
                hier = self._get_hierarchy(time, filename(pop), hier=hier, **kwargs)
            return hier
        return self._get_hierarchy(time, filename(pop_name), hier=hier, **kwargs)

    def GetParticleCount(self, time, **kwargs):
        c = self._get_hierarchy(time, "particle_count.h5", **kwargs)
        return c

    def GetMass(self, pop_name, **kwargs):
        list_of_qty = ["density", "flux", "domain", "levelGhost"]
        list_of_mass = []

        import h5py

        for qty in list_of_qty:
            file = os.path.join(self.path, "ions_pop_{}_{}.h5".format(pop_name, qty))
            if os.path.isfile(file):
                h5_file = h5py.File(file, "r")
                list_of_mass.append(h5_file.attrs["pop_mass"])

        assert all(m == list_of_mass[0] for m in list_of_mass)

        return list_of_mass[0]

    def GetDomainSize(self, **kwargs):
        hier = self._get_any_hierarchy(self.default_time)
        root_cell_width = hier.level(0).cell_width
        domain_box = hier.domain_box
        return (domain_box.upper + 1) * root_cell_width

    def GetDl(self, level="finest", time=None):
        """
        gives the ndarray containing the grid sizes at the given time
        for the hierarchy defined in the given run, and for the given level
        (default is 'finest', but can also be a int)

        :param level: the level at which get the associated grid size
        :param time: the time because level depends on it
        """

        time = time if time is not None else self.default_time
        hier = self._get_any_hierarchy(time)
        level = hier.finest_level(time) if level == "finest" else level
        return hier.level(level).cell_width

    def all_times(self):
        return {Path(file).stem: all_times_from(file) for file in self.available_diags}

    def times(self, qty):
        return self.all_times()[qty]

    def _available_diags(self):
        files = glob.glob(os.path.join(self.path, "*.h5"))
        if files:
            return files
        files = glob.glob(os.path.join(self.path, "*.vtkhdf"))
        if files:
            return files
        raise RuntimeError(f"No HDF5 files found at: {self.path}")

    def _get_hier_for(self, time, qty, **kwargs):
        path = Path(self.path) / f"{qty}.h5"
        if path.exists():
            return self._get_hierarchy(time, f"{qty}.h5", **kwargs)
        path = Path(self.path) / f"{qty}.vtkhdf"
        if path.exists():
            return self._get_hierarchy(time, f"{qty}.vtkhdf", **kwargs)
        raise RuntimeError(f"No HDF5 file found for: {qty}")

    def _get_any_hierarchy(self, time):
        ref_file = Path(self.available_diags[0]).stem
        return self._get_hier_for(time, ref_file)

    def _get_hierarchy(self, times, filename, hier=None, **kwargs):
        from pyphare.core.box import Box

        times = phut.listify(times)
        times = [f"{t:.10f}" for t in times]
        if "selection_box" in kwargs:
            if isinstance(kwargs["selection_box"], tuple):
                lower = kwargs["selection_box"][:2]
                upper = kwargs["selection_box"][2:]
                kwargs["selection_box"] = Box(lower, upper)

        return hierarchy_from(
            h5_filename=os.path.join(self.path, filename),
            times=times,
            hier=hier,
            **kwargs,
        )

    # TODO maybe transform that so multiple times can be accepted
    def _get(self, hierarchy, time, merged, interp, drop_ghosts=False):
        """
        if merged=True, will return an interpolator and a tuple of 1d arrays
        with the coordinates of the finest grid where the interpolator
        can be calculated (that is the return of flat_finest_field)
        """
        if merged:
            domain = self.GetDomainSize()
            dl = self.GetDl(time=time)

            # assumes all qties in the hierarchy have the same ghost width
            # so take the first patch data of the first patch of the first level....
            nbrGhosts = list(hierarchy.level(0).patches[0].patch_datas.values())[
                0
            ].ghosts_nbr
            merged_qties = {}
            for qty in hierarchy.quantities():
                data, coords = flat_finest_field(hierarchy, qty, time=time)
                merged_qties[qty] = make_interpolator(
                    data, coords, interp, domain, dl, qty, nbrGhosts
                )
            return merged_qties
        else:
            return (
                compute_hier_from(hc.drop_ghosts, hierarchy)
                if drop_ghosts
                else hierarchy
            )

    @property
    def default_time(self):
        if self.default_time_ is None:
            ref_file = self.available_diags[0]
            self.default_time_ = default_time_from(ref_file)
        return self.default_time_
