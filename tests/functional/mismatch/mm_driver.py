#!/usr/bin/env python3
"""Shared-face mismatch repro driver (investigation only).

env:
  MM_LEVELS   1 or 2 (static refinement box on L0)
  MM_RECON    Linear | WENOZ
  MM_HALL     0/1
  MM_STEPPER  TVDRK2 | TVDRK3 | Euler | SSPRK4_5
  MM_STEPS    number of coarse steps
  MM_N        cells per dim
  MM_PATCH    largest patch size
  MM_CASE     ot | alfven | uniform
"""

import os
import numpy as np

import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator, startMPI

ph.NO_GUI()

levels = int(os.environ.get("MM_LEVELS", "1"))
recon = os.environ.get("MM_RECON", "WENOZ")
hall = os.environ.get("MM_HALL", "0") == "1"
stepper = os.environ.get("MM_STEPPER", "TVDRK2")
nsteps = int(os.environ.get("MM_STEPS", "3"))
N = int(os.environ.get("MM_N", "64"))
patch = int(os.environ.get("MM_PATCH", "16"))
case = os.environ.get("MM_CASE", "ot")
order = 2  # master: no mhd_order key

limiter = "VanLeer" if recon == "Linear" else "None"
dt = float(os.environ.get("MM_DT", 0.2 / N))
final_time = nsteps * dt


def config():
    kw = dict(
        smallest_patch_size=8,
        largest_patch_size=patch,
        time_step=dt,
        final_time=final_time,
        cells=(N, N),
        dl=(1.0 / N, 1.0 / N),
        hyper_resistivity=0.0,
        resistivity=0.0,
        strict=True,
        hyper_mode="spatial",
        eta=0.0,
        nu=float(os.environ.get("MM_NU", 0.02 if (hall and order == 2) else 0.0)),
        gamma=5.0 / 3.0,
        reconstruction=recon,
        limiter=limiter,
        riemann="Rusanov",
        mhd_timestepper=stepper,
        hall=hall,
        model_options=["MHDModel"],
        nesting_buffer=int(os.environ.get("MM_NESTBUF", "1")),
    )
    if levels == 1:
        kw.update(refinement="tagging", max_mhd_level=1, max_nbr_levels=1)
    elif levels < 0:  # dynamic tagging with -levels levels
        kw.update(
            refinement="tagging",
            max_mhd_level=-levels,
            max_nbr_levels=-levels,
            tagging_threshold=float(os.environ.get("MM_TAGTH", "0.1")),
        )
    else:
        lo, hi = N // 4, 3 * N // 4 - 1
        boxes = {"L0": {"B0": [(lo, lo), (hi, hi)]}}
        if levels == 3:
            # L1 index space: L0 box [lo,hi] -> [2lo, 2hi+1]; nest L2 with 2-cell margin
            m = int(os.environ.get("MM_L2MARGIN", "2"))
            l1lo, l1hi = 2 * lo + m, 2 * hi + 1 - m
            boxes["L1"] = {"B0": [(l1lo, l1lo), (l1hi, l1hi)]}
        kw.update(
            refinement="boxes",
            refinement_boxes=boxes,
            max_mhd_level=levels,
        )
    sim = ph.Simulation(**kw)

    B0 = 1.0 / np.sqrt(4.0 * np.pi)

    if case == "uniform":

        def density(x, y):
            return 1.0

        def vx(x, y):
            return 0.3

        def vy(x, y):
            return 0.2

        def bx(x, y):
            return 0.5

        def by(x, y):
            return 0.3

        def p(x, y):
            return 1.0

    elif case == "alfven":
        # 45 deg CP Alfven wave, k = 2pi(1,1): periodic on the unit square, Bx depends on x
        ca = sa = np.sqrt(0.5)

        def ph_(x, y):
            return 2.0 * np.pi * (x + y)

        def density(x, y):
            return 1.0

        def vx(x, y):
            return -0.1 * np.sin(ph_(x, y)) * sa

        def vy(x, y):
            return 0.1 * np.sin(ph_(x, y)) * ca

        def bx(x, y):
            return ca - 0.1 * np.sin(ph_(x, y)) * sa

        def by(x, y):
            return sa + 0.1 * np.sin(ph_(x, y)) * ca

        def p(x, y):
            return 0.1

    else:

        def density(x, y):
            return 25.0 / (36.0 * np.pi)

        def vx(x, y):
            return -np.sin(2.0 * np.pi * y)

        def vy(x, y):
            return np.sin(2.0 * np.pi * x)

        def bx(x, y):
            return -B0 * np.sin(2.0 * np.pi * y)

        def by(x, y):
            return B0 * np.sin(4.0 * np.pi * x)

        def p(x, y):
            return 5.0 / (12.0 * np.pi)

    def vz(x, y):
        return 0.0

    def bz(x, y):
        return 0.0

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)
    return sim


if __name__ == "__main__":
    startMPI()
    Simulator(config()).run().reset()
    ph.global_vars.sim = None
