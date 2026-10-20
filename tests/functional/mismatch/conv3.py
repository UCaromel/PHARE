#!/usr/bin/env python3
"""3-level AMR convergence probe (investigation only): 2D oblique CP Alfven wave
(same physics as tests/simulator/amr_convergence/test_alfven2d_amr.py), one period,
L0 box = central quarter, L2 box = L1 box shrunk by CV_MARGIN L1 cells, so the L2 ghost
stencil reaches past L1 when the margin is small.

env:
  CV_N        L0 cells per dim
  CV_ORDER    2 | 4
  CV_LEVELS   1 | 2 | 3
  CV_MARGIN   L2 margin inside the L1 box, in L1 cells (levels=3)
  CV_SIGMA    fraction of the CFL step
  CV_OUT      diagnostics directory
"""

import os
import numpy as np

import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator, startMPI

ph.NO_GUI()

N = int(os.environ.get("CV_N", "32"))
order = int(os.environ.get("CV_ORDER", "4"))
levels = int(os.environ.get("CV_LEVELS", "3"))
margin = int(os.environ.get("CV_MARGIN", "2"))
sigma = float(os.environ.get("CV_SIGMA", "0.32"))
out = os.environ["CV_OUT"]

alpha = 30.0 * np.pi / 180.0
cosalpha, sinalpha = np.cos(alpha), np.sin(alpha)
GAMMA = 5.0 / 3.0
P0, RHO0, DB, DV = 0.1, 1.0, 0.1, 0.1
final_time = 1.0


def cfl_dt(N):
    dx = (1.0 / N) / cosalpha
    dy = (1.0 / N) / sinalpha
    c = DV + np.sqrt(GAMMA * P0 / RHO0 + (1.0 + DB**2) / RHO0)
    return 1.0 / (c / dx + c / dy)


def boxes(N):
    lo, hi = N // 4, 3 * N // 4 - 1
    b = {"L0": {"B0": [(lo, lo), (hi, hi)]}}
    if levels == 3:
        l1lo, l1hi = 2 * lo + margin, 2 * hi + 1 - margin
        b["L1"] = {"B0": [(l1lo, l1lo), (l1hi, l1hi)]}
    return b


def config():
    n = int(np.ceil(final_time / (sigma * cfl_dt(N))))
    kw = dict(
        smallest_patch_size=8,
        largest_patch_size=int(os.environ.get("CV_PATCH", "16")),
        time_step=final_time / n,
        final_time=final_time,
        cells=(N, N),
        dl=((1.0 / N) / cosalpha, (1.0 / N) / sinalpha),
        hyper_resistivity=0.0,
        resistivity=0.0,
        strict=True,
        nesting_buffer=1,
        eta=0.0,
        nu=0.0,
        gamma=GAMMA,
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="SSPRK4_5",
        mhd_order=order,
        hall=False,
        model_options=["MHDModel"],
        diag_options={"format": "phareh5", "options": {"dir": out, "mode": "overwrite"}},
    )
    if levels == 1:
        kw.update(refinement="tagging", max_mhd_level=1, max_nbr_levels=1)
    else:
        kw.update(refinement="boxes", refinement_boxes=boxes(N), max_mhd_level=levels)
    sim = ph.Simulation(**kw)

    def phase(x, y):
        return 2 * np.pi * (x * cosalpha + y * sinalpha)

    def density(x, y):
        return RHO0

    def vx(x, y):
        return -DV * np.sin(phase(x, y)) * sinalpha

    def vy(x, y):
        return DV * np.sin(phase(x, y)) * cosalpha

    def vz(x, y):
        return DV * np.cos(phase(x, y))

    def bx(x, y):
        return cosalpha - DB * np.sin(phase(x, y)) * sinalpha

    def by(x, y):
        return sinalpha + DB * np.sin(phase(x, y)) * cosalpha

    def bz(x, y):
        return DB * np.cos(phase(x, y))

    def p(x, y):
        return P0

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)
    ts = [0.0, final_time]
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=ts)
    for q in ["rho", "rhoV", "Etot"]:
        ph.MHDDiagnostics(quantity=q, write_timestamps=ts)
    return sim


if __name__ == "__main__":
    startMPI()
    Simulator(config()).run().reset()
    ph.global_vars.sim = None
