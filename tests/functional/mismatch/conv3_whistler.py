#!/usr/bin/env python3
"""3-level AMR convergence probe (investigation only): 2D rotated Hall-MHD whistler
(same physics as tests/simulator/amr_convergence/test_whistler2d_amr.py), one traversal,
L0 box = central quarter, L2 box = L1 box shrunk by CV_MARGIN L1 cells (as conv3.py).

dt: the whistler bound dt ~ dx^2 while subcycling halves dt per level, so the effective
sigma on level l is sigma * 2^l; the 2-level test runs sigma 0.4 (L1 at 0.8), the default
here is 0.2 (L2 at 0.8).

env: CV_N, CV_ORDER, CV_LEVELS, CV_MARGIN, CV_SIGMA, CV_PATCH, CV_OUT (see conv3.py)
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
sigma = float(os.environ.get("CV_SIGMA", "0.2"))
out = os.environ["CV_OUT"]

tan_alpha = 0.5
cosalpha = 1.0 / np.sqrt(1 + tan_alpha**2)
sinalpha = tan_alpha / np.sqrt(1 + tan_alpha**2)

B0, RHO0, P0 = 1.0, 1.0, 1.0
GAMMA = 5.0 / 3.0
DELTA = 1e-3

c_A = B0 / np.sqrt(RHO0)
c_s = np.sqrt(GAMMA * P0 / RHO0)
v_fast = np.sqrt(c_s**2 + c_A**2)

K = 0.19
L = 2 * np.pi / K
OMEGA = 0.5 * K**2 + np.sqrt((K * c_A) ** 2 + (0.5 * K**2) ** 2)
C_W = OMEGA / K
V_AMP = DELTA * c_A / C_W

e1 = np.array([cosalpha, sinalpha, 0.0])
e2 = np.array([-sinalpha, cosalpha, 0.0])
e3 = np.cross(e1, e2)
B0_vec = B0 * e1

Lx = L / cosalpha
Ly = L / sinalpha
final_time = L / C_W


def cfl_dt(N):
    dxmin = min(Lx / N, Ly / N)
    return min(dxmin**2 / np.pi, dxmin / v_fast)


def boxes(N):
    lo, hi = N // 4, 3 * N // 4 - 1
    b = {"L0": {"B0": [(lo, lo), (hi, hi)]}}
    if levels == 3:
        l1lo, l1hi = 2 * lo + margin, 2 * hi + 1 - margin
        b["L1"] = {"B0": [(l1lo, l1lo), (l1hi, l1hi)]}
    return b


def config():
    n = int(np.ceil(final_time / (sigma * cfl_dt(N))))
    print(f"whistler3: N={N} levels={levels} margin={margin} sigma={sigma} n={n}")
    kw = dict(
        smallest_patch_size=8,
        largest_patch_size=int(os.environ.get("CV_PATCH", "16")),
        time_step=final_time / n,
        final_time=final_time,
        cells=(N, N),
        dl=(Lx / N, Ly / N),
        hyper_resistivity=0.0,
        resistivity=0.0,
        strict=True,
        nesting_buffer=1,
        eta=0.0,
        gamma=GAMMA,
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="SSPRK4_5",
        mhd_order=order,
        hall=True,
        model_options=["MHDModel"],
        diag_options={"format": "phareh5", "options": {"dir": out, "mode": "overwrite"}},
    )
    if levels == 1:
        kw.update(refinement="tagging", max_mhd_level=1, max_nbr_levels=1)
    else:
        kw.update(refinement="boxes", refinement_boxes=boxes(N), max_mhd_level=levels)
    sim = ph.Simulation(**kw)

    def phase(x, y):
        return K * (x * e1[0] + y * e1[1])

    def dB(x, y):
        p = phase(x, y)
        return DELTA * (np.cos(p)[:, None] * e2 - np.sin(p)[:, None] * e3)

    def dV(x, y):
        p = phase(x, y)
        return V_AMP * (-np.cos(p)[:, None] * e2 + np.sin(p)[:, None] * e3)

    def density(x, y):
        return RHO0

    def p(x, y):
        return P0

    def vx(x, y):
        return dV(x, y)[:, 0]

    def vy(x, y):
        return dV(x, y)[:, 1]

    def vz(x, y):
        return dV(x, y)[:, 2]

    def bx(x, y):
        return B0_vec[0] + dB(x, y)[:, 0]

    def by(x, y):
        return B0_vec[1] + dB(x, y)[:, 1]

    def bz(x, y):
        return B0_vec[2] + dB(x, y)[:, 2]

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)
    ts = [0.0, sim.time_stepper.final_time]  # the step-snapped final time
    ph.ElectromagDiagnostics(quantity="B", write_timestamps=ts)
    for q in ["rho", "rhoV", "Etot"]:
        ph.MHDDiagnostics(quantity=q, write_timestamps=ts)
    return sim


if __name__ == "__main__":
    startMPI()
    Simulator(config()).run().reset()
    ph.global_vars.sim = None
