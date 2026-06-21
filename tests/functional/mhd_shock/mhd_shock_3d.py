#!/usr/bin/env python3
import numpy as np
import pyphare.pharein as ph
from pyphare import cpp
from pyphare.simulator.simulator import Simulator, startMPI
from pyphare.pharesee.run import Run

ph.NO_GUI()

def config():
    cosa, sina = 2/np.sqrt(5), 1/np.sqrt(5)
    cosg, sing = 2*np.sqrt(5)/np.sqrt(21), -1/np.sqrt(21)

    T = np.array([
        [ cosa*cosg, -sina, cosa*sing],
        [ sina*cosg,  cosa, sina*sing],
        [-sing,       0,    cosg     ]
    ])

    n = T[:, 0] 

    L = 1.0
    nx = 960
    sim = ph.Simulation(
        time_step=0.0001,
        final_time=4./np.sqrt(21.) * 0.2,
        cells=(nx, 10, 10),
        dl=(L/nx, (L/96)/10, (L/96)/10),
        interp_order=2,
        reconstruction="WENOZ",
        limiter="None",
        riemann="Rusanov",
        mhd_timestepper="SSPRK4_5",
        diag_options={
            "format": "phareh5",
            "options": {"dir": "rotated_shock", "mode": "overwrite"},
        },
        gamma   = 5./3.,
        model_options=["MHDModel"],
    )

    s4pi = np.sqrt(4 * np.pi)
    VL_prim = np.array([1.08, 1.2, 0.01, 0.5, 2./s4pi, 3.6/s4pi, 2./s4pi, 0.95])
    VR_prim = np.array([1.0, 0.0, 0.0, 0.0, 2./s4pi, 4.0/s4pi, 2./s4pi, 1.0])

    def get_state(x, y, z):
        x_prime = (x - L/2)*n[0] + (y)*n[1] + (z)*n[2]

        rho = np.where(x_prime < 0, VL_prim[0], VR_prim[0])
        p   = np.where(x_prime < 0, VL_prim[7], VR_prim[7])

        v_3d = np.zeros((3,) + x_prime.shape)
        b_3d = np.zeros((3,) + x_prime.shape)

        VL_rot = T @ VL_prim[1:4]
        VR_rot = T @ VR_prim[1:4]
        BL_rot = T @ VL_prim[4:7]
        BR_rot = T @ VR_prim[4:7]

        for i in range(3):
            v_3d[i] = np.where(x_prime < 0, VL_rot[i], VR_rot[i])
            b_3d[i] = np.where(x_prime < 0, BL_rot[i], BR_rot[i])

        return rho, v_3d, b_3d, p

    def density(x,y,z):
        return get_state(x,y,z)[0]

    def vx(x,y,z):
        return get_state(x,y,z)[1][0]

    def vy(x,y,z):
        return get_state(x,y,z)[1][1]

    def vz(x,y,z):
        return get_state(x,y,z)[1][2]

    def bx(x,y,z):
        return get_state(x,y,z)[2][0]

    def by(x,y,z):
        return get_state(x,y,z)[2][1]

    def bz(x,y,z):
        return get_state(x,y,z)[2][2]

    def p(x,y,z):
        return get_state(x,y,z)[3]

    ph.MHDModel(density=density, vx=vx, vy=vy, vz=vz, bx=bx, by=by, bz=bz, p=p)

    ph.ElectromagDiagnostics(quantity="B", write_timestamps=[0, sim.final_time])

    for quantity in ["rho", "V", "P"]:
        ph.MHDDiagnostics(quantity=quantity, write_timestamps=[0, sim.final_time])

    return sim

def main():
    Simulator(config()).run()

if __name__ == "__main__":
    startMPI()
    main()
