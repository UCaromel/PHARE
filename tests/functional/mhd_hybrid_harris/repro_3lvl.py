#!/usr/bin/env python3
"""3-level coupled MHD-Hybrid repro. Levels via env:
   REPRO_MHD_LEVEL (max_mhd_level), REPRO_NBR (max_nbr_levels).
Runs the simulator only — no post-run diag asserts."""
import os
import importlib.util

import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator, startMPI

# load the sibling test module to reuse its config()
HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location(
    "_base", os.path.join(HERE, "mhd_hybrid_harris_2d.py")
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

MHD_LEVEL = int(os.environ.get("REPRO_MHD_LEVEL", "2"))
NBR = int(os.environ.get("REPRO_NBR", "3"))
NEST = os.environ.get("REPRO_NEST")  # override nesting_buffer if set
FINAL_TIME = os.environ.get("REPRO_FINAL_TIME")  # shorten run for init+few-steps gates


def main():
    ph.global_vars.sim = None
    sim = base.config()
    print(f"[repro] requested config max_mhd_level={sim.max_mhd_level} "
          f"max_nbr_levels={sim.max_nbr_levels}", flush=True)
    # override levels
    sim.max_mhd_level = MHD_LEVEL
    sim.max_nbr_levels = NBR
    if NEST is not None:
        import numpy as _np
        sim.nesting_buffer = _np.asarray([int(NEST)] * len(sim.cells))
    if FINAL_TIME is not None:
        # time_step_nbr is derived from final_time at Simulation construction —
        # override both or the run loop keeps its original length
        sim.final_time = float(FINAL_TIME)
        sim.time_step_nbr = int(sim.final_time / sim.time_step)
        print(f"[repro] OVERRIDDEN final_time={sim.final_time} "
              f"time_step_nbr={sim.time_step_nbr}", flush=True)
    print(f"[repro] OVERRIDDEN max_mhd_level={sim.max_mhd_level} "
          f"max_nbr_levels={sim.max_nbr_levels} nesting_buffer={sim.nesting_buffer}", flush=True)
    print("[repro] building simulator...", flush=True)
    s = Simulator(sim)
    print("[repro] initialize...", flush=True)
    s.initialize()
    print("[repro] init done, running...", flush=True)
    s.run()
    print("[repro] RUN DONE OK", flush=True)


if __name__ == "__main__":
    startMPI()
    main()
