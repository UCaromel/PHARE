#!/usr/bin/env python3
"""Composite L2-of-L1 error (t=1 vs t=0 of the same run) for conv3.py runs, with a
3-level covered-cell mask. usage: err3.py <diag dir> <N> <levels> <margin>"""

import os
import sys
import numpy as np

from pyphare.pharesee.run import Run
from tests.simulator.amr_convergence.compute_errors import CONSERVED, conserved_fields

d, N, levels, margin = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
lo, hi = N // 4, 3 * N // 4 - 1
covered = {}  # level -> box (in that level's index space) covered by the next finer level
if levels >= 2:
    covered[0] = ((lo, lo), (hi, hi))
if levels == 3:
    l1lo, l1hi = 2 * lo + margin, 2 * hi + 1 - margin
    covered[1] = ((l1lo, l1lo), (l1hi, l1hi))


def mask(key):
    ilvl, lower, upper = key
    m = np.ones(tuple(u - l + 1 for l, u in zip(lower, upper)))
    if ilvl in covered:
        (clo, chi) = covered[ilvl][0], covered[ilvl][1]
        sl = []
        for k in range(2):
            a, b = max(clo[k], lower[k]) - lower[k], min(chi[k], upper[k]) - lower[k]
            if b < a:
                return m
            sl.append(slice(a, b + 1))
        m[tuple(sl)] = 0.0
    return m


run = Run(d)
t1 = float(os.environ.get("CV_T1", "1.0"))  # final dump time
f1, f0 = conserved_fields(run, t1), conserved_fields(run, 0.0)
num = {v: 0.0 for v in CONSERVED}
den = 0.0
lnum, lden = {}, {}
for key in f1:
    il = key[0]
    m = mask(key)
    w = 4.0 ** (-il)
    ln = lnum.setdefault(il, {v: 0.0 for v in CONSERVED})
    for v in CONSERVED:
        diff = np.abs(f1[key][v] - f0[key][v])
        num[v] += (diff * m).sum() * w
        ln[v] += (diff * m).sum()
    den += m.sum() * w
    lden[il] = lden.get(il, 0) + m.sum()


def eps(n, dd):
    return float(np.sqrt(sum((n[v] / dd) ** 2 for v in CONSERVED)))


per = " ".join(f"L{il}={eps(lnum[il], lden[il]):.6e}" for il in sorted(lnum))
print(f"EPS N={N} levels={levels} margin={margin} composite={eps(num, den):.6e} {per}")
