#!/usr/bin/env python3
"""
Merge two grid_convergence.py JSON outputs (one per scheme/branch) into the
headline showing: error-vs-resolution curves with the converged-Nx for each.

Usage:
  python compare.py convergence_o2.json convergence_o4.json
"""
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    return json.loads(open(path).read())


def crossing_nx(nx, err, tol):
    """Resolution where the error curve crosses tol (log-log interpolation)."""
    nx, err = np.asarray(nx, float), np.asarray(err, float)
    for i in range(len(err) - 1):
        if err[i] >= tol > err[i + 1]:
            t = (np.log(tol) - np.log(err[i])) / (np.log(err[i + 1]) - np.log(err[i]))
            return float(np.exp(np.log(nx[i]) + t * (np.log(nx[i + 1]) - np.log(nx[i]))))
    if err[-1] < tol:
        return float(nx[0])
    return None


def main(paths):
    runs = [load(p) for p in paths]
    tol = runs[0]["tol"]

    plt.figure(figsize=(9, 6))
    summary = []
    for r in runs:
        nx, err = r["nx"], r["rel_l1_err"]
        plt.loglog(nx, err, "o-", label=r["label"])
        ncross = crossing_nx(nx, err, tol)
        summary.append((r["label"], ncross))
        if ncross:
            plt.axvline(ncross, ls=":", alpha=0.5)

    plt.axhline(tol, color="k", ls="--", lw=1, label=f"tol = {tol:g}")
    plt.xlabel("base resolution Nx", fontsize=14)
    plt.ylabel(f"relative L1 error in {runs[0]['qty']} (vs finest)", fontsize=14)
    plt.title("Orszag-Tang @ t=1: resolution needed for convergence", fontsize=15)
    plt.grid(True, which="both", ls="--", lw=0.4)
    plt.legend(fontsize=12)
    plt.savefig("orszag_convergence_compare.png", dpi=200, bbox_inches="tight")
    print("wrote orszag_convergence_compare.png\n")

    for label, n in summary:
        print(f"  {label:>14}: converged Nx = {n}" if n else
              f"  {label:>14}: NOT converged in sampled range")
    if len(summary) == 2 and all(n for _, n in summary):
        (_, n0), (_, n1) = summary
        print(f"\n  ratio = {max(n0, n1) / min(n0, n1):.2f}x coarser "
              f"(=> ~{(max(n0, n1) / min(n0, n1))**3:.1f}x cheaper in 2D+CFL)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: compare.py <json_o2> <json_o4>")
    main(sys.argv[1:])
