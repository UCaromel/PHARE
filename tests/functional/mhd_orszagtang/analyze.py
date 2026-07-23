#!/usr/bin/env python3
"""
Common-truth grid-convergence analysis for Orszag-Tang.

Loads all npz field dumps (fields_<label>_<nx>.npz) produced by run_one.py,
uses the HIGHEST-resolution run as the shared truth, scores EVERY coarser run
(both schemes) against that one reference -> answers "which scheme is closer to
the true solution at a given Nx, and how many points does each need".

Usage:
  python analyze.py <npz_dir> [tol]
"""
import sys
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT = re.compile(r"fields_(.+)_(\d+)\.npz$")


def restrict(fine, ratio):
    nx, ny = fine.shape[0] // ratio, fine.shape[1] // ratio
    return fine.reshape(nx, ratio, ny, ratio).mean(axis=(1, 3))


def crossing_nx(nx, err, tol):
    nx, err = np.asarray(nx, float), np.asarray(err, float)
    order = np.argsort(nx)
    nx, err = nx[order], err[order]
    for i in range(len(err) - 1):
        if err[i] >= tol > err[i + 1]:
            t = (np.log(tol) - np.log(err[i])) / (np.log(err[i + 1]) - np.log(err[i]))
            return float(np.exp(np.log(nx[i]) + t * (np.log(nx[i + 1]) - np.log(nx[i]))))
    if err[-1] < tol:
        return float(nx[0])
    return None


def main(npz_dir, tol=1e-2):
    files = {}
    for p in Path(npz_dir).glob("fields_*.npz"):
        m = PAT.search(p.name)
        if m:
            files[(m.group(1), int(m.group(2)))] = p

    if not files:
        sys.exit(f"no fields_*.npz in {npz_dir}")

    truth_nx = max(nx for _, nx in files)
    truth_key = next(k for k in files if k[1] == truth_nx)
    truth = np.load(files[truth_key])["P"]
    print(f"truth = {truth_key[0]} @ {truth_nx}^2 (shape {truth.shape})\n")

    curves = {}
    for (label, nx), p in sorted(files.items(), key=lambda kv: kv[0][1]):
        if nx == truth_nx:
            continue
        P = np.load(p)["P"]
        ref = restrict(truth, truth_nx // nx)
        e = float(np.sum(np.abs(P - ref)) / np.sum(np.abs(ref)))
        curves.setdefault(label, []).append((nx, e))

    plt.figure(figsize=(9, 6))
    summary = []
    for label, pts in sorted(curves.items()):
        pts.sort()
        nxs = [n for n, _ in pts]
        errs = [e for _, e in pts]
        plt.loglog(nxs, errs, "o-", label=label)
        print(f"  {label}:")
        for n, e in pts:
            print(f"      Nx={n:>5}:  rel L1 = {e:.4e}")
        ncross = crossing_nx(nxs, errs, tol)
        summary.append((label, ncross))

    plt.axhline(tol, color="k", ls="--", lw=1, label=f"tol = {tol:g}")
    plt.xlabel("base resolution Nx", fontsize=14)
    plt.ylabel(f"relative L1 error in P (vs {truth_nx}$^2$ truth)", fontsize=14)
    plt.title("Orszag-Tang @ t=1: error vs resolution, common reference", fontsize=14)
    plt.grid(True, which="both", ls="--", lw=0.4)
    plt.legend(fontsize=12)
    plt.savefig("orszag_common_truth.png", dpi=200, bbox_inches="tight")
    print("\nwrote orszag_common_truth.png")

    for label, n in summary:
        print(f"  {label:>6}: converged Nx (tol {tol:g}) = "
              f"{n:.0f}" if n else f"  {label:>6}: not converged in range")
    if len(summary) == 2 and all(n for _, n in summary):
        (l0, n0), (l1, n1) = summary
        print(f"\n  {max(n0,n1)/min(n0,n1):.2f}x coarser grid for the same accuracy")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: analyze.py <npz_dir> [tol]")
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-2
    main(sys.argv[1], tol)
