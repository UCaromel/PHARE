# Reflux Corner DivB Investigation

## Problem
DivB errors appear at patch boundary intersections (corners) after reflux correction.
Image shows pairs of +/- dots at every patch-patch and patch-periodic intersection.

## Test Setup
- harris.py: 100x100 coarse cells, 2 levels (L0+L1), dl=0.4, 4 MPI ranks
- Single timestep (dt=0.005), L1 substeps at dt=0.00125 (ratio=4 time substeps)
- L1 covers two horizontal strips (current sheet regions):
  - Lower strip: fine y=[32,63] (coarse y=[16,31])
  - Upper strip: fine y=[128,159] (coarse y=[64,79])
- Each strip split into 2 patches in x: fine x=[0,95] and x=[96,199]
- CF boundaries: y-direction only (locations 2 and 3), 2 per L1 patch

## Hypotheses
1. ~~Double accumulation of shared E components at corner cells~~ **RULED OUT**
2. Coarsener reads flux sum ghost cells (unfilled) at L1 patch boundaries
3. Missing accumulation at some corner cells
4. Coarsen schedule ghost-fill overwrites correct values on L0
5. Issue in reflux application step (wrong indexing, wrong sign, etc.)

---

## Round 1: Accumulation count check

**Goal:** Determine if any E component cell gets accumulated != 1 time.

**Method:** Added `std::map<string, int>` counter in `accumulateFluxSum` keyed by
(component, readIdx). After boundary loop, dump cells with count != 1.

**Result:** Zero cells with count != 1. Every E component is accumulated exactly once.

**Conclusion:** Double accumulation is NOT the issue. The corner-skip fix was addressing
a non-existent problem. The bug is downstream of accumulation — either in coarsening
or in the reflux application.

**L1 patch layout confirmed:**
- Patch 0: fine (96,128)-(199,159), 2 boundaries (y-direction)
- Patch 1: fine (0,128)-(95,159), 2 boundaries (y-direction)  
- Patch 2: fine (96,32)-(199,63), 2 boundaries (y-direction)
- Patch 3: fine (0,32)-(95,63), 2 boundaries (y-direction)

**Key observation:** CF boundaries are y-direction ONLY. No x-direction CF boundaries.
The corner errors appear where y-CF-boundary meets x-patch-boundary (at coarse x=48).

---

## Round 2: Reflux dE values at CF boundary

**Goal:** Dump fluxSumE (coarsened from L1) and timeElectric (coarse level) at every
CF boundary cell in the reflux correction. Look for anomalous dE at patch boundary
positions (coarse x=0, x=48, x=99) vs interior cells.

**Method:** Per-rank file dump in `reflux()` to `/tmp/.log/reflux_r{rank}.txt`.
Each line: location, dir, coarseIdx, fluxCoarseIdx, fluxSumE, timeE, dE, dB.

**Instrumentation:** Active in solver_mhd.hpp reflux function. File per MPI rank.

**Result:** ROOT CAUSE FOUND — two compounding bugs.

### Finding 1: cfFaceBoundaries are rank-local (missing cross-rank corrections)

Raw boundary boxes from the dump:
```
rank 0: bb[0] loc=2 box=[(95,127),(200,127)]   # from fine patch 0 only
rank 1: bb[0] loc=2 box=[(-1,127),(96,127)]    # from fine patch 1 only
rank 2: bb[0] loc=2 box=[(95,31),(200,31)]     # from fine patch 2 only
rank 3: bb[0] loc=2 box=[(-1,31),(96,31)]      # from fine patch 3 only
```

Each rank has only 2 boundary boxes (its own fine patches). The integrator at
`multiphysics_integrator.hpp:597-601` collects `cfFaceBdry` by iterating
**local** fine patches only (`for (auto& finePatch : fineLevel)`).

Consequence: rank 1's coarse patch covers x=0..49, but its boundary boxes only
reach coarse x=48 (fine x=96 → coarse 48). Coarse x=49 (from fine x=98,99 on
rank 0's fine patch) gets **no reflux correction at all** — hydro or magnetic.

### Finding 2: fsE.Ez = 0 at periodic boundary (x=0)

All ranks show `fsE.Ez = 0` at their periodic-edge cell (coarse x=0 on ranks 1,3).
The boundary box from fine patch 1 extends to fine x=-1..96. Fine x=-1 maps to
coarse x=-1 (outside domain); fine x=0 maps to coarse x=0 which IS processed.
But the coarsened fluxSumEz at coarse x=0 is zero despite the fine accumulation
being correct.

Root cause: the messenger's CoarsenAlgorithm for fluxSumE doesn't propagate
edge-centered values at periodic patch boundaries. Ez (ppd centering, primal in
x and y) has one more node than cells — the periodic wrap node at x=N is the
same physical point as x=0 but the coarsener misses it.

### Summary of root causes

1. **Rank-local cfFaceBoundaries** — `multiphysics_integrator.hpp` builds the
   list from local fine patches only. Any coarse cell whose CF data comes from
   a remote fine patch gets no correction.

2. **Edge centering mismatch** — the plan specifies "CF boundary edges" for E
   but implementation uses face boundary boxes (codimension 1). Ez is primal in
   both transverse directions → face boundary misses the last primal node.

Both cause divB errors at patch boundary intersections.

### Fix strategy

1. Use SAMRAI's `CoarseFineBoundary` with `do_all_patches=true` (already the
   case internally) and iterate over ALL fine patches globally, not just local.
   The `getGlobalizedVersion()` of the fine BoxLevel gives access to remote
   patch IDs; `getBoundaries(globalId, 1)` works for any patch.

2. For the magnetic E correction, extend the boundary iteration by +1 in the
   primal transverse direction to capture the edge node that the face boundary
   box misses.

---

## Instrumentation currently in code
1. `accumulateFluxSum`: entry trace (ACCUM-ENTRY, ACCUM-PATCH) + count map on stderr
2. `reflux`: per-rank file dump + raw boundary box extents to `/tmp/.log/reflux_r{rank}.txt`
3. Corner-skip fix for Ez at y-boundaries: REVERTED (was addressing non-issue)
4. Periodic face copy: REMOVED (was based on wrong diagnosis)

---

## Session 2026-04-27: SolverMHD::reflux() not called

Attempted to instrument probes in `SolverMHD::reflux()` to verify the stale-advance
hypothesis (Bug 2 in branch_state). Found that `reflux()` is never called at all
despite a 2-level hierarchy being confirmed in HDF5 output (pl0 and pl1 present at
t=0 and t=0.005 with 4 fine patches each).

**Resolution:** Output was going to `.log/<rank>.out` (PHARE redirects all stdout/stderr
there). Probes DID fire. Results from `.log/*.out`:

```
[PRE-REFLUX]  Bx(1,80)=-1.01033   [POST-REFLUX] Bx(1,80)=-1.01023   Δ=+1.0e-4 ← CHANGES
[PRE-REFLUX]  Bx(2,80)=-1.01027   [POST-REFLUX] Bx(2,80)=-1.01017   Δ=+1.0e-4 ← CHANGES
[PRE-REFLUX]  Bx(1,82)=-1.01422   [POST-REFLUX] Bx(1,82)=-1.01422   Δ=0       ← LOCKED
[PRE-REFLUX]  Bx(2,82)=-1.01414   [POST-REFLUX] Bx(2,82)=-1.01414   Δ=0       ← LOCKED
```

**Stale-advance hypothesis CONFIRMED:**
- Reflux corrects Bx at y=80 (CF-adjacent row) by ~1e-4
- Bx at y=82 (two rows above CF boundary) is never touched by reflux — it was committed
  during the coarse advance using the PRE-reflux value of Bx(y=80)
- The ~1e-4 Bx error at y=82 is exactly the magnitude of the reflux correction at y=80

**Probe gotchas (for future reference):**
- PHARE routes stdout AND stderr to `.log/<rank>.out` — never appears in terminal
- `AMRToLocal` takes cell (dual) coords; gate patch selection on BOTH x and y range
  before calling, or `assert(local >= 0)` fires for patches not containing the probe cell
