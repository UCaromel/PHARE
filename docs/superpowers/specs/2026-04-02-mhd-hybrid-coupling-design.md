# MHD–Hybrid Coupling Design

**Date:** 2026-04-02  
**Branch context:** `3d-mhd` / `high-order-deriv`  
**Status:** Design approved, pending implementation

---

## Overview

This document specifies the design for coupling the MHD (coarse) and Hybrid PIC (fine) models in PHARE's AMR hierarchy. MHD always occupies the coarser levels; Hybrid always occupies the finer levels on top of MHD. The coupling is fully conservative: mass, momentum, magnetic field, and total energy are all conserved across the interface.

---

## Level Hierarchy

```
Level 0 .. N_mhd    →  MHDModel    +  SolverMHD
Level N_mhd+1 .. M  →  HybridModel +  SolverPPC
```

`MultiPhysicsIntegrator` and `MessengerFactory` already handle this correctly — no changes needed to either. The factory creates:
- `MHDMessenger` for MHD-MHD pairs
- `HybridMessenger<MHDHybridMessengerStrategy>` at the interface level (N_mhd+1)
- `HybridMessenger<HybridHybridMessengerStrategy>` for all finer Hybrid-Hybrid pairs

---

## Per-Coarse-Step Data Flow

```
MHD is at t^n
  └─ MHDHybridMessengerStrategy::firstStep()
       - snapshot MHD EM into EM_old_ (B and E at t^n)
       - fill Hybrid ghost cells: B via existing MagneticFieldRefinOp
                                  E via existing MHD E refine op (static, t^n snapshot)
  └─ SolverPPC subcycles (ratio² substeps)
       ├─ MHDHybridMessengerStrategy::prepareStep(α)
       │    - time-interpolate B ghost cells: B = (1-α)*EM_old_.B + α*MHD.B
       │    - E ghost cells remain at static t^n snapshot throughout
       ├─ SolverPPC::advanceLevel()
       └─ SolverPPC::accumulateFluxSum()
            - accumulate fluxSumRho_, fluxSumRhoV_, fluxSumEtot_, fluxSumE_
  └─ standardLevelSynchronization()
       ├─ messenger.synchronize()        coarsen Hybrid moments → MHD conserved vars
       ├─ messenger.reflux()             coarsen Hybrid flux sums → MHD face registers
       ├─ solver.reflux()               retake MHD coarse step with corrected fluxes
       └─ messenger.postSynchronize()   refill MHD ghosts from corrected state
```

---

## Component 1: MHDHybridMessengerStrategy

**File:** `src/amr/messengers/mhd_hybrid_messenger_strategy.hpp` (implement existing stub)

### `registerQuantities(fromCoarserInfo, fromFinerInfo)`

Sets up SAMRAI algorithms (schedules created per-level in `registerLevel()`):

**Refine algorithms (MHD coarser → Hybrid finer):**
- B: reuse existing `MagneticFieldRefinOp` (face-centered staggering identical between MHD and Hybrid)
- E: reuse existing MHD E refine operator (edge-centered Yee grid identical between MHD and Hybrid)
- rho, V: standard cell-centered linear interpolation (for particle injection context data)
- `EM_old_` snapshot of B and E for `firstStep()` storage

**Coarsen algorithms (Hybrid finer → MHD coarser):**
- Moments (ρ, ρV, reconstructed Etot, B) → MHD conserved: `HybridMHDMomentCoarsenOp`
- Flux sums (fluxSumRho_, fluxSumRhoV_, fluxSumEtot_, fluxSumE_): `HybridFluxCoarsenOp`

### `registerLevel(hierarchy, levelNumber)`

Creates `RefineSchedule` and `CoarsenSchedule` instances per level from the algorithms above.

### `firstStep()`

Snapshots current MHD B and E into `EM_old_`. Fills Hybrid ghost cells with static refinement of B and E from the MHD t^n state. E ghost cells are not updated again during subcycling (E is a UCT temporary in MHD, not stored persistently — static refinement is the correct treatment).

### `lastStep()`

No-op.

### `prepareStep(α)`

Time-interpolates B ghost cells on the Hybrid level for substep fraction `α = t_substep / dt_coarse`:

```
B_ghost = (1-α) * EM_old_.B  +  α * MHD.B
```

E ghost cells remain at the static t^n snapshot set in `firstStep()`.

### `initLevel()` and `regrid()`

1. Fill B and E on new Hybrid patches from MHD coarser (via refine schedules above)
2. Invoke `MHDHybridParticleInjectionPatchStrategy` to populate particles on each new patch

For `regrid()`: SAMRAI restricts `fineBox` in `postprocessRefine()` to truly new regions — particle injection is automatically scoped to new regions only.

### `fillMagneticGhosts()` / `fillElectricGhosts()`

Delegate to the corresponding refine schedules.

### `synchronize()`

Coarsens Hybrid state onto MHD at end of all substeps:

1. Coarsen Hybrid face B → MHD B (existing coarsen op, same staggering)
2. Coarsen Hybrid ion density → MHD rho
3. Coarsen Hybrid ρV (ion momentum moment) → MHD rhoV
4. Per patch: compute `P_i_scalar = Tr(ion_pressure_tensor) / 3`
5. Per patch: compute `Etot = ½ρV² + (P_i_scalar + P_e) / (γ-1) + B²/(2μ₀)`  
   where `P_e` comes from the Hall MHD electron pressure (isothermal initially)
6. Coarsen Etot → MHD Etot
7. Call existing `ToPrimitiveConverter` on MHD level to re-derive V and P from updated rho, rhoV, Etot, B

### `reflux(coarseLevelNumber, fineLevelNumber, syncTime)`

Follows the existing global-level reflux procedure (not classical per-patch boundary correction):

1. Coarsen Hybrid `fluxSumRho_`, `fluxSumRhoV_`, `fluxSumEtot_`, `fluxSumE_` globally from the whole fine level onto MHD face flux registers using `HybridFluxCoarsenOp`
2. Delegate to `SolverMHD::reflux()` on the coarse level — retakes the coarse step with corrected fluxes via existing `EulerUsingComputedFlux` mechanism
3. B correction from E applied via Faraday (same path as Hybrid-Hybrid E-based B reflux)

### `postSynchronize()`

Refills MHD patch ghosts from the corrected state using existing MHD refine schedules.

---

## Component 2: MHDHybridParticleInjectionPatchStrategy

**File:** `src/amr/messengers/mhd_hybrid_particle_injection_patch_strategy.hpp` (new)

Extends `SAMRAI::xfer::RefinePatchStrategy`. Constructed inside `MHDHybridMessengerStrategy`, passed to the `RefineAlgorithm` used for `initLevel()` and `regrid()`.

### `postprocessRefine(finePatch, coarsePatch, fineBox, ratio)`

Called by SAMRAI after standard field data filling (ensuring B is already available for field-aligned Maxwellian basis if requested).

1. Read MHD primitives from `coarsePatch`: rho, V, P, Pe
2. Compute `P_i = P - P_e` on coarse patch
3. Build callables wrapping coarse patch data linearly interpolated to fine resolution (1D/2D/3D consistent):
   - `density(x)` → bilinear interpolation of coarse rho
   - `bulkVelocity[3](x)` → bilinear interpolation of coarse V
   - `thermalVelocity[3](x)` → `sqrt(P_i(x) / rho(x))` (isotropic, uniform in x/y/z)
4. Construct `MaxwellianParticleInitializer` from these callables with population charge, nbrParticlesPerCell, seed, basis
5. Call `loadParticles()` restricted to `fineBox`
6. Apply `densityCutOff` to avoid particle injection in vacuum regions

### `preprocessRefine()`

No-op.

### Seeding

Deterministic seed per patch: `patch.getGlobalId() + population_index + level_number`. Ensures reproducibility across MPI ranks and restarts.

---

## Component 3: SolverPPC Flux Accumulation

**File:** `src/amr/solvers/solver_ppc.hpp` (modified)

### New members

```cpp
Field    fluxSumRho_;    // face-centered, accumulated density flux  ρV·n̂
VecField fluxSumRhoV_;   // face-centered, accumulated momentum flux (ρVV + P_i_scalar)·n̂
Field    fluxSumEtot_;   // face-centered, accumulated total energy flux
// existing:
VecField fluxSumE_;      // accumulated E for Faraday/B reflux
Field    Bold_;
```

All registered and allocated via `ResourcesManager` following the existing pattern.

### Face flux computation (per substep, per patch)

After moment deposition, cell-centered ρ, J=ρV, ion pressure tensor, E, B are available.

**Density flux** (face-averaged):
```
F_rho(face) = avg(ρV·n̂, left cell, right cell)
```

**Momentum flux** (isotropic pressure approximation, consistent with MHD scalar P closure):
```
F_rhoV(face) = avg(ρV·V_n + P_i_scalar, left cell, right cell)
  where P_i_scalar = Tr(ion_pressure_tensor) / 3
```

**Total energy flux** (Poynting + enthalpy):
```
F_Etot(face) = avg((E×B/μ₀)·n̂, left cell, right cell)
             + avg((½ρV² + γ(P_i_scalar + P_e)/(γ-1)) * V·n̂, left cell, right cell)
```

The Poynting term reuses E already available from the Ohm solve, consistent with E-based B reflux.

### `accumulateFluxSum(dt_substep, dt_coarse)`

```cpp
fluxSumRho_  += dt_substep * F_rho
fluxSumRhoV_ += dt_substep * F_rhoV
fluxSumEtot_ += dt_substep * F_Etot
fluxSumE_    += dt_substep * E        // unchanged
```

### `resetFluxSum()`

Zeros all four accumulators at start of first substep. Unchanged pattern.

### `fillMessengerInfo()` update

Adds `fluxSumRho_`, `fluxSumRhoV_`, `fluxSumEtot_` field names to `HybridMessengerInfo` so that `MHDHybridMessengerStrategy::registerQuantities()` can set up the corresponding coarsen algorithms.

---

## Component 4: New Coarsen Operators

### `HybridMHDMomentCoarsenOp`

**File:** `src/amr/data/coarsen/hybrid_mhd_moment_coarsen_op.hpp` (new)

Coarsens cell-centered Hybrid quantities onto MHD via volume-weighted averaging:
- ion density → rho
- ρV (ion momentum moment) → rhoV
- reconstructed Etot (computed in `synchronize()` before coarsening) → Etot
- Face B reuses existing coarsen operator (same staggering)

### `HybridFluxCoarsenOp`

**File:** `src/amr/data/coarsen/hybrid_flux_coarsen_op.hpp` (new)

Coarsens face-centered Hybrid flux sums onto MHD face flux registers via area-weighted averaging. Same logic as existing `MHDFluxCoarsener`. Applied globally to the whole fine level (not per-patch at boundary) — consistent with PHARE's current global reflux procedure.

---

## Field Staggering Summary

| Quantity | MHD | Hybrid | Operator needed |
|---|---|---|---|
| B | face-centered (Yee) | face-centered (Yee) | existing `MagneticFieldRefinOp` |
| E | edge-centered (Yee) | edge-centered (Yee) | existing MHD E refine op |
| rho, V, P | cell-centered | cell-centered (moments) | standard cell linear interp |
| Flux sums | face-centered | face-centered | `HybridFluxCoarsenOp` (new) |

No new refine operators are needed — all field staggerings are compatible between MHD and Hybrid.

---

## Files Changed

### New files
| File | Purpose |
|---|---|
| `src/amr/messengers/mhd_hybrid_particle_injection_patch_strategy.hpp` | Particle injection via `RefinePatchStrategy` |
| `src/amr/data/coarsen/hybrid_mhd_moment_coarsen_op.hpp` | Coarsen Hybrid moments → MHD conserved vars |
| `src/amr/data/coarsen/hybrid_flux_coarsen_op.hpp` | Coarsen Hybrid face flux sums → MHD |

### Modified files
| File | Changes |
|---|---|
| `src/amr/messengers/mhd_hybrid_messenger_strategy.hpp` | Implement all stubbed methods |
| `src/amr/solvers/solver_ppc.hpp` | Add flux accumulators, update accumulate/reset/reflux |
| `src/amr/messengers/hybrid_messenger_info.hpp` | Add flux sum field names for ρ, ρV, Etot |

### Unchanged
`MultiPhysicsIntegrator`, `MessengerFactory`, all MHD solver/messenger code, `MaxwellianParticleInitializer`, all existing converters, `MagneticRefinePatchStrategy`, all existing refine/coarsen operators.

---

## Open Questions / Future Work

- **E ghost cells during subcycling:** Static t^n refinement is used. This may warrant revisiting once Hall MHD electron pressure dynamics are more developed — a time-interpolated E could be explored if static refinement proves insufficient at the interface.
- **Reflux formulation:** Current implementation uses global-level flux accumulation + coarsening (not classical per-patch boundary reflux). If a more classical formulation is adopted for MHD-MHD in the future, the Hybrid→MHD reflux should be updated to match.
- **Multiple ion species:** This design initializes a single Maxwellian ion population from MHD fluid data. Multi-species initialization (e.g. core + beam from a single MHD fluid) is deferred.
- **Pressure tensor in momentum flux:** Off-diagonal terms of the ion pressure tensor are dropped in `F_rhoV` (isotropic approximation). This is consistent with MHD's scalar pressure closure but may be revisited for more accurate momentum conservation.
- **B_old access for time interpolation:** `prepareStep()` time-interpolates B ghost cells between t^n and t^n+1. The t^n B snapshot lives in `SolverMHD::stateOld_`, but the messenger does not currently hold a reference to the solver. The access mechanism (e.g. exposing B_old on MHDModel, or passing it through MHDMessengerInfo) needs to be resolved during implementation.
- **Validation:** A 1D shock or Alfvén wave crossing the MHD-Hybrid interface will be needed to validate conservation properties in practice.
