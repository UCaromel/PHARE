#ifndef PHARE_REFLUX_GEOMETRY_HPP
#define PHARE_REFLUX_GEOMETRY_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/BoxContainer.h>
#include <SAMRAI/hier/CoarseFineBoundary.h>
#include <SAMRAI/hier/GlobalId.h>
#include <SAMRAI/hier/Index.h>

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield_component.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "core/utilities/constants.hpp"
#include "core/utilities/point/point.hpp"

// Dimension-generic geometry primitives for Berger-Colella reflux synchronization.
//
// These extract the Yee-grid coarse-fine boundary bookkeeping that the MHD solver's
// accumulateFluxSum (fine-side) and reflux (coarse-side) functions need. The single
// root cause of the historical mess is that SAMRAI's CoarseFineBoundary hands boxes in
// FACE space (codim-1) / EDGE space (codim-2), while E is edge-centered (3D) /
// Ez node-centered (2D) and B-correction targets faces anchored on cells.
//
// Ownership / dedup is expressed as box-level set algebra (BoxContainer::simplify, which
// produces a canonical non-overlapping cover) rather than per-index std::set membership.
namespace PHARE::solver::reflux_geometry
{
// ---- P2 — Yee Faraday curl map (unified) ----
//
// For a CF boundary with the given normal direction, the two transverse B faces that the
// boundary E correction touches, with the E component read and the curl sign. Replaces the
// hand-written per-dimension B-correction dispatch. The 1D solver only ever has normal
// direction X, so the dirX terms (By/Bz) cover it.
//   dirX -> {By,Y,eZ,+1}, {Bz,Z,eY,-1}
//   dirY -> {Bx,X,eZ,-1}, {Bz,Z,eX,+1}
//   dirZ -> {Bx,X,eY,+1}, {By,Y,eX,-1}
// bQty's quantity enum is a template parameter because GridLayout::centering dispatches on
// the model's quantity type: MHD instantiates with MHDQuantity::Scalar, Hybrid with
// HybridQuantity::Scalar. The Yee B-face centering pattern is identical across models, so
// only the enum type differs. Callers must name BQ explicitly (no default).
template<typename BQ>
struct FaradayTerm
{
    BQ              bQty;
    core::Component bComp;
    core::Component eComp;
    double          eSign;
};

template<typename BQ>
inline std::array<FaradayTerm<BQ>, 2> faradayTerms(int normalDir)
{
    using Q = BQ;
    using C = core::Component;
    switch (normalDir)
    {
        case static_cast<int>(core::dirX):
            return {{{Q::By, C::Y, C::Z, +1.0}, {Q::Bz, C::Z, C::Y, -1.0}}};
        case static_cast<int>(core::dirY):
            return {{{Q::Bx, C::X, C::Z, -1.0}, {Q::Bz, C::Z, C::X, +1.0}}};
        default: // core::dirZ
            return {{{Q::Bx, C::X, C::Y, +1.0}, {Q::By, C::Y, C::X, -1.0}}};
    }
}

// ---- P1 — fine-side CF electric-edge enumeration (specialized 2D/3D, A-ready) ----
//
// Read-shifted, box-deduped containers of AMR indices at which each E component must be
// accumulated into fluxSumE_. Each box is already shifted to the boundary-flux read
// coordinate, so the caller iterates phare_box_from over it and accumulates directly.
template<std::size_t dimension>
struct EAccumBoxes
{
    SAMRAI::hier::BoxContainer ex, ey, ez;
};

// Yields each Yee E-edge on a patch's CF boundary ONCE, per E-component.
//
// Specialized per dimension behind one interface: the node<->edge degeneracy of Ez is the
// only irreducible dim-divergence. The 2D path ports the codim-1 component assignment plus
// the Ez primal-endpoint patching (nodes SAMRAI clips off the codim-1 boxes) verbatim; the
// 3D path ports the codim-2 edge loop. Cross-patch shared E-nodes are NOT handled here —
// SAMRAI coarsening of fluxSumE_ owns them (intra-patch box-dedup suffices).
template<std::size_t dimension>
EAccumBoxes<dimension>
cfElectricBoxes(SAMRAI::hier::CoarseFineBoundary const& cfBoundary,
                SAMRAI::hier::GlobalId const& patchId,
                SAMRAI::hier::Box const& patchCellBox)
{
    using SAMRAI::hier::Box;
    using SAMRAI::hier::Index;

    EAccumBoxes<dimension> out;
    auto const dim = patchCellBox.getDim();

    // Shift a SAMRAI boundary box to the boundary-flux read coordinate (lower boundaries
    // read the fine flux one cell inward) and clip its transverse extent to the patch cell
    // box (the inPatchTransverse filter expressed as box algebra). Returns an empty box if
    // the transverse clip is empty.
    auto const shiftedClipped = [&](Box bb, int normalDir, bool isLower) {
        Index lo = bb.lower(), hi = bb.upper();
        if (isLower)
        {
            lo(normalDir) += 1;
            hi(normalDir) += 1;
        }
        for (int d = 0; d < static_cast<int>(dimension); ++d)
        {
            if (d == normalDir) continue;
            lo(d) = std::max(lo(d), patchCellBox.lower(d));
            hi(d) = std::min(hi(d), patchCellBox.upper(d));
        }
        return Box(lo, hi, bb.getBlockId());
    };

    auto const push = [](SAMRAI::hier::BoxContainer& c, Box const& b) {
        if (!b.empty()) c.push_back(b);
    };

    if constexpr (dimension == 1)
    {
        // 1D: codim-1 is a node — E and fluxes share the same boundary type.
        for (auto const& bb : cfBoundary.getBoundaries(patchId, 1))
        {
            bool const isLower = (bb.getLocationIndex() % 2 == 0);
            auto const box     = shiftedClipped(bb.getBox(), core::dirX, isLower);
            push(out.ey, box);
            push(out.ez, box);
        }
    }
    else if constexpr (dimension == 2)
    {
        // 2D codim-1: hydro E-field accumulation at coarse-fine boundaries.
        for (auto const& bb : cfBoundary.getBoundaries(patchId, 1))
        {
            auto const location = bb.getLocationIndex();
            bool const isLower  = (location % 2 == 0);
            int const normalDir = location / 2;

            auto const box = shiftedClipped(bb.getBox(), normalDir, isLower);

            if (normalDir == core::dirX)
            {
                // Ey: primal in x (own direction), dual in y (transverse).
                push(out.ey, box);
                // Ez: primal in x and y.
                push(out.ez, box);
            }
            else // normalDir == core::dirY
            {
                // Ex: primal in y (own direction), dual in x (transverse).
                push(out.ex, box);
                // Ez: primal in x and y.
                push(out.ez, box);
            }

            // --- Explicit primal endpoint accumulation (1-cell boxes) ---
            // SAMRAI clips the transverse extent of a codim-1 box to the patch CC box when
            // a CF boundary is present on the adjacent side. The main loop above reaches
            // bb.upper(transverse) but not bb.upper(transverse)+1. These primal Ez nodes
            // are covered by reflux() via makeComponentBox (+1 extension) and must be
            // filled here. Box-dedup (simplify) over out.ez folds them in once.
            auto const& rawBox = bb.getBox();
            if (normalDir == core::dirY)
            {
                // Rightmost primal-x node: always at patchCellBox.upper(x)+1.
                Index pidx(dim);
                pidx(core::dirX) = patchCellBox.upper(core::dirX) + 1;
                pidx(core::dirY)
                    = isLower ? rawBox.lower(core::dirY) + 1 : rawBox.upper(core::dirY);
                push(out.ez, Box(pidx, pidx, rawBox.getBlockId()));

                // Inner clip: when SAMRAI clipped bb.upper(x) < patchCellBox.upper(x).
                if (rawBox.upper(core::dirX) < patchCellBox.upper(core::dirX))
                {
                    Index cidx(dim);
                    cidx(core::dirX) = rawBox.upper(core::dirX) + 1;
                    cidx(core::dirY) = pidx(core::dirY);
                    push(out.ez, Box(cidx, cidx, rawBox.getBlockId()));
                }
            }
            else // normalDir == core::dirX
            {
                // Topmost primal-y node: always at patchCellBox.upper(y)+1.
                Index pidx(dim);
                pidx(core::dirX)
                    = isLower ? rawBox.lower(core::dirX) + 1 : rawBox.upper(core::dirX);
                pidx(core::dirY) = patchCellBox.upper(core::dirY) + 1;
                push(out.ez, Box(pidx, pidx, rawBox.getBlockId()));

                // Inner clip: when SAMRAI clipped bb.upper(y) < patchCellBox.upper(y).
                if (rawBox.upper(core::dirY) < patchCellBox.upper(core::dirY))
                {
                    Index cidx(dim);
                    cidx(core::dirX) = pidx(core::dirX);
                    cidx(core::dirY) = rawBox.upper(core::dirY) + 1;
                    push(out.ez, Box(cidx, cidx, rawBox.getBlockId()));
                }
            }
        }
    }
    else // dimension == 3
    {
        // 3D codim-1: the fine-side reflux E must be accumulated over each whole CF FACE,
        // not just its codim-2 edges (the earlier codim-2 port accumulated E only on the
        // cube's literal edges, leaving every face-interior E-edge at zero → mis-corrected
        // tangential B). This mirrors the 2D codim-1 path. For a face with normal d, the two
        // tangential E components (axes != d) are accumulated; each E_aE is dual along its
        // own axis aE and primal along the third axis t = 3-d-aE, so it needs the +1 primal
        // endpoint node SAMRAI clips off the cell-space box. E-edges on shared cube edges are
        // pushed by both adjacent faces and folded to a single cover by simplify().
        auto const containerFor = [&](int axis) -> SAMRAI::hier::BoxContainer& {
            if (axis == core::dirX) return out.ex;
            if (axis == core::dirY) return out.ey;
            return out.ez;
        };

        for (auto const& bb : cfBoundary.getBoundaries(patchId, 1))
        {
            auto const location = bb.getLocationIndex();
            bool const isLower   = (location % 2 == 0);
            int const normalDir  = location / 2;
            auto const& rawBox   = bb.getBox();

            for (int aE = 0; aE < 3; ++aE)
            {
                if (aE == normalDir) continue;        // only tangential E components
                int const t = 3 - normalDir - aE;     // primal transverse axis of E_aE

                Index lo = rawBox.lower(), hi = rawBox.upper();
                if (isLower)                           // shift to boundary-flux read coord
                {
                    lo(normalDir) += 1;
                    hi(normalDir) += 1;
                }
                // dual axis aE: clip to patch cells (no node extension).
                lo(aE) = std::max(lo(aE), patchCellBox.lower(aE));
                hi(aE) = std::min(hi(aE), patchCellBox.upper(aE));
                // primal axis t: clip to patch cells, then +1 to reach the primal endpoint
                // node beyond the last cell (the node SAMRAI clips off).
                lo(t) = std::max(lo(t), patchCellBox.lower(t));
                hi(t) = std::min(hi(t), patchCellBox.upper(t)) + 1;
                if (lo(aE) > hi(aE) || lo(t) > hi(t)) continue;

                push(containerFor(aE), Box(lo, hi, rawBox.getBlockId()));
            }
        }
    }

    // Box-level dedup: canonical, non-overlapping cover. Each E node is enumerated exactly
    // once across all contributing boundary boxes — replaces the per-index seenEzNodes set.
    if (!out.ex.empty()) out.ex.simplify();
    if (!out.ey.empty()) out.ey.simplify();
    if (!out.ez.empty()) out.ez.simplify();

    return out;
}

// ---- shared geometry helpers (coarse side) ----

// AMR normal-direction coordinate of the coarse cell adjacent to a CF interface.
// Lower side: one cell below the fine box; upper side: one cell above.
inline int coarseCellNormal(SAMRAI::hier::Box const& cfBox, int dir, bool isLower)
{
    return isLower ? cfBox.lower(dir) - 1 : cfBox.upper(dir) + 1;
}

// Coarse cells adjacent to a single coarsened-fine box's CF interface for (dir, side):
// a slab at coarseCellNormal, transverse extent = cfBox (+/- expand in non-normal dirs),
// clipped to the patch and with all fine-covered cells removed.
inline SAMRAI::hier::BoxContainer
adjacentCellsForBox(SAMRAI::hier::Box const& cfBox, int dir, int side,
                    SAMRAI::hier::Box const& patchAMRBox,
                    std::vector<SAMRAI::hier::Box> const& coarsenedFine, int expand)
{
    using SAMRAI::hier::Box;
    using SAMRAI::hier::Index;
    auto const dim         = patchAMRBox.getDim();
    int const ndim         = dim.getValue();
    bool const isLower     = (side == 0);
    int const cellCoord    = coarseCellNormal(cfBox, dir, isLower);

    Index lo(dim), hi(dim);
    for (int d = 0; d < ndim; ++d)
    {
        if (d == dir) { lo(d) = cellCoord; hi(d) = cellCoord; }
        else
        {
            lo(d) = cfBox.lower(d) - expand;
            hi(d) = cfBox.upper(d) + expand;
        }
    }
    SAMRAI::hier::BoxContainer cells(Box(lo, hi, cfBox.getBlockId()));
    cells.intersectBoxes(patchAMRBox);
    for (auto const& cb : coarsenedFine)
        cells.removeIntersections(cb);
    return cells;
}

// Coarse cells adjacent to the whole CF interface for (dir, side), box-deduped across all
// coarsened-fine boxes. expand=0 yields the hydro flux-correction cells; expand=1 the B
// extension cells. The normal-direction flux read coordinate is recoverable per cell from
// amrIdx[dir] (= isLower ? amrIdx[dir]+1 : amrIdx[dir]), so cfBox identity need not survive.
inline SAMRAI::hier::BoxContainer
cfAdjacentCoarseCells(int dir, int side, SAMRAI::hier::Box const& patchAMRBox,
                      std::vector<SAMRAI::hier::Box> const& coarsenedFine, int expand)
{
    SAMRAI::hier::BoxContainer cells;
    for (auto const& cfBox : coarsenedFine)
        for (auto const& cb : adjacentCellsForBox(cfBox, dir, side, patchAMRBox, coarsenedFine,
                                                  expand))
            cells.push_back(cb);
    if (!cells.empty()) cells.simplify();
    return cells;
}

// Centering-driven coarse Yee B-face box: normal direction pinned to the adjacent coarse
// cell coordinate, transverse extent following ccBox with a +1 for each primal direction.
template<typename GridLayout, typename BQ>
SAMRAI::hier::Box
makeComponentBox(GridLayout const& layout, BQ bQty, int normalDir,
                 int cellCoord, SAMRAI::hier::Box const& ccBox)
{
    constexpr auto dimension = GridLayout::dimension;
    auto const dim           = ccBox.getDim();
    SAMRAI::hier::Index lo(dim), hi(dim);
    auto const centering = layout.centering(bQty);
    for (int d = 0; d < static_cast<int>(dimension); ++d)
    {
        if (d == normalDir) { lo(d) = cellCoord; hi(d) = cellCoord; }
        else
        {
            lo(d) = ccBox.lower(d);
            hi(d) = ccBox.upper(d) + (centering[d] == core::QtyCentering::primal ? 1 : 0);
        }
    }
    return SAMRAI::hier::Box(lo, hi, ccBox.getBlockId());
}

// ---- P3 — coarse Yee B-face enumeration on the CF boundary (unified) ----

// Yields each coarse Yee B-face that needs a Faraday correction for (bQty, dir, side) ONCE,
// box-deduped across all coarsened-fine boxes (replaces the per-index seenBi set).
//
// M6 seam reach: cells extend +/-1 in non-normal dirs (expand=1) so a patch whose AMR box
// starts just past a cfBox transverse extent still reaches the shared primal seam face;
// dual transverse dirs clip the extension back to the cfBox extent (a dual face gains no
// new boundary face from the extension), and primal transverse dirs clip the face box to
// [cfBox.lower, cfBox.upper+1] so the extension reaches only the seam face, not beyond.
template<typename GridLayout, typename BQ>
SAMRAI::hier::BoxContainer
cfBFaceBoxes(GridLayout const& layout, BQ bQty, int dir, int side,
             SAMRAI::hier::Box const& patchAMRBox,
             std::vector<SAMRAI::hier::Box> const& coarsenedFine)
{
    using SAMRAI::hier::Box;
    using SAMRAI::hier::Index;
    constexpr auto dimension = GridLayout::dimension;
    bool const isLower       = (side == 0);
    auto const centering     = layout.centering(bQty);

    SAMRAI::hier::BoxContainer faces;
    for (auto const& cfBox : coarsenedFine)
    {
        int const cellCoord = coarseCellNormal(cfBox, dir, isLower);
        for (auto const& rawBox :
             adjacentCellsForBox(cfBox, dir, side, patchAMRBox, coarsenedFine, 1))
        {
            // Dual non-normal directions: the +/-1 extension cells are invalid (no new
            // primal boundary face), so clip them back to the cfBox transverse extent.
            Index ccLo = rawBox.lower(), ccHi = rawBox.upper();
            bool ccEmpty = false;
            for (int d = 0; d < static_cast<int>(dimension); ++d)
            {
                if (d == dir || centering[d] == core::QtyCentering::primal) continue;
                ccLo(d) = std::max(ccLo(d), cfBox.lower(d));
                ccHi(d) = std::min(ccHi(d), cfBox.upper(d));
                if (ccLo(d) > ccHi(d)) { ccEmpty = true; break; }
            }
            if (ccEmpty) continue;
            Box const ccBox(ccLo, ccHi, rawBox.getBlockId());

            Box const biBox = makeComponentBox(layout, bQty, dir, cellCoord, ccBox);
            Index newLo = biBox.lower(), newHi = biBox.upper();
            for (int d = 0; d < static_cast<int>(dimension); ++d)
            {
                if (d == dir || centering[d] != core::QtyCentering::primal) continue;
                newLo(d) = std::max(newLo(d), cfBox.lower(d));
                newHi(d) = std::min(newHi(d), cfBox.upper(d) + 1);
            }
            Box const biBoxClipped(newLo, newHi, biBox.getBlockId());
            if (!biBoxClipped.empty()) faces.push_back(biBoxClipped);
        }
    }
    if (!faces.empty()) faces.simplify();
    return faces;
}

// M8 step-corner guard: a B-face is inside the fine region (and must NOT be corrected) when
// either coarse cell neighboring it in a primal transverse direction is fine-covered.
template<typename GridLayout, typename BQ>
bool bFaceInsideFine(GridLayout const& layout, BQ bQty, int dir,
                     core::Point<int, GridLayout::dimension> const& amrIdx,
                     std::vector<SAMRAI::hier::Box> const& coarsenedFine)
{
    constexpr auto dimension = GridLayout::dimension;
    if (coarsenedFine.empty()) return false;
    auto const dim       = coarsenedFine.front().getDim();
    auto const centering = layout.centering(bQty);
    for (int d = 0; d < static_cast<int>(dimension); ++d)
    {
        if (d == dir || centering[d] != core::QtyCentering::primal) continue;
        for (int delta : {-1, 0})
        {
            SAMRAI::hier::Index ccNeighbor(dim);
            for (int dd = 0; dd < static_cast<int>(dimension); ++dd)
                ccNeighbor(dd) = amrIdx[dd];
            ccNeighbor(d) += delta;
            for (auto const& cb : coarsenedFine)
                if (cb.contains(ccNeighbor)) return true;
        }
    }
    return false;
}

} // namespace PHARE::solver::reflux_geometry

#endif
