#ifndef PHARE_COMPOSITE_FIELD_REFINER_HPP
#define PHARE_COMPOSITE_FIELD_REFINER_HPP


#include "core/def/phare_mpi.hpp" // IWYU pragma: keep

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/utilities/point/point.hpp"
#include "core/numerics/slope_limiters/min_mod.hpp"
#include "core/numerics/slope_limiters/van_leer.hpp"

#include "amr/resources_manager/amr_utils.hpp"
#include "amr/utilities/box/amr_box.hpp"
#include "field_refiner_kernel.hpp"

#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/IntVector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>


namespace PHARE::amr
{

/** @brief Tag for the unlimited (smooth) prolongation path. Limiters (Step 4) are distinct types. */
struct NoLimiter
{
};


/**
 * @brief Runtime field-refinement kernel built from the two 1-D primitives tensor-producted.
 *
 * For ratio 2, each fine index maps to a coarse anchor I = floor(f/2) (toCoarseIndex) and a parity
 * p = f − 2I ∈ {0,1} per direction. Per direction the 1-D weight row is chosen by centering+parity:
 *   - primal, p=0 : exact copy (coincident node)            → {offset 0, 1.0}
 *   - primal, p=1 : half-point midpoint (Primitive A.2)     → directionalInterp<dir,PrimalToDual,order>
 *   - dual,   p   : ±¼ child ladder (Primitive B)           → directionalProlongation<dir, σ=2p−1, order>
 * The multi-D stencil is the outer product of the rows (tensorProductRuntime). Centering is constant
 * over a refineBox call, so the 2^dim parity-combo stencils are built once per box, then looked up
 * per fine index. Offsets are relative to the anchor I; values are gathered from the coarse field.
 */
template<typename GridLayoutT, typename FieldT, std::size_t order, typename Limiter = NoLimiter,
         bool sharedFacesOnly = false, Representation representation = Representation::Average>
class CompositeFieldRefiner : public IFieldRefineKernel<GridLayoutT, FieldT>
{
    static constexpr std::size_t dimension = GridLayoutT::dimension;
    using GridLayoutImpl                   = typename GridLayoutT::implT;
    using Point_t                          = core::Point<int, dimension>;
    using WeightPoint_t                    = core::WeightPoint<dimension>;

    static_assert(order == 2 || order == 4, "composite refiner ladder is order 2 (Linear) / 4 (Cubic)");
    static_assert(std::is_same_v<Limiter, NoLimiter> || std::is_same_v<Limiter, core::MinModLimiter>
                      || std::is_same_v<Limiter, core::VanLeerLimiter>,
                  "Limiter must be NoLimiter, MinModLimiter or VanLeerLimiter");

    static constexpr bool limited = !std::is_same_v<Limiter, NoLimiter>;
    // guard against 0/0 when the high-order slope vanishes (local flat / extremum).
    static constexpr double slopeEps_ = std::numeric_limits<double>::min();

public:
    void refineBox(FieldT const& sourceField, FieldT& destinationField,
                   SAMRAI::hier::Box const& intersectionBox,
                   std::array<core::QtyCentering, dimension> const& centering,
                   SAMRAI::hier::Box const& destFieldBox, SAMRAI::hier::Box const& sourceFieldBox,
                   SAMRAI::hier::IntVector const& ratio) const override
    {
        for (std::size_t d = 0; d < dimension; ++d)
            if (ratio(d) != 2)
                throw std::runtime_error("CompositeFieldRefiner supports refinement ratio 2 only");

        // For a face field (B): the single primal direction is the face normal. Shared faces are
        // the even-normal-parity ones (coincident with a coarse face); the odd-normal (interior)
        // faces are owned by the Tóth-Roe postprocess, which overwrites them and never reads them,
        // so we skip them here. Tangential directions are dual ⇒ their children are always filled.
        // A B component is primal in at most one direction (its face normal). When it has one
        // (in-plane Bx/By/Bz), the odd-normal interior faces are owned by the Tóth-Roe postprocess
        // and skipped below. When it has none (out-of-plane / reduced-dimension component: By,Bz in
        // 1D, Bz in 2D), there is no normal face and no ∇·B interior to protect, so it prolongs like
        // any same-centered (all-dual) quantity — every fine face is filled.
        std::size_t normalDir = 0;
        bool hasNormal        = false;
        if constexpr (sharedFacesOnly)
        {
            int primalCount = 0;
            for (std::size_t d = 0; d < dimension; ++d)
                if (centering[d] == core::QtyCentering::primal)
                {
                    normalDir = d;
                    ++primalCount;
                }
            if (primalCount > 1)
                throw std::runtime_error(
                    "magnetic shared-face refiner expects at most one primal (normal) direction");
            hasNormal = (primalCount == 1);
        }

        // Unlimited fast path: centering is fixed over the box, so the rows are data-independent.
        // Build the 2^dim parity-combo stencils once, then look up per fine index.
        // Limited path: the rows depend on the local coarse values (slope clip), so they must be
        // (re)built per fine index from the central line through the anchor.
        std::array<std::vector<WeightPoint_t>, (1u << dimension)> combos;
        if constexpr (!limited)
            for (std::size_t combo = 0; combo < combos.size(); ++combo)
                combos[combo] = makeStencil_(centering, combo);

        for (auto const fineIndex : phare_box_from<dimension>(intersectionBox))
        {
            auto const anchor = toCoarseIndex<dimension>(fineIndex);

            std::size_t combo = 0;
            for (std::size_t d = 0; d < dimension; ++d)
                combo |= static_cast<std::size_t>(fineIndex[d] - 2 * anchor[d]) << d;

            if constexpr (sharedFacesOnly)
                if (hasNormal && ((combo >> normalDir) & 1u) != 0u)
                    continue; // interior-normal face: left to Tóth-Roe postprocess

            auto const anchorLocal = AMRToLocal(anchor, sourceFieldBox);
            auto const fineLocal   = AMRToLocal(fineIndex, destFieldBox);

            double value = 0.;
            if constexpr (limited)
            {
                auto const stencil = makeLimitedStencil_(centering, combo, sourceField, anchorLocal);
                for (auto const& w : stencil)
                    value += w.coef * sampleCoarse_(sourceField, anchorLocal, w.indexes);
            }
            else
            {
                for (auto const& w : combos[combo])
                    value += w.coef * sampleCoarse_(sourceField, anchorLocal, w.indexes);
            }

            assignFine_(destinationField, fineLocal, value);
        }
    }

    // order 2 reads ±1 coarse cell, order 4 reads ±2 (max |offset| over both 1-D primitives).
    int coarseStencilWidth() const override { return order / 2; }

private:
    // 1-D weight row for direction d, given its centering and parity (0/1), as a runtime vector.
    template<std::size_t d>
    static std::vector<WeightPoint_t> oneDRow_(core::QtyCentering c, std::size_t parity)
    {
        std::vector<WeightPoint_t> row;
        if (c == core::QtyCentering::primal)
        {
            if (parity == 0)
                row.push_back({Point_t{}, 1.0}); // coincident node: exact copy
            else
                for (auto const& w :
                     GridLayoutImpl::template directionalInterp<d, GridLayoutImpl::InterpDir::PrimalToDual, order>())
                    row.push_back(w);
        }
        else // dual: σ = 2·parity − 1
        {
            // Average world means back to the coarse cell average (conservative ladder); point-value
            // world is a plain point Lagrange at ±¼. The rows differ by an O(H²·u″) curvature term at
            // every order, but that gap only exceeds the scheme truncation error (and so only matters)
            // at order ≥4 — at order 2 either row is correct to scheme order.
            if constexpr (representation == Representation::PointValue)
            {
                if (parity == 0)
                    for (auto const& w :
                         GridLayoutImpl::template directionalProlongationPointValue<d, -1, order>())
                        row.push_back(w);
                else
                    for (auto const& w :
                         GridLayoutImpl::template directionalProlongationPointValue<d, +1, order>())
                        row.push_back(w);
            }
            else
            {
                if (parity == 0)
                    for (auto const& w : GridLayoutImpl::template directionalProlongation<d, -1, order>())
                        row.push_back(w);
                else
                    for (auto const& w : GridLayoutImpl::template directionalProlongation<d, +1, order>())
                        row.push_back(w);
            }
        }
        return row;
    }

    static std::vector<WeightPoint_t>
    makeStencil_(std::array<core::QtyCentering, dimension> const& centering, std::size_t combo)
    {
        auto const parity = [combo](std::size_t d) { return (combo >> d) & 1u; };

        if constexpr (dimension == 1)
            return oneDRow_<0>(centering[0], parity(0));
        else if constexpr (dimension == 2)
            return GridLayoutImpl::template tensorProductRuntime<0, 1>(
                oneDRow_<0>(centering[0], parity(0)), oneDRow_<1>(centering[1], parity(1)));
        else
            return GridLayoutImpl::template tensorProductRuntime<0, 1, 2>(
                oneDRow_<0>(centering[0], parity(0)), oneDRow_<1>(centering[1], parity(1)),
                oneDRow_<2>(centering[2], parity(2)));
    }

    // ---- limited-prolongation helpers (Step 4) --------------------------------------------------
    //
    // Limiting is per-direction (separable): each 1-D row is computed from the coarse central line
    // through the anchor along that direction, then tensor-producted as in the unlimited path. The
    // rows stay fixed-offset separable rows, so tensorProductRuntime + the σ-antisymmetry argument
    // (⇒ conservation, ⇒ ∇·B-neutral for B tangential) carry over unchanged.

    // sample the coarse field at anchor + k·e_d (central line along direction d).
    template<std::size_t d>
    static double sampleAxis_(FieldT const& src, Point_t const& anchorLocal, int k)
    {
        Point_t off{};
        off[d] = k;
        return sampleCoarse_(src, anchorLocal, off);
    }

    static double median_(double a, double b, double c)
    {
        return std::max(std::min(a, b), std::min(std::max(a, b), c));
    }

    // sign-consistent clip of the high-order slope to the Sweby envelope 2|s2| (s2 the classic
    // limited slope). Keeps s_HO in smooth monotone flow (preserves order), reverts at extrema.
    static double clipToEnvelope_(double s_HO, double s2)
    {
        if (s2 == 0.0)
            return 0.0;
        if ((s_HO > 0.0) != (s2 > 0.0))
            return 0.0; // opposite signs ⇒ non-monotone ⇒ no correction
        return std::copysign(std::min(std::abs(s_HO), 2.0 * std::abs(s2)), s_HO);
    }

    // Primitive-B (dual) limited row: scale the off-center weights of the unlimited ±¼ ladder by
    // φ = s_lim/s_HO so the child becomes C + σ·s_lim/4. φ is σ-independent ⇒ children stay
    // antisymmetric about C ⇒ conservative + ∇·B-neutral, limited or not. φ ≤ 1 by construction
    // (|s_lim| ≤ |s_HO|), so no overshoot from the scaling itself.
    template<std::size_t d>
    static std::vector<WeightPoint_t>
    limitedDualRow_(std::size_t parity, FieldT const& src, Point_t const& anchorLocal)
    {
        double const C   = sampleAxis_<d>(src, anchorLocal, 0);
        double const L   = sampleAxis_<d>(src, anchorLocal, -1);
        double const R   = sampleAxis_<d>(src, anchorLocal, 1);
        double const Dil = C - L;
        double const Dir = R - C;

        double s_HO, s_lim;
        if constexpr (order == 2)
        {
            s_HO  = 0.5 * (R - L);             // centered slope
            s_lim = Limiter::limit(Dil, Dir);  // classic minmod / van Leer
        }
        else // order == 4
        {
            double const L2 = sampleAxis_<d>(src, anchorLocal, -2);
            double const R2 = sampleAxis_<d>(src, anchorLocal, 2);
            double const d1 = R - L;
            double const d2 = R2 - L2;
            s_HO            = (22.0 * d1 - 3.0 * d2) / 32.0; // 5-pt cubic slope
            s_lim           = clipToEnvelope_(s_HO, Limiter::limit(Dil, Dir));
        }

        double const phi = (std::abs(s_HO) > slopeEps_) ? (s_lim / s_HO) : 0.0;

        std::vector<WeightPoint_t> row;
        auto scaleInto = [&phi, &row](auto const& unl) {
            for (auto const& w : unl)
            {
                bool isCenter = true;
                for (std::size_t k = 0; k < dimension; ++k)
                    if (w.indexes[k] != 0)
                        isCenter = false;
                row.push_back({w.indexes, isCenter ? w.coef : w.coef * phi});
            }
        };
        if (parity == 0)
            scaleInto(GridLayoutImpl::template directionalProlongation<d, -1, order>());
        else
            scaleInto(GridLayoutImpl::template directionalProlongation<d, +1, order>());
        return row;
    }

    // Median/bracket-clamped blend for a non-collocated point child sitting between two coarse nodes
    // loA, loB (offsets along d) with monotone linear-fallback weights wA, wB. The high-order row
    // hoRow can overshoot the node bracket [v_loA, v_loB] (negative outer weights): median-clamp it
    // and blend toward the linear fallback by ψ = (m_lim−m_lin)/(m_HO−m_lin) ∈ [0,1]. The returned
    // row is ψ·hoRow + (1−ψ)·linRow (≡ lin + ψ(ho−lin)). Shared by the primal half-point and the
    // point-value dual ±¼ limited paths — both are point interpolations, so the same ψ logic applies
    // (unlike the average dual, which φ-scales a mean-back slope).
    template<std::size_t d>
    static std::vector<WeightPoint_t>
    limitedPointRow_(std::vector<WeightPoint_t> const& hoRow, int loA, int loB, double wA, double wB,
                     FieldT const& src, Point_t const& anchorLocal)
    {
        double m_HO = 0.;
        for (auto const& w : hoRow)
            m_HO += w.coef * sampleCoarse_(src, anchorLocal, w.indexes);

        double const vA    = sampleAxis_<d>(src, anchorLocal, loA);
        double const vB    = sampleAxis_<d>(src, anchorLocal, loB);
        double const m_lin = wA * vA + wB * vB;
        double const m_lim = median_(m_HO, vA, vB);
        double const denom = m_HO - m_lin;
        double const psi   = (std::abs(denom) > slopeEps_) ? (m_lim - m_lin) / denom : 1.0;

        auto make_p = [](int off) {
            Point_t p{};
            p[d] = off;
            return p;
        };
        std::vector<WeightPoint_t> row;
        auto add = [&row](Point_t const& idx, double c) {
            for (auto& w : row)
                if (w.indexes[d] == idx[d])
                {
                    w.coef += c;
                    return;
                }
            row.push_back({idx, c});
        };
        for (auto const& w : hoRow)
            add(w.indexes, psi * w.coef);
        add(make_p(loA), (1.0 - psi) * wA);
        add(make_p(loB), (1.0 - psi) * wB);
        return row;
    }

    // Primitive-A.2 (primal half-point, odd parity) limited row. Order 2 is the convex 2-pt mean —
    // never overshoots, returned as-is (limiting self-disables). Order 4's 4-pt midpoint (Lagrange@½)
    // has negative outer weights and can overshoot the node bracket [v_I, v_{I+1}]: clamp via
    // limitedPointRow_ with the symmetric ½/½ linear fallback. B never reaches the half-point (normal
    // is copy or Tóth-Roe), so this carries no ∇·B concern.
    template<std::size_t d>
    static std::vector<WeightPoint_t>
    limitedPrimalHalfRow_(FieldT const& src, Point_t const& anchorLocal)
    {
        std::vector<WeightPoint_t> hoRow;
        for (auto const& w : GridLayoutImpl::template directionalInterp<
                 d, GridLayoutImpl::InterpDir::PrimalToDual, order>())
            hoRow.push_back(w);

        if constexpr (order == 2)
            return hoRow; // convex 2-pt mean — never overshoots
        else                // order == 4: half-point between nodes at offsets 0 and +1
            return limitedPointRow_<d>(hoRow, 0, 1, 0.5, 0.5, src, anchorLocal);
    }

    // Point-value (non-mean-back) dual ±¼ limited row. Unlike the average φ-scale (limitedDualRow_),
    // a point value has no conservation constraint to lean on, so we clamp it like any other point
    // interpolation: median/bracket-clamp the PV Lagrange row toward the asymmetric ¾/¼ linear
    // fallback between the two bracketing coarse nodes (the child at σ·¼ is ¾ of the way from the far
    // node to the near one). Clamps at order 2 as well (the 3-pt PV row is not monotone-safe).
    template<std::size_t d>
    static std::vector<WeightPoint_t>
    limitedDualRowPointValue_(std::size_t parity, FieldT const& src, Point_t const& anchorLocal)
    {
        std::vector<WeightPoint_t> hoRow;
        if (parity == 0) // σ = −1: child at −¼, between nodes −1 and 0
        {
            for (auto const& w :
                 GridLayoutImpl::template directionalProlongationPointValue<d, -1, order>())
                hoRow.push_back(w);
            return limitedPointRow_<d>(hoRow, -1, 0, 0.25, 0.75, src, anchorLocal);
        }
        else // σ = +1: child at +¼, between nodes 0 and +1
        {
            for (auto const& w :
                 GridLayoutImpl::template directionalProlongationPointValue<d, +1, order>())
                hoRow.push_back(w);
            return limitedPointRow_<d>(hoRow, 0, 1, 0.75, 0.25, src, anchorLocal);
        }
    }

    // 1-D limited row for direction d, by centering + parity. Primal even = exact copy (never
    // limited); primal odd = ψ-clamped half-point (world-shared); dual = φ-scaled ±¼ ladder in the
    // average world, ψ-clamped PV ±¼ row in the point-value world.
    template<std::size_t d>
    static std::vector<WeightPoint_t> limitedRow_(core::QtyCentering c, std::size_t parity,
                                                  FieldT const& src, Point_t const& anchorLocal)
    {
        if (c == core::QtyCentering::primal)
        {
            if (parity == 0)
                return std::vector<WeightPoint_t>{{Point_t{}, 1.0}};
            return limitedPrimalHalfRow_<d>(src, anchorLocal);
        }
        if constexpr (representation == Representation::PointValue)
            return limitedDualRowPointValue_<d>(parity, src, anchorLocal);
        return limitedDualRow_<d>(parity, src, anchorLocal);
    }

    static std::vector<WeightPoint_t>
    makeLimitedStencil_(std::array<core::QtyCentering, dimension> const& centering, std::size_t combo,
                        FieldT const& src, Point_t const& anchorLocal)
    {
        auto const parity = [combo](std::size_t d) { return (combo >> d) & 1u; };

        if constexpr (dimension == 1)
            return limitedRow_<0>(centering[0], parity(0), src, anchorLocal);
        else if constexpr (dimension == 2)
            return GridLayoutImpl::template tensorProductRuntime<0, 1>(
                limitedRow_<0>(centering[0], parity(0), src, anchorLocal),
                limitedRow_<1>(centering[1], parity(1), src, anchorLocal));
        else
            return GridLayoutImpl::template tensorProductRuntime<0, 1, 2>(
                limitedRow_<0>(centering[0], parity(0), src, anchorLocal),
                limitedRow_<1>(centering[1], parity(1), src, anchorLocal),
                limitedRow_<2>(centering[2], parity(2), src, anchorLocal));
    }

    static double sampleCoarse_(FieldT const& src, Point_t const& anchorLocal, Point_t const& offset)
    {
        if constexpr (dimension == 1)
            return src(anchorLocal[0] + offset[0]);
        else if constexpr (dimension == 2)
            return src(anchorLocal[0] + offset[0], anchorLocal[1] + offset[1]);
        else
            return src(anchorLocal[0] + offset[0], anchorLocal[1] + offset[1],
                       anchorLocal[2] + offset[2]);
    }

    // preserve the legacy NaN-guard: only fill fine indices not already set
    static void assignFine_(FieldT& dst, Point_t const& fineLocal, double value)
    {
        if constexpr (dimension == 1)
        {
            if (std::isnan(dst(fineLocal[0])))
                dst(fineLocal[0]) = value;
        }
        else if constexpr (dimension == 2)
        {
            if (std::isnan(dst(fineLocal[0], fineLocal[1])))
                dst(fineLocal[0], fineLocal[1]) = value;
        }
        else
        {
            if (std::isnan(dst(fineLocal[0], fineLocal[1], fineLocal[2])))
                dst(fineLocal[0], fineLocal[1], fineLocal[2]) = value;
        }
    }
};


// ---- factory (declared in field_refiner_kernel.hpp) ---------------------------------------------

template<typename GridLayoutT, typename FieldT, Representation R>
std::shared_ptr<IFieldRefineKernel<GridLayoutT, FieldT>>
makeRefineKernel(int order, std::string const& limiter)
{
    auto build = [&limiter]<std::size_t O>() -> std::shared_ptr<IFieldRefineKernel<GridLayoutT, FieldT>> {
        if (limiter == "none" || limiter.empty())
            return std::make_shared<
                CompositeFieldRefiner<GridLayoutT, FieldT, O, NoLimiter, false, R>>();
        if (limiter == "minmod")
            return std::make_shared<
                CompositeFieldRefiner<GridLayoutT, FieldT, O, core::MinModLimiter, false, R>>();
        if (limiter == "vanleer")
            return std::make_shared<
                CompositeFieldRefiner<GridLayoutT, FieldT, O, core::VanLeerLimiter, false, R>>();
        throw std::runtime_error("makeRefineKernel: unknown limiter (none|minmod|vanleer): " + limiter);
    };

    switch (order)
    {
        case 2: return build.template operator()<2>();
        case 4: return build.template operator()<4>();
        default:
            throw std::runtime_error("makeRefineKernel: order must be 2 (Linear) or 4 (Cubic)");
    }
}


} // namespace PHARE::amr


#endif
