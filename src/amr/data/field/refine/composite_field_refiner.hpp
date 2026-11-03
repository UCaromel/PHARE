#ifndef PHARE_COMPOSITE_FIELD_REFINER_HPP
#define PHARE_COMPOSITE_FIELD_REFINER_HPP


#include "phare_mpi.hpp" // IWYU pragma: keep

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/utilities/point/point.hpp"
#include "core/utilities/types.hpp"

#include "amr/resources_manager/amr_utils.hpp"
#include "amr/utilities/box/amr_box.hpp"
#include "field_refiner_kernel.hpp"

#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/IntVector.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>


namespace PHARE::amr
{

//! Tag: the unlimited composite fill (compile-time stencils).
struct NoLimiter
{
};

/**
 * @brief Tag: the WENO-limited composite fill.
 *
 * Each dual ±¼ row blends its two one-sided slopes with Jiang-Shu weights (WENO-AO, linear
 * weights ½,½, q = 2) instead of using their central mean, after Balsara, Samantaray &
 * Subramanian (2023), doi:10.1007/s42967-021-00182-x. Limiting acts on the 1-D rows, which keep
 * summing to 1 with a σ-antisymmetric correction, so the children still mean back to the coarse
 * value: conservation and the coarse face means that ∇·B relies on hold limited or not.
 */
struct WenoLimiter
{
};

namespace detail
{
    /**
     * @brief The composite refinement fill: the order-`order` stencil tables, and the loop that
     * applies them to every fine index of a region.
     *
     * For ratio 2, each fine index maps to a coarse anchor I = floor(f/2) (toCoarseIndex) and a
     * parity p = f − 2I ∈ {0,1} per direction. Per direction the 1-D weight row is chosen by
     * centering+parity:
     *   - primal, p=0 : exact copy (coincident node)            → {offset 0, 1.0}
     *   - primal, p=1 : half-point midpoint (Primitive A.2)     → directionalInterp<dir,
     *                                                             PrimalToDual>
     *   - dual,   p   : ±¼ child ladder (Primitive B)           → directionalProlongation<dir,
     *                                                             σ=2p−1>
     * (both at <order>). The multi-D stencil is the outer product of the rows (consteval
     * tensorProduct), which sizes each multi-D stencil exactly from its rows' own lengths (copy=1,
     * half-point=2, dual-σ±=3) — no padding to a common length; the full set of centering×parity
     * combinations (4^dim of them) is a compile-time table, indexed at runtime per fine index by
     * centering+parity. Offsets are relative to the anchor I; values are gathered from the coarse
     * field.
     *
     * Stateless, with no virtual of its own: the kernels below are thin IFieldRefineKernel
     * adapters over it, and differ only in the region they hand it. So a kernel costs one virtual
     * call per overlap box and none per fine index.
     *
     * With WenoLimiter the dual rows depend on the data, so they are built per fine index from the
     * coarse line through the anchor along each direction, then tensor-producted at runtime. They
     * read the same ±1 coarse cells as the unlimited rows, and reduce to them exactly on linear
     * data.
     */
    template<typename GridLayoutT, typename FieldT, std::size_t order, typename Limiter = NoLimiter>
    struct CompositeFill
    {
        static constexpr std::size_t dimension = GridLayoutT::dimension;

        static_assert(order == 2, "composite refiner ladder is order 2 (Linear)");
        static_assert(std::is_same_v<Limiter, NoLimiter> or std::is_same_v<Limiter, WenoLimiter>,
                      "Limiter must be NoLimiter or WenoLimiter");

        // order 2 reads ±1 coarse cell (max |offset| over both 1-D primitives).
        static constexpr int coarseStencilWidth() { return order / 2; }

        /**
         * @brief Fill every fine index of `region` from the coarse field.
         *
         * `region` is the caller's to choose: it must lie inside the destination allocation, and
         * the kernel writes only the fine indices it contains.
         */
        static void fill(FieldT const& sourceField, FieldT& destinationField,
                         SAMRAI::hier::Box const& region,
                         std::array<core::QtyCentering, dimension> const& centering,
                         SAMRAI::hier::Box const& destFieldBox,
                         SAMRAI::hier::Box const& sourceFieldBox,
                         SAMRAI::hier::IntVector const& ratio)
        {
            for (std::size_t d = 0; d < dimension; ++d)
                if (ratio(d) != 2)
                    throw std::runtime_error(
                        "composite field refinement supports refinement ratio 2 only");

            // Centering is fixed over the box, parity varies per fine index. The 2-bits-per-
            // direction index built below (bit0 = parity, bit1 = dual) selects the matching
            // compile-time stencil.
            for (auto const fineIndex : phare_box_from<dimension>(region))
            {
                auto const anchor = toCoarseIndex<dimension>(fineIndex);

                std::size_t combined = 0; // 2 bits/dir: (dual<<1)|parity
                for (std::size_t d = 0; d < dimension; ++d)
                {
                    auto const parity = static_cast<std::size_t>(fineIndex[d] - 2 * anchor[d]);
                    std::size_t const dualBit
                        = (centering[d] == core::QtyCentering::dual) ? 1u : 0u;
                    combined |= ((dualBit << 1) | parity) << (2u * d);
                }

                auto const anchorLocal = AMRToLocal(anchor, sourceFieldBox);
                auto const fineLocal   = AMRToLocal(fineIndex, destFieldBox);

                double value;
                if constexpr (std::is_same_v<Limiter, WenoLimiter>)
                    value = limitedGather_(sourceField, anchorLocal, combined);
                else
                    value = gatherers_[combined](sourceField, anchorLocal);

                assignFine_(destinationField, fineLocal, value);
            }
        }

    private:
        using GridLayoutImpl = GridLayoutT::implT;
        using Point_t        = core::Point<int, dimension>;
        using WeightPoint_t  = core::WeightPoint<dimension>;

        // 2 bits of choice per direction (dual<<1 | parity) → 4^dim combinations.
        static constexpr std::size_t nCombined = std::size_t{1} << (2 * dimension);

        // 1-D weight row for direction d and the compile-time choice (dual<<1 | parity).
        template<std::size_t d, std::size_t choice>
        static consteval auto oneDRow_()
        {
            static_assert(choice < 4, "choice is 2 bits: dual << 1 | parity");

            if constexpr (choice == 0) // primal, parity 0: coincident node, exact copy
                return std::array{WeightPoint_t{Point_t{}, 1.0}};
            else if constexpr (choice == 1) // primal, parity 1: half-point midpoint
                return GridLayoutImpl::template directionalInterp<
                    d, GridLayoutImpl::InterpDir::PrimalToDual, order>();
            else if constexpr (choice == 2) // dual, parity 0: σ = −1 child
                return GridLayoutImpl::template directionalProlongation<d, -1, order>();
            else // choice == 3 : dual, parity 1: σ = +1 child
                return GridLayoutImpl::template directionalProlongation<d, +1, order>();
        }

        // full multi-D stencil (exact size) for the combined choice index (2 bits per direction).
        template<std::size_t combined>
        static consteval auto makeStencil_()
        {
            static_assert(dimension >= 1 and dimension <= 3);

            if constexpr (dimension == 1)
                return oneDRow_<0, combined & 3u>();
            else if constexpr (dimension == 2)
                return GridLayoutImpl::template tensorProduct<0, 1>(
                    oneDRow_<0, combined & 3u>(), oneDRow_<1, (combined >> 2) & 3u>());
            else
                return GridLayoutImpl::template tensorProduct<0, 1, 2>(
                    oneDRow_<0, combined & 3u>(), oneDRow_<1, (combined >> 2) & 3u>(),
                    oneDRow_<2, (combined >> 4) & 3u>());
        }

        // gather the coarse contributions for one fine index using the exact stencil of choice
        // `combined` (its size is baked in at compile time — no zero-coef points iterated).
        template<std::size_t combined>
        static double gatherStencil_(FieldT const& src, Point_t const& anchorLocal)
        {
            return core::sum_from(makeStencil_<combined>(), [&](auto const& w) {
                return w.coef * src(anchorLocal + w.indexes);
            });
        }

        // runtime centering+parity → compile-time stencil: one gather fn per combination,
        // dispatched by a plain index.
        using Gatherer_t                 = double (*)(FieldT const&, Point_t const&);
        static constexpr auto gatherers_ = core::for_N_make_array<nCombined>(
            [](auto ic) { return Gatherer_t{&gatherStencil_<ic()>}; });

        // ---- WENO-limited path
        // -------------------------------------------------------------------

        // one 1-D row of at most 3 points (offsets along its own direction)
        struct Row_
        {
            std::array<int, 3> offset{};
            std::array<double, 3> weight{};
            std::size_t size = 0;
        };

        // a compile-time row (copy or half-point) as a Row_
        template<std::size_t d, std::size_t choice>
        static consteval Row_ fixedRow_()
        {
            auto const row = oneDRow_<d, choice>();
            Row_ out;
            for (std::size_t k = 0; k < row.size(); ++k)
            {
                out.offset[k] = row[k].indexes[d];
                out.weight[k] = row[k].coef;
            }
            out.size = row.size();
            return out;
        }

        template<std::size_t d>
        static double alongAxis_(FieldT const& src, Point_t const& anchorLocal, int const k)
        {
            Point_t offset{};
            offset[d] = k;
            return src(anchorLocal + offset);
        }

        // Dual row of child σ: ū + σ·m/4 with m = wL·(ū − ū₋₁) + wR·(ū₊₁ − ū), the WENO blend
        // of the two one-sided slopes. Weights ∝ γ/IS² with γ = ½ each and IS = slope², i.e.
        // wL = IS_R²/(IS_L² + IS_R²); equal slopes give wL = wR = ½, the unlimited row
        // {−σ/8, 1, σ/8}. Two flat slopes (0/0) fall back to that same row.
        template<std::size_t d>
        static Row_ wenoDualRow_(double const sigma, FieldT const& src, Point_t const& anchorLocal)
        {
            double const left
                = alongAxis_<d>(src, anchorLocal, 0) - alongAxis_<d>(src, anchorLocal, -1);
            double const right
                = alongAxis_<d>(src, anchorLocal, 1) - alongAxis_<d>(src, anchorLocal, 0);

            double const isL = left * left;
            double const isR = right * right;
            double const den = isL * isL + isR * isR;

            double const wL = den > 0. ? (isR * isR) / den : 0.5;
            double const wR = 1. - wL;

            return Row_{
                {-1, 0, 1}, {-sigma * wL / 4., 1. + sigma * (wL - wR) / 4., sigma * wR / 4.}, 3};
        }

        // Only the dual rows are limited. The primal copy needs nothing, and the order-2
        // half-point is a convex two-point mean with a single admissible stencil, so there is
        // nothing to blend.
        template<std::size_t d>
        static Row_ limitedRow_(std::size_t const choice, FieldT const& src,
                                Point_t const& anchorLocal)
        {
            static constexpr Row_ copy      = fixedRow_<d, 0>();
            static constexpr Row_ halfPoint = fixedRow_<d, 1>();

            switch (choice)
            {
                case 0: return copy;
                case 1: return halfPoint;
                case 2: return wenoDualRow_<d>(-1., src, anchorLocal);
                default: return wenoDualRow_<d>(+1., src, anchorLocal);
            }
        }

        template<std::size_t d>
        static double tensorGather_(std::array<Row_, dimension> const& rows, FieldT const& src,
                                    Point_t const& anchorLocal, Point_t& offset)
        {
            double sum = 0.;
            for (std::size_t k = 0; k < rows[d].size; ++k)
            {
                offset[d] = rows[d].offset[k];
                if constexpr (d + 1 == dimension)
                    sum += rows[d].weight[k] * src(anchorLocal + offset);
                else
                    sum += rows[d].weight[k] * tensorGather_<d + 1>(rows, src, anchorLocal, offset);
            }
            return sum;
        }

        static double limitedGather_(FieldT const& src, Point_t const& anchorLocal,
                                     std::size_t const combined)
        {
            auto const rows = [&]<std::size_t... d>(std::index_sequence<d...>) {
                return std::array<Row_, dimension>{
                    limitedRow_<d>((combined >> (2u * d)) & 3u, src, anchorLocal)...};
            }(std::make_index_sequence<dimension>{});

            Point_t offset{};
            return tensorGather_<0>(rows, src, anchorLocal, offset);
        }

        // preserve the legacy NaN-guard: only fill fine indices not already set
        static void assignFine_(FieldT& dst, Point_t const& fineLocal, double const value)
        {
            if (auto& dst_val = dst(fineLocal); std::isnan(dst_val))
                dst_val = value;
        }
    };

} // namespace detail


/**
 * @brief Runtime field-refinement kernel: the composite fill over exactly the overlap SAMRAI hands
 * it.
 *
 * Every quantity but B refines this way. B needs its gather widened to whole coarse cells, which
 * is its own kernel (magnetic_composite_refiner.hpp).
 */
template<typename GridLayoutT, typename FieldT, std::size_t order, typename Limiter = NoLimiter>
class CompositeFieldRefiner final : public IFieldRefineKernel<GridLayoutT, FieldT>
{
    static constexpr std::size_t dimension = GridLayoutT::dimension;

    using Fill = detail::CompositeFill<GridLayoutT, FieldT, order, Limiter>;

public:
    void refineBox(FieldT const& sourceField, FieldT& destinationField,
                   SAMRAI::hier::Box const& intersectionBox,
                   std::array<core::QtyCentering, dimension> const& centering,
                   SAMRAI::hier::Box const& destFieldBox, SAMRAI::hier::Box const& sourceFieldBox,
                   SAMRAI::hier::IntVector const& ratio) const override
    {
        Fill::fill(sourceField, destinationField, intersectionBox, centering, destFieldBox,
                   sourceFieldBox, ratio);
    }

    int coarseStencilWidth() const override { return Fill::coarseStencilWidth(); }
};


// ---- factory (declared in field_refiner_kernel.hpp) ---------------------------------------------

// FieldRefinementOrder has a single enumerator and RefinementConfig::FROM is the only place a
// dict value is validated into it, so there is nothing to branch on. A second order adds a switch
// here, one case per compile-time stencil.
template<typename GridLayoutT, typename FieldT>
std::unique_ptr<IFieldRefineKernel<GridLayoutT, FieldT>>
makeRefineKernel([[maybe_unused]] FieldRefinementOrder const order,
                 FieldRefinementLimiter const limiter)
{
    constexpr auto linear = static_cast<std::size_t>(FieldRefinementOrder::Linear);
    if (limiter == FieldRefinementLimiter::WENO)
        return std::make_unique<CompositeFieldRefiner<GridLayoutT, FieldT, linear, WenoLimiter>>();
    return std::make_unique<CompositeFieldRefiner<GridLayoutT, FieldT, linear>>();
}


} // namespace PHARE::amr


#endif
