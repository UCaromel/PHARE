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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>


namespace PHARE::amr
{
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
     */
    template<typename GridLayoutT, typename FieldT, std::size_t order>
    struct CompositeFill
    {
        static constexpr std::size_t dimension = GridLayoutT::dimension;

        static_assert(order == 2 || order == 4,
                      "composite refiner supports orders 2 (Linear) and 4 (Cubic)");

        // The stencil reach is half the approximation order: ±1 for Linear and ±2 for Cubic.
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

                double const value = gatherers_[combined](sourceField, anchorLocal);

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

        // preserve the legacy NaN-guard: only fill fine indices not already set
        static void assignFine_(FieldT& dst, Point_t const& fineLocal, double const value)
        {
            if (auto& dst_val = dst(fineLocal); std::isnan(dst_val))
                dst_val = value;
        }
    };

    // EXP (exp/o4-refine-limit, do not merge): limited order-4 refinement of a cell-centred row
    // in 2D. child = u0 +- m per direction, m from a 5-point rule; the two sweep orders are
    // averaged. See refine_exp (field_refiner_kernel.hpp) for the knobs.
    template<typename FieldT>
    struct LimitedO4Fill
    {
        using Point_t = core::Point<int, 2>;
        using Block   = std::array<std::array<double, 5>, 5>; // [x offset + 2][y offset + 2]

        static double minmod4(double const a, double const b)
        {
            double const c1 = 4 * a - b, c2 = 4 * b - a;
            auto const sgn = [](double const x) { return (x > 0) - (x < 0); };
            int const sg   = sgn(a);
            if (sgn(b) != sg or sgn(c1) != sg or sgn(c2) != sg)
                return 0.;
            return sg
                   * std::min(std::min(std::abs(a), std::abs(b)),
                              std::min(std::abs(c1), std::abs(c2)));
        }

        // m = (upper child) - u0
        static double quarter(double const um2, double const um1, double const u0, double const u1,
                              double const u2)
        {
            double const sbig = (22. * (u1 - um1) - 3. * (u2 - um2)) / 32.;
            int const rule    = refine_exp::rule();
            if (rule == 3)
                return sbig / 4;
            if (rule == 1) // MP bounds at h/4, curvature factor 3/8
            {
                constexpr double c = 0.375;
                double const L = u0 - um1, R = u1 - u0;
                double const d0 = R - L, dm = u0 - 2 * um1 + um2, dp = u2 - 2 * u1 + u0;
                double const dP = minmod4(d0, dp), dM = minmod4(dm, d0);
                double const a3 = u0 + R / 4 - c * dP, b2 = u0 + L, b3 = u0 + L / 4 + c * dM;
                double const lo = std::max(std::min({u0, u1, a3}), std::min({u0, b2, b3}));
                double const hi = std::min(std::max({u0, u1, a3}), std::max({u0, b2, b3}));
                double v        = u0 + sbig / 4;
                if ((lo - v) * (hi - v) > 0)
                    v += std::abs(lo - v) < std::abs(hi - v) ? lo - v : hi - v;
                return v - u0;
            }
            // WENO-AO(5,3)-Z between the quartic slope and the three parabola slopes
            double const m1 = 41 * u1 / 60 - 11 * u2 / 120 - 41 * um1 / 60 + 11 * um2 / 120;
            double const m2
                = -37 * u0 / 28 + 5 * u1 / 7 - 3 * u2 / 56 + 5 * um1 / 7 - 3 * um2 / 56;
            double const m3 = -u1 / 6 + u2 / 12 + um1 / 6 - um2 / 12;
            double const t  = m1 + m3 / 10;
            double const ib = t * t + 13. / 3. * m2 * m2 + 781. / 20. * m3 * m3;
            std::array<double, 3> const sl{3 * u0 / 2 - 2 * um1 + um2 / 2, u1 / 2 - um1 / 2,
                                           -3 * u0 / 2 + 2 * u1 - u2 / 2};
            std::array<double, 3> const cv{u0 / 2 - um1 + um2 / 2, -u0 + u1 / 2 + um1 / 2,
                                           u0 / 2 - u1 + u2 / 2};
            std::array<double, 3> is{}, al{};
            for (int i = 0; i < 3; ++i)
                is[i] = sl[i] * sl[i] + 13. / 3. * cv[i] * cv[i];
            double const eps = 1e-28 * u0 * u0 + 1e-300;
            constexpr double g0 = 0.85, g = 0.05;
            double const tau
                = (std::abs(ib - is[0]) + std::abs(ib - is[1]) + std::abs(ib - is[2])) / 3;
            double const r0 = tau / (ib + eps);
            double const a0 = g0 * (1 + r0 * r0);
            double S        = a0, lin = 0, nl = 0;
            for (int i = 0; i < 3; ++i)
            {
                double const r = tau / (is[i] + eps);
                al[i]          = g * (1 + r * r);
                S += al[i];
                lin += sl[i];
                nl += al[i] * sl[i];
            }
            double const s = (a0 / S / g0) * (sbig - g * lin) + nl / S;
            return std::isfinite(s) ? s / 4 : 0.;
        }

        // child (px, py in {0, 1}) of the block's central cell, symmetrised sequential sweeps
        static double child(Block const& b, int const px, int const py)
        {
            double const sx = 2 * px - 1, sy = 2 * py - 1;
            std::array<double, 5> v{};
            for (int j = 0; j < 5; ++j)
                v[j] = b[2][j] + sx * quarter(b[0][j], b[1][j], b[2][j], b[3][j], b[4][j]);
            double const xy = v[2] + sy * quarter(v[0], v[1], v[2], v[3], v[4]);
            for (int i = 0; i < 5; ++i)
                v[i] = b[i][2] + sy * quarter(b[i][0], b[i][1], b[i][2], b[i][3], b[i][4]);
            double const yx = v[2] + sx * quarter(v[0], v[1], v[2], v[3], v[4]);
            return 0.5 * (xy + yx);
        }

        static std::array<int, 4> key(Point_t const& anchor, int const i, int const j)
        {
            return {anchor[0] + i - 2, anchor[1] + j - 2, 0, refine_exp::coarseKey};
        }

        static void fill(FieldT const& src, FieldT& dst, SAMRAI::hier::Box const& region,
                         SAMRAI::hier::Box const& destFieldBox,
                         SAMRAI::hier::Box const& sourceFieldBox)
        {
            using refine_exp::Qty;
            int const slot  = refine_exp::qty == Qty::Rho ? 0 : 1 + refine_exp::comp;
            bool const isE  = refine_exp::qty == Qty::Etot;
            bool const eRow = refine_exp::energy();

            for (auto const fineIndex : phare_box_from<2>(region))
            {
                auto const anchor      = toCoarseIndex<2>(fineIndex);
                auto const anchorLocal = AMRToLocal(anchor, sourceFieldBox);
                auto const fineLocal   = AMRToLocal(fineIndex, destFieldBox);
                int const px = fineIndex[0] - 2 * anchor[0], py = fineIndex[1] - 2 * anchor[1];

                Block u;
                for (int i = 0; i < 5; ++i)
                    for (int j = 0; j < 5; ++j)
                        u[i][j] = src(Point_t{anchorLocal[0] + i - 2, anchorLocal[1] + j - 2});

                if (eRow and not isE) // store this row's coarse block for the Etot fill
                {
                    auto& centre = refine_exp::ecache[key(anchor, 2, 2)];
                    if (not(centre.blockDone & (1u << slot)))
                    {
                        centre.blockDone |= 1u << slot;
                        for (int i = 0; i < 5; ++i)
                            for (int j = 0; j < 5; ++j)
                            {
                                auto& c   = refine_exp::ecache[key(anchor, i, j)];
                                c.v[slot] = u[i][j];
                                c.have |= 1u << slot;
                            }
                    }
                }

                double value = 0;
                bool done    = false;
                if (eRow and isE)
                {
                    Block rho, q;
                    std::array<Block, 3> m;
                    bool ok = true;
                    for (int i = 0; i < 5 and ok; ++i)
                        for (int j = 0; j < 5 and ok; ++j)
                        {
                            auto const it = refine_exp::ecache.find(key(anchor, i, j));
                            if (it == refine_exp::ecache.end() or it->second.have != 0xFu)
                            {
                                ok = false;
                                break;
                            }
                            auto const& c = it->second.v;
                            rho[i][j]     = c[0];
                            for (int k = 0; k < 3; ++k)
                                m[k][i][j] = c[k + 1];
                            q[i][j] = u[i][j] - 0.5 * (c[1] * c[1] + c[2] * c[2] + c[3] * c[3]) / c[0];
                        }
                    if (ok)
                    {
                        double keSum = 0, keMine = 0;
                        for (int sx = 0; sx < 2; ++sx)
                            for (int sy = 0; sy < 2; ++sy)
                            {
                                double m2 = 0;
                                for (int k = 0; k < 3; ++k)
                                {
                                    double const mk = child(m[k], sx, sy);
                                    m2 += mk * mk;
                                }
                                double const ke = 0.5 * m2 / child(rho, sx, sy);
                                keSum += ke;
                                if (sx == px and sy == py)
                                    keMine = ke;
                            }
                        value = u[2][2] + (child(q, px, py) - q[2][2]) + keMine - 0.25 * keSum;
                        done  = true;
                        ++refine_exp::eApplied;
                    }
                    else
                        ++refine_exp::eMissed;
                }
                if (not done)
                    value = child(u, px, py);
                ++refine_exp::nLimited;

                if (auto& dst_val = dst(fineLocal); std::isnan(dst_val))
                    dst_val = value;
            }
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
template<typename GridLayoutT, typename FieldT, std::size_t order>
class CompositeFieldRefiner final : public IFieldRefineKernel<GridLayoutT, FieldT>
{
    static constexpr std::size_t dimension = GridLayoutT::dimension;

    using Fill = detail::CompositeFill<GridLayoutT, FieldT, order>;

public:
    void refineBox(FieldT const& sourceField, FieldT& destinationField,
                   SAMRAI::hier::Box const& intersectionBox,
                   std::array<core::QtyCentering, dimension> const& centering,
                   SAMRAI::hier::Box const& destFieldBox, SAMRAI::hier::Box const& sourceFieldBox,
                   SAMRAI::hier::IntVector const& ratio) const override
    {
        if constexpr (order == 4 and dimension == 2)
            if (refine_exp::active() and centering[0] == core::QtyCentering::dual
                and centering[1] == core::QtyCentering::dual)
            {
                detail::LimitedO4Fill<FieldT>::fill(sourceField, destinationField, intersectionBox,
                                                    destFieldBox, sourceFieldBox);
                return;
            }
        Fill::fill(sourceField, destinationField, intersectionBox, centering, destFieldBox,
                   sourceFieldBox, ratio);
    }

    int coarseStencilWidth() const override { return Fill::coarseStencilWidth(); }
};


// ---- factory (declared in field_refiner_kernel.hpp) ---------------------------------------------

template<typename GridLayoutT, typename FieldT>
std::unique_ptr<IFieldRefineKernel<GridLayoutT, FieldT>>
makeRefineKernel(FieldRefinementOrder const order)
{
    switch (order)
    {
        case FieldRefinementOrder::Linear:
            return std::make_unique<CompositeFieldRefiner<
                GridLayoutT, FieldT, static_cast<std::size_t>(FieldRefinementOrder::Linear)>>();
        case FieldRefinementOrder::Cubic:
            return std::make_unique<CompositeFieldRefiner<
                GridLayoutT, FieldT, static_cast<std::size_t>(FieldRefinementOrder::Cubic)>>();
    }

    throw std::logic_error("invalid field-refinement order");
}


} // namespace PHARE::amr


#endif
