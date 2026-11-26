#ifndef PHARE_LIMITED_FIELD_REFINER_HPP
#define PHARE_LIMITED_FIELD_REFINER_HPP


#include "phare_mpi.hpp" // IWYU pragma: keep

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/numerics/refinement/slope_rules.hpp"
#include "core/utilities/box/box.hpp"
#include "core/utilities/box/box_spans.hpp"
#include "core/utilities/point/point.hpp"

#include "amr/resources_manager/amr_utils.hpp"
#include "amr/utilities/box/amr_box.hpp"
#include "coarse_cell_round_out.hpp"
#include "composite_field_refiner.hpp"
#include "field_refiner_kernel.hpp"
#include "refine_probe.hpp"

#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/IntVector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>


namespace PHARE::amr
{

/**
 * @brief Call `fn` with `std::type_identity<Rule>` for the slope rule of `limiter`.
 *
 * The one place mapping the runtime choice to a rule type: a new rule is a policy type in
 * slope_rules.hpp, an enumerator, and a case here.
 */
template<typename Fn>
decltype(auto) withRule(FieldRefinementLimiter const limiter, Fn&& fn)
{
    using namespace core::refinement;
    using enum FieldRefinementLimiter;

    switch (limiter)
    {
        case None: return fn(std::type_identity<CentralRule>{});
        case VanLeer: return fn(std::type_identity<VanLeerRule>{});
        case MC: return fn(std::type_identity<MCRule>{});
        case MCGate: return fn(std::type_identity<MCGateRule>{});
        case AO5Z: return fn(std::type_identity<AO5ZRule>{});
        case MP: return fn(std::type_identity<MPRule>{});
    }
    throw std::logic_error("unknown FieldRefinementLimiter");
}


//! Coarse stencil half-width read by the rule of `settings` (1 for master's table).
inline int limiterStencilWidth(LimiterSettings const& settings)
{
    return withRule(settings.limiter,
                    []<typename Rule>(std::type_identity<Rule>) -> int { return Rule::width; });
}


namespace detail
{
    //! Values on a box of AMR indices, NaN until written.
    template<std::size_t dim>
    struct BoxValues
    {
        using Point_t = core::Point<int, dim>;

        explicit BoxValues(core::Box<int, dim> const& box_)
            : box{box_}
            , values{shape_(box_), std::numeric_limits<double>::quiet_NaN()}
        {
        }

        core::Box<int, dim> box;
        core::NdArrayVector<dim, double> values;

        NO_DISCARD double& operator()(Point_t const& amr) { return values(local_(amr)); }
        NO_DISCARD double operator()(Point_t const& amr) const { return values(local_(amr)); }

        NO_DISCARD bool empty() const
        {
            for (std::size_t d = 0; d < dim; ++d)
                if (box.upper[d] < box.lower[d])
                    return true;
            return false;
        }

    private:
        static std::array<std::uint32_t, dim> shape_(core::Box<int, dim> const& b)
        {
            std::array<std::uint32_t, dim> shape{};
            for (std::size_t d = 0; d < dim; ++d)
                shape[d] = b.upper[d] < b.lower[d]
                               ? 0u
                               : static_cast<std::uint32_t>(b.upper[d] - b.lower[d] + 1);
            return shape;
        }

        std::array<std::uint32_t, dim> local_(Point_t const& amr) const
        {
            std::array<std::uint32_t, dim> local{};
            for (std::size_t d = 0; d < dim; ++d)
                local[d] = static_cast<std::uint32_t>(amr[d] - box.lower[d]);
            return local;
        }
    };

    //! Children of a fine region, before they are written to a field.
    template<std::size_t dim>
    using FineBlock = BoxValues<dim>;


    /**
     * @brief The limited refinement fill: master's composite stencil with each dual row's slope
     * chosen by `Rule`.
     *
     * Along a dual direction d, at the coarse anchor I of a fine index, the rule reads the line
     * u(I + k e_d), k = -width..width, and returns a slope s_d from it and the unlimited slope
     * c_d = (u(I + e_d) - u(I - e_d)) / 2.
     *
     * Product mode adds the rule's correction of each dual direction to master's unlimited
     * tensor-product value (all cross terms):
     *   child = master_child + sum_{dual d} sigma_d (s_d - c_d) / 4,
     * sigma_d = -1 for the even (left) child, +1 for the odd (right) one, s_d and c_d from the
     * line through the anchor along d. Where the fine index lies between the coarse nodes I and
     * I+1 of a primal direction p, the correction of d is the mean of the two node lines',
     * (s_d - c_d)(I) and (s_d - c_d)(I + e_p); with several such directions, the mean over the
     * 2^m node lines. The correction sums to zero over the children of a coarse cell, is bounded
     * by the slopes for every rule and continuous in the data (no division). In 1D the child is
     * u0 + sigma s / 4. With CentralRule s = c: master.
     *
     * Sweeps mode refines one direction at a time. For a direction order (d1, d2[, d3]) it
     * refines along d1 every coarse line of the anchors grown by the stencil in the other
     * directions (dual: children u0 + sigma s / 4 with s the rule's slope of that line; primal:
     * copy of node I, midpoint of I and I+1), then along d2 the lines of that half-refined block,
     * one per fine index of d1, then along d3. Slopes of later sweeps are taken from already
     * refined values. The result is the mean over the direction orders of the option set. Every
     * sweep conserves, so does the mean; with CentralRule every sweep is linear and the result
     * is master's tensor product.
     */
    template<std::size_t dim, std::size_t order, core::refinement::SlopeRule Rule>
    struct LimitedFill
    {
        static_assert(order == 2, "limited refinement is order 2 (Linear)");

        static constexpr int width = Rule::width;

        static constexpr int coarseStencilWidth() { return Rule::width; }

        struct Options
        {
            RefinementTensorMode tensor       = RefinementTensorMode::Product;
            RefinementSweepOrders sweepOrders = RefinementSweepOrders::All;
            double param                      = Rule::defaultParam;
        };

        /**
         * @brief Children on `fineRegion` (AMR fine indices of the quantity's centering) from
         * `coarse`, a callable Point<int, dim> (AMR coarse index) -> double defined on the anchors
         * of the region grown by `width`.
         *
         * A non-finite value read beyond master's reach (|k| > 1) throws: a rule never falls back
         * silently to a narrower one. Within |k| <= 1 a NaN propagates into the child as in master.
         */
        template<typename CoarseFn>
        static FineBlock<dim> refineRegion(CoarseFn const& coarse,
                                           core::Box<int, dim> const& fineRegion,
                                           std::array<core::QtyCentering, dim> const& centering,
                                           Options const& options, RefineProbe* probe)
        {
            if (options.tensor == RefinementTensorMode::Sweeps)
                return sweeps_(coarse, fineRegion, centering, options, probe);

            return product_(coarse, fineRegion, centering, options.param, probe);
        }

        /**
         * @brief refineRegion on `region`, then write the fine indices still NaN, anchored as
         * CompositeFill::fill.
         */
        template<typename FieldT>
        static void
        fill(FieldT const& sourceField, FieldT& destinationField, SAMRAI::hier::Box const& region,
             std::array<core::QtyCentering, dim> const& centering,
             SAMRAI::hier::Box const& destFieldBox, SAMRAI::hier::Box const& sourceFieldBox,
             SAMRAI::hier::IntVector const& ratio, Options const& options, RefineProbe* probe)
        {
            for (std::size_t d = 0; d < dim; ++d)
                if (ratio(d) != 2)
                    throw std::runtime_error(
                        "limited field refinement supports refinement ratio 2 only");

            if (region.empty())
                return;

            auto const sourceBox = phare_box_from<dim>(sourceFieldBox);
            auto const coarse    = [&](core::Point<int, dim> const& coarseAMR) -> double {
                if (!core::isIn(coarseAMR, sourceBox))
                    throw std::runtime_error("limited field refinement reads coarse index "
                                                + pointStr_(coarseAMR)
                                                + " outside the source field box");
                return sourceField(AMRToLocal(coarseAMR, sourceFieldBox));
            };

            auto const block
                = refineRegion(coarse, phare_box_from<dim>(region), centering, options, probe);

            core::for_each_point(block.box, [&](auto const& fineIndex) {
                if (auto& dst = destinationField(AMRToLocal(fineIndex, destFieldBox));
                    std::isnan(dst))
                    dst = block(fineIndex);
            });
        }

    private:
        using Point_t = core::Point<int, dim>;

        /**
         * @brief The anchors whose lines a region reads: the anchors of the region, and
         * in a primal direction also node I+1 of the last anchor when the region ends
         * between nodes.
         */
        static core::Box<int, dim>
        lineAnchors_(core::Box<int, dim> const& fineRegion,
                     std::array<core::QtyCentering, dim> const& centering)
        {
            core::Box<int, dim> box{toCoarseIndex<dim>(fineRegion.lower),
                                    toCoarseIndex<dim>(fineRegion.upper)};
            for (std::size_t d = 0; d < dim; ++d)
                if (centering[d] == core::QtyCentering::primal and isOddIndex(fineRegion.upper[d]))
                    box.upper[d] += 1;
            return box;
        }

        template<typename CoarseFn>
        static FineBlock<dim> product_(CoarseFn const& coarse,
                                       core::Box<int, dim> const& fineRegion,
                                       std::array<core::QtyCentering, dim> const& centering,
                                       double const param, RefineProbe* probe)
        {
            FineBlock<dim> block{fineRegion};
            if (block.empty())
                return block;

            // s_d - c_d of every line the region reads, dual directions only
            auto const lines = lineAnchors_(fineRegion, centering);
            std::array<std::optional<BoxValues<dim>>, dim> excess;
            for (std::size_t d = 0; d < dim; ++d)
            {
                if (centering[d] != core::QtyCentering::dual)
                    continue;

                auto& excessD = excess[d].emplace(lines);
                core::for_each_point(lines, [&](auto const& a) {
                    Point_t const anchor{a};
                    std::array<double, 2 * width + 1> line;
                    Point_t at{anchor};
                    for (int k = -width; k <= width; ++k)
                    {
                        at[d]          = anchor[d] + k;
                        double const u = coarse(at);
                        if (std::abs(k) > 1 and !std::isfinite(u))
                            throw nonFinite_(anchor, d, k, fineRegion, centering);
                        line[k + width] = u;
                    }

                    double const c = (line[width + 1] - line[width - 1]) / 2.;
                    double const s = Rule::slope(line, c, param);
                    if (probe)
                        probe->row(s, c);

                    excessD(anchor) = s - c;
                });
            }

            core::for_each_point(fineRegion, [&](auto const& fineIndex) {
                auto const anchor = toCoarseIndex<dim>(Point_t{fineIndex});

                // master's 1D rows: offsets and weights around the anchor
                std::array<std::array<int, 3>, dim> offsets{};
                std::array<std::array<double, 3>, dim> weights{};
                std::array<std::size_t, dim> sizes{};
                std::array<double, dim> sigma{};
                std::array<std::size_t, dim> betweenNodes{}; // primal-odd directions
                std::size_t nBetween = 0;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    bool const odd = isOddIndex(fineIndex[d]);
                    if (centering[d] == core::QtyCentering::primal)
                    {
                        if (odd)
                        {
                            offsets[d]               = {0, 1, 0};
                            weights[d]               = {0.5, 0.5, 0.};
                            sizes[d]                 = 2;
                            betweenNodes[nBetween++] = d;
                        }
                        else
                        {
                            offsets[d] = {0, 0, 0};
                            weights[d] = {1., 0., 0.};
                            sizes[d]   = 1;
                        }
                    }
                    else
                    {
                        sigma[d]   = odd ? 1. : -1.;
                        offsets[d] = {-1, 0, 1};
                        weights[d] = {-sigma[d] / 8., 1., sigma[d] / 8.};
                        sizes[d]   = 3;
                    }
                }

                // master's value: tensor product of the rows
                double value = 0.;
                std::array<std::size_t, dim> j{};
                while (true)
                {
                    double w = 1.;
                    Point_t at{anchor};
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        w *= weights[d][j[d]];
                        at[d] += offsets[d][j[d]];
                    }
                    value += w * coarse(at);

                    std::size_t d = 0;
                    for (; d < dim; ++d)
                    {
                        if (++j[d] < sizes[d])
                            break;
                        j[d] = 0;
                    }
                    if (d == dim)
                        break;
                }

                // the rule's correction of each dual direction, mean over the node lines
                // I .. I+1 of the primal directions where the index lies between nodes
                std::size_t const nCorners = std::size_t{1} << nBetween;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    if (centering[d] != core::QtyCentering::dual)
                        continue;

                    double excessMean = 0.;
                    for (std::size_t corner = 0; corner < nCorners; ++corner)
                    {
                        Point_t at{anchor};
                        for (std::size_t b = 0; b < nBetween; ++b)
                            at[betweenNodes[b]] += static_cast<int>((corner >> b) & 1u);
                        excessMean += (*excess[d])(at);
                    }
                    value += sigma[d] * (excessMean / static_cast<double>(nCorners)) / 4.;
                }

                block(Point_t{fineIndex}) = value;
            });

            return block;
        }

        //! direction orders averaged by sweeps mode
        static std::vector<std::array<std::size_t, dim>>
        directionOrders_(RefinementSweepOrders const orders)
        {
            std::array<std::size_t, dim> first{};
            std::iota(first.begin(), first.end(), std::size_t{0});

            std::vector<std::array<std::size_t, dim>> result{first};
            if (orders == RefinementSweepOrders::Single)
                return result;

            if (orders == RefinementSweepOrders::Cyclic)
            {
                for (std::size_t shift = 1; shift < dim; ++shift)
                {
                    auto sequence = first;
                    std::rotate(sequence.begin(), sequence.begin() + shift, sequence.end());
                    result.push_back(sequence);
                }
                return result;
            }

            auto sequence = first;
            while (std::next_permutation(sequence.begin(), sequence.end()))
                result.push_back(sequence);
            return result;
        }


        template<typename CoarseFn>
        static FineBlock<dim> sweeps_(CoarseFn const& coarse, core::Box<int, dim> const& fineRegion,
                                      std::array<core::QtyCentering, dim> const& centering,
                                      Options const& options, RefineProbe* probe)
        {
            FineBlock<dim> result{fineRegion};
            if (result.empty())
                return result;

            // every coarse value a sweep reads, whatever the order: per direction, the anchors
            // grown by the rule's width (dual) or the nodes I .. I+1 of the anchors (primal)
            core::Box<int, dim> coarseBox = lineAnchors_(fineRegion, centering);
            for (std::size_t d = 0; d < dim; ++d)
                if (centering[d] == core::QtyCentering::dual)
                {
                    coarseBox.lower[d] -= width;
                    coarseBox.upper[d] += width;
                }

            BoxValues<dim> coarseBlock{coarseBox};
            core::for_each_point(
                coarseBox, [&](auto const& I) { coarseBlock(Point_t{I}) = coarse(Point_t{I}); });

            auto const orders = directionOrders_(options.sweepOrders);
            core::for_each_point(fineRegion, [&](auto const& f) { result(Point_t{f}) = 0.; });

            for (auto const& sequence : orders)
            {
                std::array<bool, dim> refined{};
                FineBlock<dim> block = coarseBlock;
                for (std::size_t const dir : sequence)
                {
                    block = sweepAlong_(block, dir, fineRegion, centering, refined, options.param,
                                        probe);
                    refined[dir] = true;
                }
                core::for_each_point(
                    fineRegion, [&](auto const& f) { result(Point_t{f}) += block(Point_t{f}); });
            }

            double const nOrders = static_cast<double>(orders.size());
            core::for_each_point(fineRegion, [&](auto const& f) { result(Point_t{f}) /= nOrders; });

            return result;
        }


        /**
         * @brief One 1D refinement along `dir` of every line of `in`, whose indices are fine in
         * the directions `refined` and coarse in the others. The output spans the fine region
         * along `dir`.
         */
        static FineBlock<dim> sweepAlong_(BoxValues<dim> const& in, std::size_t const dir,
                                          core::Box<int, dim> const& fineRegion,
                                          std::array<core::QtyCentering, dim> const& centering,
                                          std::array<bool, dim> const& refined, double const param,
                                          RefineProbe* probe)
        {
            core::Box<int, dim> outBox{in.box};
            outBox.lower[dir] = fineRegion.lower[dir];
            outBox.upper[dir] = fineRegion.upper[dir];
            FineBlock<dim> out{outBox};

            if (centering[dir] == core::QtyCentering::primal)
            {
                core::for_each_point(outBox, [&](auto const& f) {
                    Point_t node{f};
                    node[dir] = toCoarseIndex<dim>(Point_t{f})[dir];
                    if (!isOddIndex(f[dir]))
                        out(Point_t{f}) = in(node);
                    else
                    {
                        Point_t next{node};
                        next[dir] += 1;
                        out(Point_t{f}) = 0.5 * (in(node) + in(next));
                    }
                });
                return out;
            }

            // dual: one rule evaluation per line and anchor, writing the children in the region
            core::Box<int, dim> lineAnchors{outBox};
            lineAnchors.lower[dir] = in.box.lower[dir] + width;
            lineAnchors.upper[dir] = in.box.upper[dir] - width;

            core::for_each_point(lineAnchors, [&](auto const& a) {
                Point_t const anchor{a};
                std::array<double, 2 * width + 1> line;
                Point_t at{anchor};
                for (int k = -width; k <= width; ++k)
                {
                    at[dir]        = anchor[dir] + k;
                    double const u = in(at);
                    if (std::abs(k) > 1 and !std::isfinite(u))
                        throw nonFinite_(coarseAnchorOf_(anchor, refined), dir, k, fineRegion,
                                         centering);
                    line[k + width] = u;
                }

                // a non-finite neighbour propagates into the children, as in master's stencil,
                // whatever slope the rule would return
                bool const finiteNeighbours
                    = std::isfinite(line[width - 1]) and std::isfinite(line[width + 1]);
                double const c = (line[width + 1] - line[width - 1]) / 2.;
                double const s = finiteNeighbours ? Rule::slope(line, c, param)
                                                  : std::numeric_limits<double>::quiet_NaN();
                if (probe)
                    probe->row(s, c);

                Point_t child{anchor};
                for (int const sigma : {-1, 1})
                {
                    child[dir] = 2 * anchor[dir] + (sigma + 1) / 2;
                    if (child[dir] >= fineRegion.lower[dir] and child[dir] <= fineRegion.upper[dir])
                        out(child) = line[width] + sigma * s / 4.;
                }
            });

            return out;
        }

        //! the coarse anchor of a point of a partly refined block, for error messages
        static Point_t coarseAnchorOf_(Point_t const& p, std::array<bool, dim> const& refined)
        {
            auto const coarsened = toCoarseIndex<dim>(p);
            Point_t anchor{p};
            for (std::size_t d = 0; d < dim; ++d)
                if (refined[d])
                    anchor[d] = coarsened[d];
            return anchor;
        }

        static std::string pointStr_(Point_t const& p)
        {
            std::ostringstream os;
            os << "(";
            for (std::size_t d = 0; d < dim; ++d)
                os << p[d] << (d + 1 < dim ? "," : ")");
            return os.str();
        }

        static std::runtime_error nonFinite_(Point_t const& anchor, std::size_t const dir,
                                             int const offset,
                                             core::Box<int, dim> const& fineRegion,
                                             std::array<core::QtyCentering, dim> const& centering)
        {
            // first fine index of the region refined from this anchor
            Point_t fine{};
            for (std::size_t d = 0; d < dim; ++d)
                fine[d]
                    = std::min(std::max(2 * anchor[d], fineRegion.lower[d]), fineRegion.upper[d]);

            std::ostringstream os;
            os << "limited field refinement: non-finite coarse value at offset " << offset
               << " along direction " << dir << " of anchor " << pointStr_(anchor)
               << " (fine index " << pointStr_(fine) << ", centering (";
            for (std::size_t d = 0; d < dim; ++d)
                os << (centering[d] == core::QtyCentering::primal ? "primal" : "dual")
                   << (d + 1 < dim ? "," : "))");
            os << "; the rule reads " << width << " coarse cells and does not fall back";
            return std::runtime_error(os.str());
        }
    };

} // namespace detail


/**
 * @brief Runtime field-refinement kernel: the limited fill of `Rule` over exactly the overlap
 * SAMRAI hands it.
 */
template<typename GridLayoutT, typename FieldT, std::size_t order, core::refinement::SlopeRule Rule>
class LimitedFieldRefiner final : public IFieldRefineKernel<GridLayoutT, FieldT>
{
    static constexpr std::size_t dimension = GridLayoutT::dimension;

    using Fill = detail::LimitedFill<dimension, order, Rule>;

public:
    LimitedFieldRefiner(LimiterSettings const& settings, std::shared_ptr<RefineProbe> probe)
        : options_{.tensor      = settings.tensor,
                   .sweepOrders = settings.sweepOrders,
                   .param       = settings.param.value_or(Rule::defaultParam)}
        , probe_{std::move(probe)}
    {
    }

    void refineBox(FieldT const& sourceField, FieldT& destinationField,
                   SAMRAI::hier::Box const& intersectionBox,
                   std::array<core::QtyCentering, dimension> const& centering,
                   SAMRAI::hier::Box const& destFieldBox, SAMRAI::hier::Box const& sourceFieldBox,
                   SAMRAI::hier::IntVector const& ratio) const override
    {
        Fill::fill(sourceField, destinationField, intersectionBox, centering, destFieldBox,
                   sourceFieldBox, ratio, options_, probe_.get());
    }

    int coarseStencilWidth() const override { return Fill::coarseStencilWidth(); }

private:
    typename Fill::Options options_;
    std::shared_ptr<RefineProbe> probe_;
};


// ---- factory (declared in field_refiner_kernel.hpp) ---------------------------------------------

// FieldRefinementOrder has a single enumerator and RefinementConfig::FROM is the only place a
// dict value is validated into it, so there is nothing to branch on. Master's settings select the
// compile-time table (bitwise master); any other selects the limited kernel of the rule.
template<typename GridLayoutT, typename FieldT>
std::unique_ptr<IFieldRefineKernel<GridLayoutT, FieldT>>
makeRefineKernel([[maybe_unused]] FieldRefinementOrder const order, LimiterSettings const& settings,
                 std::shared_ptr<RefineProbe> probe)
{
    using Kernel_ptr             = std::unique_ptr<IFieldRefineKernel<GridLayoutT, FieldT>>;
    constexpr std::size_t linear = static_cast<std::size_t>(FieldRefinementOrder::Linear);

    if (settings.isMasterTable())
        return std::make_unique<CompositeFieldRefiner<GridLayoutT, FieldT, linear>>();

    return withRule(settings.limiter, [&]<typename Rule>(std::type_identity<Rule>) -> Kernel_ptr {
        return std::make_unique<LimitedFieldRefiner<GridLayoutT, FieldT, linear, Rule>>(
            settings, std::move(probe));
    });
}


} // namespace PHARE::amr


#endif
