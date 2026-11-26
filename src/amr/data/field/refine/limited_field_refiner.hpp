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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>


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
     * Product mode keeps master's tensor product and scales each dual row by phi_d:
     *   row_d = {-sigma phi_d / 8, 1, +sigma phi_d / 8} at offsets -1, 0, +1,
     *   phi_d = 1 where s_d == c_d, else s_d / c_d, and 0 where c_d == 0,
     * sigma = -1 for the even (left) child, +1 for the odd (right) one. Primal rows are master's
     * (copy of node I, midpoint of I and I+1). phi_d is taken from the anchor's line only, so the
     * cross terms of master's stencil are kept with frozen weights. In 1D the child is u0 +
     * sigma s / 4, and every rule conserves the coarse average. With CentralRule phi = 1: master.
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
                throw std::runtime_error("refinement tensor mode 'sweeps' is not implemented yet");

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

        template<typename CoarseFn>
        static FineBlock<dim> product_(CoarseFn const& coarse,
                                       core::Box<int, dim> const& fineRegion,
                                       std::array<core::QtyCentering, dim> const& centering,
                                       double const param, RefineProbe* probe)
        {
            FineBlock<dim> block{fineRegion};
            if (block.empty())
                return block;

            core::Box<int, dim> const anchors{toCoarseIndex<dim>(fineRegion.lower),
                                              toCoarseIndex<dim>(fineRegion.upper)};

            // phi_d of every anchor, dual directions only
            std::array<std::optional<BoxValues<dim>>, dim> phi;
            for (std::size_t d = 0; d < dim; ++d)
            {
                if (centering[d] != core::QtyCentering::dual)
                    continue;

                auto& phiD = phi[d].emplace(anchors);
                core::for_each_point(anchors, [&](auto const& anchor) {
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

                    phiD(anchor) = s == c ? 1. : (c == 0. ? 0. : s / c);
                });
            }

            core::for_each_point(fineRegion, [&](auto const& fineIndex) {
                auto const anchor = toCoarseIndex<dim>(Point_t{fineIndex});

                // the 1D row of each direction: offsets and weights around the anchor
                std::array<std::array<int, 3>, dim> offsets{};
                std::array<std::array<double, 3>, dim> weights{};
                std::array<std::size_t, dim> sizes{};
                for (std::size_t d = 0; d < dim; ++d)
                {
                    bool const odd = isOddIndex(fineIndex[d]);
                    if (centering[d] == core::QtyCentering::primal)
                    {
                        if (odd)
                        {
                            offsets[d] = {0, 1, 0};
                            weights[d] = {0.5, 0.5, 0.};
                            sizes[d]   = 2;
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
                        double const sigma = odd ? 1. : -1.;
                        double const h     = sigma * (*phi[d])(anchor) / 8.;
                        offsets[d]         = {-1, 0, 1};
                        weights[d]         = {-h, 1., h};
                        sizes[d]           = 3;
                    }
                }

                // tensor product of the rows
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

                block(Point_t{fineIndex}) = value;
            });

            return block;
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
