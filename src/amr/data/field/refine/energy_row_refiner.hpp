#ifndef PHARE_ENERGY_ROW_REFINER_HPP
#define PHARE_ENERGY_ROW_REFINER_HPP


#include "phare_mpi.hpp" // IWYU pragma: keep

#include "core/def.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/numerics/refinement/slope_rules.hpp"
#include "core/utilities/box/box.hpp"
#include "core/utilities/box/box_spans.hpp"
#include "core/utilities/point/point.hpp"

#include "amr/resources_manager/amr_utils.hpp"
#include "amr/utilities/box/amr_box.hpp"
#include "coarse_cell_round_out.hpp"
#include "field_refiner_kernel.hpp"
#include "limited_field_refiner.hpp"
#include "refine_probe.hpp"

#include <SAMRAI/hier/Box.h>
#include <SAMRAI/hier/IntVector.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>


namespace PHARE::amr
{

/**
 * @brief Total-energy refinement seam: the fine Etot from the coarse Etot, density and momentum.
 *
 * All four quantities are cell-centred with the same ghost box, so one source box serves them.
 */
template<typename GridLayoutT, typename FieldT>
struct IEnergyRowKernel
{
    static constexpr std::size_t dimension = GridLayoutT::dimension;

    virtual void
    refineBox(FieldT const& coarseRho, std::array<FieldT const*, 3> const& coarseRhoV,
              FieldT const& coarseEtot, FieldT& fineEtot, SAMRAI::hier::Box const& intersectionBox,
              SAMRAI::hier::Box const& destFieldBox, SAMRAI::hier::Box const& sourceFieldBox,
              SAMRAI::hier::IntVector const& ratio) const
        = 0;

    virtual int coarseStencilWidth() const = 0;

    virtual ~IEnergyRowKernel() = default;
};


/**
 * @brief Etot refined as q = Etot - |rhoV|^2 / (2 rho) plus the kinetic energy of the refined
 * density and momentum.
 *
 * rho, rhoV and q are refined with the limited fill of `Rule` (limited_field_refiner.hpp), and the
 * child k of coarse cell I gets
 *   E_k = E_I + (q_k - q_I) + KE_k - mean_siblings KE,   KE_k = |m_k|^2 / (2 rho_k),
 * the siblings being the 2^dim children of I. The children mean back to E_I whenever the rule
 * conserves q, and the internal energy of each child follows q's profile instead of the
 * difference of two independently refined profiles.
 *
 * rho and rhoV are recomputed here from coarse data, the same way their own operators compute
 * them, so a child set is always complete and self-consistent: the region is rounded out to whole
 * coarse cells before refining, and only the fine cells of the overlap still unset are written.
 */
template<typename GridLayoutT, typename FieldT, std::size_t order, core::refinement::SlopeRule Rule>
class EnergyRowRefiner final : public IEnergyRowKernel<GridLayoutT, FieldT>
{
    static constexpr std::size_t dimension = GridLayoutT::dimension;
    static constexpr std::size_t nChildren = std::size_t{1} << dimension;

    using Fill    = detail::LimitedFill<dimension, order, Rule>;
    using Point_t = core::Point<int, dimension>;
    using Box_t   = core::Box<int, dimension>;

public:
    EnergyRowRefiner(LimiterSettings const& settings, std::shared_ptr<RefineProbe> probe)
        : options_{.tensor      = settings.tensor,
                   .sweepOrders = settings.sweepOrders,
                   .param       = settings.param.value_or(Rule::defaultParam)}
        , probe_{std::move(probe)}
    {
    }

    void refineBox(FieldT const& coarseRho, std::array<FieldT const*, 3> const& coarseRhoV,
                   FieldT const& coarseEtot, FieldT& fineEtot,
                   SAMRAI::hier::Box const& intersectionBox, SAMRAI::hier::Box const& destFieldBox,
                   SAMRAI::hier::Box const& sourceFieldBox,
                   SAMRAI::hier::IntVector const& ratio) const override
    {
        for (std::size_t d = 0; d < dimension; ++d)
            if (ratio(d) != 2)
                throw std::runtime_error("energy-row refinement supports refinement ratio 2 only");

        if (intersectionBox.empty())
            return;

        std::array<core::QtyCentering, dimension> centering;
        centering.fill(core::QtyCentering::dual);

        auto const sourceBox = phare_box_from<dimension>(sourceFieldBox);
        auto const reader    = [&](FieldT const& field) {
            return [&field, &sourceBox, &sourceFieldBox](Point_t const& coarseAMR) -> double {
                if (!core::isIn(coarseAMR, sourceBox))
                    throw std::runtime_error(
                        "energy-row refinement reads a coarse index outside the source field box");
                return field(AMRToLocal(coarseAMR, sourceFieldBox));
            };
        };

        auto const rho  = reader(coarseRho);
        auto const mx   = reader(*coarseRhoV[0]);
        auto const my   = reader(*coarseRhoV[1]);
        auto const mz   = reader(*coarseRhoV[2]);
        auto const Etot = reader(coarseEtot);
        auto const q
            = [&](Point_t const& I) { return Etot(I) - kinetic_(rho(I), mx(I), my(I), mz(I)); };

        // whole coarse cells: the anchors, hence the coarse reach, are those of the overlap
        auto const region = phare_box_from<dimension>(
            roundFieldBoxOutToCoarseCells<dimension>(intersectionBox, centering));
        auto const overlap = phare_box_from<dimension>(intersectionBox);

        auto* probe     = probe_.get();
        auto const rhoF = Fill::refineRegion(rho, region, centering, options_, probe);
        auto const mxF  = Fill::refineRegion(mx, region, centering, options_, probe);
        auto const myF  = Fill::refineRegion(my, region, centering, options_, probe);
        auto const mzF  = Fill::refineRegion(mz, region, centering, options_, probe);
        auto const qF   = Fill::refineRegion(q, region, centering, options_, probe);

        Box_t const anchors{toCoarseIndex<dimension>(region.lower),
                            toCoarseIndex<dimension>(region.upper)};

        core::for_each_point(anchors, [&](auto const& anchor) {
            Point_t const I{anchor};
            Box_t const children{I * 2, I * 2 + 1};

            std::array<double, nChildren> ke{};
            double meanKE = 0.;
            std::size_t k = 0;
            core::for_each_point(children, [&](auto const& child) {
                Point_t const f{child};
                ke[k] = kinetic_(rhoF(f), mxF(f), myF(f), mzF(f));
                meanKE += ke[k++];
            });
            meanKE /= static_cast<double>(nChildren);

            double const E0 = Etot(I);
            double const q0 = q(I);

            k = 0;
            core::for_each_point(children, [&](auto const& child) {
                Point_t const f{child};
                double const keChild = ke[k++];
                if (!core::isIn(f, overlap))
                    return;
                if (auto& dst = fineEtot(AMRToLocal(f, destFieldBox)); std::isnan(dst))
                    dst = E0 + (qF(f) - q0) + keChild - meanKE;
            });
        });
    }

    int coarseStencilWidth() const override { return Fill::coarseStencilWidth(); }

private:
    static double kinetic_(double const rho, double const mx, double const my, double const mz)
    {
        return (mx * mx + my * my + mz * mz) / (2. * rho);
    }

    typename Fill::Options options_;
    std::shared_ptr<RefineProbe> probe_;
};


//! The energy-row kernel of the rule of `settings` (order 2, Linear, only).
template<typename GridLayoutT, typename FieldT>
std::unique_ptr<IEnergyRowKernel<GridLayoutT, FieldT>>
makeEnergyRowKernel([[maybe_unused]] FieldRefinementOrder const order,
                    LimiterSettings const& settings, std::shared_ptr<RefineProbe> probe = nullptr)
{
    using Kernel_ptr             = std::unique_ptr<IEnergyRowKernel<GridLayoutT, FieldT>>;
    constexpr std::size_t linear = static_cast<std::size_t>(FieldRefinementOrder::Linear);

    return withRule(settings.limiter, [&]<typename Rule>(std::type_identity<Rule>) -> Kernel_ptr {
        return std::make_unique<EnergyRowRefiner<GridLayoutT, FieldT, linear, Rule>>(
            settings, std::move(probe));
    });
}


} // namespace PHARE::amr


#endif
