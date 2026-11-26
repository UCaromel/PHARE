#ifndef PHARE_ENERGY_ROW_REFINE_OPERATOR_HPP
#define PHARE_ENERGY_ROW_REFINE_OPERATOR_HPP

#include "core/def.hpp"
#include "phare_mpi.hpp" // IWYU pragma: keep

#include "amr/data/field/field_data.hpp"
#include "amr/data/field/field_geometry.hpp"
#include "amr/data/field/field_overlap.hpp"

#include "energy_row_refiner.hpp"

#include <SAMRAI/hier/Patch.h>
#include <SAMRAI/hier/RefineOperator.h>
#include <SAMRAI/tbox/Dimension.h>

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>


namespace PHARE::amr
{

/**
 * @brief Total-energy refine operator of the energy row: refines Etot with the density and
 * momentum read from the same coarse patch.
 *
 * SAMRAI hands every operator of a schedule the coarse-interpolation patch as `source`, on which
 * the scratch data of every item of the schedule is allocated and filled before any operator
 * runs (RefineSchedule::recursiveFill, then refineScratchData). The messenger registers its items
 * with scratch == destination, so the density and momentum of the state are found there under
 * their own patch-data ids, whatever order the items are refined in.
 */
template<typename GridLayoutT, typename FieldT, typename VectorFieldDataT>
class EnergyRowRefineOperator : public SAMRAI::hier::RefineOperator
{
public:
    static constexpr std::size_t dimension = GridLayoutT::dimension;
    using FieldDataT                       = FieldData<GridLayoutT, FieldT>;
    using Kernel_t                         = IEnergyRowKernel<GridLayoutT, FieldT>;

    //! rhoId, rhoVId: patch-data ids of the density and momentum items of the same schedule
    EnergyRowRefineOperator(std::unique_ptr<Kernel_t> kernel, int const rhoId, int const rhoVId)
        : SAMRAI::hier::RefineOperator{"EnergyRowRefineOperator"}
        , kernel_{std::move(kernel)}
        , rhoId_{rhoId}
        , rhoVId_{rhoVId}
    {
    }

    virtual ~EnergyRowRefineOperator() = default;

    NO_DISCARD int getOperatorPriority() const override { return 0; }

    NO_DISCARD SAMRAI::hier::IntVector
    getStencilWidth(SAMRAI::tbox::Dimension const& dim) const override
    {
        return SAMRAI::hier::IntVector(dim, kernel_->coarseStencilWidth());
    }

    void refine(SAMRAI::hier::Patch& destination, SAMRAI::hier::Patch const& source,
                int const destinationId, int const sourceId,
                SAMRAI::hier::BoxOverlap const& destinationOverlap,
                SAMRAI::hier::IntVector const& ratio) const override
    {
        for (auto const id : {rhoId_, rhoVId_})
            if (!source.checkAllocated(id))
                throw std::runtime_error(
                    "EnergyRowRefineOperator: patch data " + std::to_string(id)
                    + " (density or momentum) is not allocated on the coarse patch");

        auto const& overlapBoxes
            = dynamic_cast<FieldOverlap const&>(destinationOverlap).getDestinationBoxContainer();

        auto& fineEtot         = FieldDataT::getField(destination, destinationId);
        auto const& destLayout = FieldDataT::getLayout(destination, destinationId);
        auto const& coarseEtot = FieldDataT::getField(source, sourceId);
        auto const& srcLayout  = FieldDataT::getLayout(source, sourceId);
        auto const& coarseRho  = FieldDataT::getField(source, rhoId_);
        auto const& coarseRhoV = VectorFieldDataT::getFields(source, rhoVId_);
        auto const& rhoVLayout = VectorFieldDataT::getLayout(source, rhoVId_);
        auto const& rhoLayout  = FieldDataT::getLayout(source, rhoId_);

        auto const& qty         = fineEtot.physicalQuantity();
        auto const destFieldBox = FieldDataT::Geometry::toFieldBox(
            destination.getPatchData(destinationId)->getGhostBox(), qty, destLayout);
        auto const sourceFieldBox = FieldDataT::Geometry::toFieldBox(
            source.getPatchData(sourceId)->getGhostBox(), qty, srcLayout);

        for (auto const c : destLayout.centering(qty))
            if (c != core::QtyCentering::dual)
                throw std::runtime_error("EnergyRowRefineOperator: Etot must be cell-centred");

        // the kernel reads the four quantities through one source field box
        auto const sameSourceBox = [&](auto const& field, auto const& layout, int const id) {
            using Geometry
                = FieldGeometry<GridLayoutT, std::decay_t<decltype(field.physicalQuantity())>>;
            auto const box = Geometry::toFieldBox(source.getPatchData(id)->getGhostBox(),
                                                  field.physicalQuantity(), layout);
            for (auto const c : layout.centering(field.physicalQuantity()))
                if (c != core::QtyCentering::dual)
                    return false;
            return box.isSpatiallyEqual(sourceFieldBox);
        };
        bool consistent = sameSourceBox(coarseRho, rhoLayout, rhoId_);
        for (auto const& component : coarseRhoV)
            consistent = consistent and sameSourceBox(component, rhoVLayout, rhoVId_);
        if (!consistent)
            throw std::runtime_error("EnergyRowRefineOperator: density, momentum and Etot must be "
                                     "cell-centred on the same ghost box");

        std::array<FieldT const*, 3> const rhoV{std::addressof(coarseRhoV[0]),
                                                std::addressof(coarseRhoV[1]),
                                                std::addressof(coarseRhoV[2])};

        for (auto const& box : overlapBoxes)
            kernel_->refineBox(coarseRho, rhoV, coarseEtot, fineEtot, destFieldBox * box,
                               destFieldBox, sourceFieldBox, ratio);
    }

private:
    std::unique_ptr<Kernel_t> kernel_;
    int rhoId_;
    int rhoVId_;
};


} // namespace PHARE::amr


#endif
