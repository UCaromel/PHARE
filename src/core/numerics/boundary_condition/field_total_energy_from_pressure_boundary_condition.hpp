#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_TOTAL_ENERGY_FROM_PRESSURE_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_TOTAL_ENERGY_FROM_PRESSURE_BOUNDARY_CONDITION_HPP

#include "core/boundary/boundary_defs.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition.hpp"
#include "core/numerics/primite_conservative_converter/conversion_utils.hpp"
#include "core/numerics/thermo/thermo.hpp"

#include <memory>

namespace PHARE::core
{

/**
 * @brief Boundary condition for the total energy field that derives ghost values from a
 * Neumann (zero-gradient) pressure condition rather than from a prescribed energy value.
 *
 * @tparam FieldT       Scalar field type (must satisfy IsField).
 * @tparam GridLayoutT  Grid layout type (must satisfy IsGridLayout).
 */
template<typename FieldT, typename GridLayoutT>
class FieldTotalEnergyFromPressureBoundaryCondition
    : public IFieldBoundaryCondition<FieldT, GridLayoutT>
{
public:
    using Super                  = IFieldBoundaryCondition<FieldT, GridLayoutT>;
    using field_type             = Super::field_type;
    using physical_quantity_type = typename GridLayoutT::Quantity;
    using scalar_quantity_type   = typename physical_quantity_type::Scalar;
    using vector_quantity_type   = typename physical_quantity_type::Vector;
    using vectorfield_type       = VecField<FieldT, physical_quantity_type>;

    using scalar_bc_type = IFieldBoundaryCondition<FieldT, GridLayoutT>;
    using vector_bc_type = IFieldBoundaryCondition<vectorfield_type, GridLayoutT>;

    static constexpr size_t dimension = Super::dimension;
    static constexpr size_t N         = Super::N;
    static_assert(N == 1,
                  "FieldTotalEnergyFromPressureBoundaryCondition only applies to scalar fields.");

    FieldTotalEnergyFromPressureBoundaryCondition(std::shared_ptr<scalar_bc_type> rho_bc,
                                                  std::shared_ptr<vector_bc_type> rhoV_bc,
                                                  std::shared_ptr<vector_bc_type> B1_bc,
                                                  std::shared_ptr<scalar_bc_type> P_bc,
                                                  std::shared_ptr<Thermo> thermo)
        : rho_bc_{std::move(rho_bc)}
        , rhoV_bc_{std::move(rhoV_bc)}
        , B1_bc_{std::move(B1_bc)}
        , P_bc_{std::move(P_bc)}
        , thermo_{std::move(thermo)}
    {
    }

    FieldTotalEnergyFromPressureBoundaryCondition(
        FieldTotalEnergyFromPressureBoundaryCondition const&)
        = default;
    FieldTotalEnergyFromPressureBoundaryCondition&
    operator=(FieldTotalEnergyFromPressureBoundaryCondition const&)
        = default;
    FieldTotalEnergyFromPressureBoundaryCondition(FieldTotalEnergyFromPressureBoundaryCondition&&)
        = default;
    FieldTotalEnergyFromPressureBoundaryCondition&
    operator=(FieldTotalEnergyFromPressureBoundaryCondition&&)
        = default;

    virtual ~FieldTotalEnergyFromPressureBoundaryCondition() = default;

    FieldBoundaryConditionType getType() const override
    {
        return FieldBoundaryConditionType::TotalEnergyFromPressure;
    }

    void apply(FieldT& Etot1Field, BoundaryLocation const boundaryLocation,
               Box<std::uint32_t, dimension> const& localGhostBox, GridLayoutT const& gridLayout,
               Super::boundary_condition_context_type const& ctx) override
    {
        Direction const direction = getDirection(boundaryLocation);
        Side const side           = getSide(boundaryLocation);
        QtyCentering const centering
            = GridLayoutT::centering(Etot1Field.physicalQuantity())[static_cast<size_t>(direction)];

        auto const& fieldAccessor = ctx.accessor_new;
        auto& rhoField = fieldAccessor.getField(scalar_quantity_type::rho);
        auto& PField   = fieldAccessor.getField(scalar_quantity_type::P);
        auto rhoVField = fieldAccessor.getVecField(vector_quantity_type::rhoV);
        auto B1Field   = fieldAccessor.getVecField(vector_quantity_type::B1);

        auto rhoVComps = rhoVField.components();
        auto B1Comps   = B1Field.components();
        auto& rhoVx    = std::get<0>(rhoVComps);
        auto& rhoVy    = std::get<1>(rhoVComps);
        auto& rhoVz    = std::get<2>(rhoVComps);
        auto& B1x      = std::get<0>(B1Comps);
        auto& B1y      = std::get<1>(B1Comps);
        auto& B1z      = std::get<2>(B1Comps);

        auto const etotFieldBox
            = gridLayout.toFieldBox(localGhostBox, Etot1Field.physicalQuantity());

        // Step 1: fill P at mirror of each ghost cell from current conservative variables
        for (auto const& index : etotFieldBox)
        {
            auto const mirrorIdx = gridLayout.boundaryMirrored(direction, side, centering, index);

            double const b1x
                = GridLayoutT::template project<GridLayoutT::faceXToCellCenter>(B1x, mirrorIdx);
            double const b1y
                = GridLayoutT::template project<GridLayoutT::faceYToCellCenter>(B1y, mirrorIdx);
            double const b1z
                = GridLayoutT::template project<GridLayoutT::faceZToCellCenter>(B1z, mirrorIdx);

            double const rho_m = rhoField(mirrorIdx);
            double const vx    = rhoVx(mirrorIdx) / rho_m;
            double const vy    = rhoVy(mirrorIdx) / rho_m;
            double const vz    = rhoVz(mirrorIdx) / rho_m;

            double const e_int = internalEnergyFromTotalEnergy(Etot1Field(mirrorIdx), rho_m, vx, vy,
                                                               vz, b1x, b1y, b1z);
            thermo_->setState_DU(rho_m, e_int / rho_m);
            PField(mirrorIdx) = thermo_->pressure();
        }

        // Step 2: apply sub-BCs to fill ghost layers of ρ, ρv, B, P
        rho_bc_->apply(rhoField, boundaryLocation, localGhostBox, gridLayout, ctx);
        rhoV_bc_->apply(rhoVField, boundaryLocation, localGhostBox, gridLayout, ctx);
        B1_bc_->apply(B1Field, boundaryLocation, localGhostBox, gridLayout, ctx);
        P_bc_->apply(PField, boundaryLocation, localGhostBox, gridLayout, ctx);

        // Step 3: compute Etot in ghost cells from freshly filled P, ρ, ρv, B
        for (auto const& index : etotFieldBox)
        {
            double const rho_g = rhoField(index);
            double const vx    = rhoVx(index) / rho_g;
            double const vy    = rhoVy(index) / rho_g;
            double const vz    = rhoVz(index) / rho_g;

            double const b1x
                = GridLayoutT::template project<GridLayoutT::faceXToCellCenter>(B1x, index);
            double const b1y
                = GridLayoutT::template project<GridLayoutT::faceYToCellCenter>(B1y, index);
            double const b1z
                = GridLayoutT::template project<GridLayoutT::faceZToCellCenter>(B1z, index);

            thermo_->setState_DP(rho_g, PField(index));
            double const e_int = thermo_->internalEnergy() * rho_g;
            Etot1Field(index)
                = totalEnergyFromInternalEnergy(e_int, rho_g, vx, vy, vz, b1x, b1y, b1z);
        }
    }

private:
    std::shared_ptr<scalar_bc_type> rho_bc_;
    std::shared_ptr<vector_bc_type> rhoV_bc_;
    std::shared_ptr<vector_bc_type> B1_bc_;
    std::shared_ptr<scalar_bc_type> P_bc_;
    std::shared_ptr<Thermo> thermo_;
};

} // namespace PHARE::core
#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_TOTAL_ENERGY_FROM_PRESSURE_BOUNDARY_CONDITION_HPP
