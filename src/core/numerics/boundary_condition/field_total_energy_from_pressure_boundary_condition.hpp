#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_TOTAL_ENERGY_FROM_PRESSURE_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_TOTAL_ENERGY_FROM_PRESSURE_BOUNDARY_CONDITION_HPP

#include "core/boundary/boundary_defs.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition_dispatcher.hpp"
#include "core/numerics/primite_conservative_converter/conversion_utils.hpp"
#include "core/numerics/thermo/thermo.hpp"

#include <memory>

namespace PHARE::core
{

namespace detail
{
/** @brief Convert compile-time Direction and Side to a runtime BoundaryLocation. */
template<Direction direction, Side side>
constexpr BoundaryLocation toBoundaryLocation()
{
    if constexpr (direction == Direction::X)
    {
        if constexpr (side == Side::Lower)
            return BoundaryLocation::XLower;
        else
            return BoundaryLocation::XUpper;
    }
    else if constexpr (direction == Direction::Y)
    {
        if constexpr (side == Side::Lower)
            return BoundaryLocation::YLower;
        else
            return BoundaryLocation::YUpper;
    }
    else
    {
        if constexpr (side == Side::Lower)
            return BoundaryLocation::ZLower;
        else
            return BoundaryLocation::ZUpper;
    }
}
} // namespace detail

/**
 * @brief Boundary condition for the total energy field that derives ghost values from a
 * Neumann (zero-gradient) pressure condition rather than from a prescribed energy value.
 *
 * This BC implements the following algorithm in its @c apply_specialized method:
 *
 * 1. Recover the pressure at the domain cells adjacent to the boundary (the "mirror" points)
 *    from the up-to-date conservative variables (Etot, ρ, ρv, B) via the EOS, because the
 *    stored P field is not guaranteed to be up to date at those cells at the time ghost BCs
 *    are applied.
 * 2. Apply sub-BCs for ρ, ρv, B, and P to fill their respective ghost layers.
 * 3. Compute the energy in each ghost cell from the freshly filled P, ρ, ρv, B.
 *
 * @tparam FieldT       Scalar field type (must satisfy IsField).
 * @tparam GridLayoutT  Grid layout type (must satisfy IsGridLayout).
 */
template<typename FieldT, typename GridLayoutT>
class FieldTotalEnergyFromPressureBoundaryCondition
    : public FieldBoundaryConditionDispatcher<
          FieldT, GridLayoutT,
          FieldTotalEnergyFromPressureBoundaryCondition<FieldT, GridLayoutT>>
{
public:
    using Super = FieldBoundaryConditionDispatcher<
        FieldT, GridLayoutT,
        FieldTotalEnergyFromPressureBoundaryCondition<FieldT, GridLayoutT>>;
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

    /**
     * @param rho_bc   BC to apply to the density field.
     * @param rhoV_bc  BC to apply to the momentum vector field.
     * @param B_bc     BC to apply to the magnetic field.
     * @param P_bc     BC to apply to the pressure field (typically Neumann).
     * @param thermo   EOS object used to convert between (ρ, u) and P.
     */
    FieldTotalEnergyFromPressureBoundaryCondition(std::shared_ptr<scalar_bc_type> rho_bc,
                                                  std::shared_ptr<vector_bc_type> rhoV_bc,
                                                  std::shared_ptr<vector_bc_type> B_bc,
                                                  std::shared_ptr<scalar_bc_type> P_bc,
                                                  std::shared_ptr<Thermo> thermo)
        : rho_bc_{std::move(rho_bc)}
        , rhoV_bc_{std::move(rhoV_bc)}
        , B_bc_{std::move(B_bc)}
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
    FieldTotalEnergyFromPressureBoundaryCondition(
        FieldTotalEnergyFromPressureBoundaryCondition&&)
        = default;
    FieldTotalEnergyFromPressureBoundaryCondition&
    operator=(FieldTotalEnergyFromPressureBoundaryCondition&&)
        = default;

    virtual ~FieldTotalEnergyFromPressureBoundaryCondition() = default;

    FieldBoundaryConditionType getType() const override
    {
        return FieldBoundaryConditionType::TotalEnergyFromPressure;
    }

    /**
     * @brief Apply the BC using compile-time specialized parameters.
     *
     * @tparam direction  Normal direction of the boundary.
     * @tparam side       Boundary side (Lower or Upper).
     * @tparam Centerings Centering of Etot (single element for a scalar field).
     */
    template<Direction direction, Side side, QtyCentering... Centerings>
    void apply_specialized(FieldT& etotField,
                           Box<std::uint32_t, dimension> const& localGhostBox,
                           GridLayoutT const& gridLayout, double const time,
                           Super::patch_field_accessor_type const& fieldAccessor)
    {
        constexpr std::array centerings = {Centerings...};
        constexpr auto centering        = centerings[0];
        constexpr BoundaryLocation boundaryLoc
            = detail::toBoundaryLocation<direction, side>();

        // Retrieve the other fields needed for the energy reconstruction
        auto& rhoField   = fieldAccessor.getField(scalar_quantity_type::rho);
        auto& PField     = fieldAccessor.getField(scalar_quantity_type::P);
        auto rhoVField   = fieldAccessor.getVecField(vector_quantity_type::rhoV);
        auto BField      = fieldAccessor.getVecField(vector_quantity_type::B);

        auto rhoVComps = rhoVField.components();
        auto BComps    = BField.components();
        auto& rhoVx    = std::get<0>(rhoVComps);
        auto& rhoVy    = std::get<1>(rhoVComps);
        auto& rhoVz    = std::get<2>(rhoVComps);
        auto& Bx       = std::get<0>(BComps);
        auto& By       = std::get<1>(BComps);
        auto& Bz       = std::get<2>(BComps);

        auto const etotFieldBox
            = gridLayout.toFieldBox(localGhostBox, etotField.physicalQuantity());

        // Step 1: fill P at the domain mirror of each ghost cell from the current
        // conservative variables (which are up to date in the domain at this stage)
        for (auto const& index : etotFieldBox)
        {
            auto const mirrorIdx
                = gridLayout.template boundaryMirrored<dimension, direction, side, centering>(
                    index);

            double const bx
                = GridLayoutT::project(Bx, mirrorIdx, GridLayoutT::faceXToCellCenter());
            double const by
                = GridLayoutT::project(By, mirrorIdx, GridLayoutT::faceYToCellCenter());
            double const bz
                = GridLayoutT::project(Bz, mirrorIdx, GridLayoutT::faceZToCellCenter());

            double const rho_m = rhoField(mirrorIdx);
            double const vx    = rhoVx(mirrorIdx) / rho_m;
            double const vy    = rhoVy(mirrorIdx) / rho_m;
            double const vz    = rhoVz(mirrorIdx) / rho_m;

            double const e_int
                = internalEnergyFromTotalEnergy(etotField(mirrorIdx), rho_m, vx, vy, vz, bx, by, bz);
            thermo_->setState_DU(rho_m, e_int / rho_m);
            PField(mirrorIdx) = thermo_->pressure();
        }

        // Step 2: apply sub-BCs to fill ghost layers of ρ, ρv, B, and P
        rho_bc_->apply(rhoField, boundaryLoc, localGhostBox, gridLayout, time, fieldAccessor);
        rhoV_bc_->apply(rhoVField, boundaryLoc, localGhostBox, gridLayout, time, fieldAccessor);
        B_bc_->apply(BField, boundaryLoc, localGhostBox, gridLayout, time, fieldAccessor);
        P_bc_->apply(PField, boundaryLoc, localGhostBox, gridLayout, time, fieldAccessor);

        // Step 3: compute Etot in ghost cells from the freshly filled P, ρ, ρv, B
        for (auto const& index : etotFieldBox)
        {
            double const rho_g = rhoField(index);
            double const vx    = rhoVx(index) / rho_g;
            double const vy    = rhoVy(index) / rho_g;
            double const vz    = rhoVz(index) / rho_g;

            double const bx
                = GridLayoutT::project(Bx, index, GridLayoutT::faceXToCellCenter());
            double const by
                = GridLayoutT::project(By, index, GridLayoutT::faceYToCellCenter());
            double const bz
                = GridLayoutT::project(Bz, index, GridLayoutT::faceZToCellCenter());

            thermo_->setState_DP(rho_g, PField(index));
            double const e_int = thermo_->internalEnergy() * rho_g;
            etotField(index)
                = totalEnergyFromInternalEnergy(e_int, rho_g, vx, vy, vz, bx, by, bz);
        }
    }

private:
    std::shared_ptr<scalar_bc_type> rho_bc_;
    std::shared_ptr<vector_bc_type> rhoV_bc_;
    std::shared_ptr<vector_bc_type> B_bc_;
    std::shared_ptr<scalar_bc_type> P_bc_;
    std::shared_ptr<Thermo> thermo_;
};

} // namespace PHARE::core

#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_TOTAL_ENERGY_FROM_PRESSURE_BOUNDARY_CONDITION_HPP
