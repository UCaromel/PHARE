#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_NEUMANN_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_NEUMANN_BOUNDARY_CONDITION_HPP

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition_dispatcher.hpp"

#include <cstddef>
#include <tuple>

namespace PHARE::core
{
/**
 * @brief Neumann boundary condition implementation for fields and tensor fields.
 *
 * This class implements a zero-gradient boundary condition by mirroring values
 * from the physical domain into the ghost regions.
 *
 * @tparam ScalarOrTensorFieldT Type of the field or tensor field.
 * @tparam GridLayoutT Grid layout configuration.
 *
 */
template<typename ScalarOrTensorFieldT, typename GridLayoutT>
class FieldNeumannBoundaryCondition
    : public FieldBoundaryConditionDispatcher<
          ScalarOrTensorFieldT, GridLayoutT,
          FieldNeumannBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT>>
{
public:
    using Super = FieldBoundaryConditionDispatcher<
        ScalarOrTensorFieldT, GridLayoutT,
        FieldNeumannBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT>>;
    using tensor_quantity_type = Super::tensor_quantity_type;
    using field_type             = Super::field_type;

    static constexpr size_t dimension = Super::dimension;
    static constexpr size_t N         = Super::N;
    static constexpr bool is_scalar   = Super::is_scalar;

    FieldNeumannBoundaryCondition() = default;

    FieldNeumannBoundaryCondition(FieldNeumannBoundaryCondition const&)            = default;
    FieldNeumannBoundaryCondition& operator=(FieldNeumannBoundaryCondition const&) = default;
    FieldNeumannBoundaryCondition(FieldNeumannBoundaryCondition&&)                 = default;
    FieldNeumannBoundaryCondition& operator=(FieldNeumannBoundaryCondition&&)      = default;

    virtual ~FieldNeumannBoundaryCondition() = default;


    /** @brief Implement getType to return Neumann. */
    FieldBoundaryConditionType getType() const override
    {
        return FieldBoundaryConditionType::Neumann;
    }


    /**
     * @brief Apply the Neumann condition using compile-time specialized parameters.
     *
     * @tparam direction Normal direction of the boundary.
     * @tparam side Boundary side (Lower or Upper).
     * @tparam Centerings Component-wise centerings.
     *
     */
    template<Direction direction, Side side, QtyCentering... Centerings>
    void apply_specialized(ScalarOrTensorFieldT& scalarOrTensorField,
                           Box<std::uint32_t, dimension> const& localGhostBox,
                           GridLayoutT const& gridLayout, double const time,
                           [[maybe_unused]] Super::patch_field_accessor_type const&
                               fieldAccessor)
    {
        using Index = Point<std::uint32_t, dimension>;

        constexpr std::array centerings = {Centerings...};

        // no other way than using a lambda builder
        auto fields = [&]() {
            if constexpr (is_scalar)
                return std::make_tuple(scalarOrTensorField);
            else
                return scalarOrTensorField.components();
        }();

        for_N<N>([&](auto i) {
            constexpr QtyCentering centering = centerings[i];
            field_type& field                = std::get<i>(fields);
            auto fieldBox = gridLayout.toFieldBox(localGhostBox, field.physicalQuantity());
            Index physicalLimitIndex = (side == Side::Lower)
                                           ? gridLayout.physicalStartIndex(centering)
                                           : gridLayout.physicalEndIndex(centering);
            for (Index const& index : fieldBox)
            {
                Index mirrorIndex
                    = gridLayout.template boundaryMirrored<dimension, direction, side, centering>(
                        index);
                field(index) = field(mirrorIndex);
            }
        });
    }
}; // class FieldNeumannBoundaryCondition

} // namespace PHARE::core

#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_NEUMANN_BOUNDARY_CONDITION_HPP
