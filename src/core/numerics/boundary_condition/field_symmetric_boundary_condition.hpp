#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_SYMMETRIC_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_SYMMETRIC_BOUNDARY_CONDITION_HPP

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/numerics/boundary_condition/field_neumann_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_dirichlet_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition_dispatcher.hpp"

namespace PHARE::core
{
/**
 * @brief Symmetric boundary condition for scalar and vector fields.
 *
 * For scalars, this class imposes a null derivative along the normal (equivalent to a Neumann
 * boudary condition). For vectors it imposes a Neumann bonditions on the tangential components, and
 * a null value for the normal component.
 *
 * @tparam ScalarOrTensorFieldT Type of the field or tensor field.
 * @tparam GridLayoutT Grid layout configuration.
 *
 */
template<typename ScalarOrTensorFieldT, typename GridLayoutT>
class FieldSymmetricBoundaryCondition
    : public FieldBoundaryConditionDispatcher<
          ScalarOrTensorFieldT, GridLayoutT,
          FieldSymmetricBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT>>
{
public:
    using Super = FieldBoundaryConditionDispatcher<
        ScalarOrTensorFieldT, GridLayoutT,
        FieldSymmetricBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT>>;
    using tensor_quantity_type = Super::tensor_quantity_type;
    using field_type             = Super::field_type;

    static constexpr size_t dimension = Super::dimension;
    static constexpr size_t N         = Super::N;
    static constexpr bool is_scalar   = Super::is_scalar;

    FieldSymmetricBoundaryCondition() = default;

    FieldSymmetricBoundaryCondition(FieldSymmetricBoundaryCondition const&)            = default;
    FieldSymmetricBoundaryCondition& operator=(FieldSymmetricBoundaryCondition const&) = default;
    FieldSymmetricBoundaryCondition(FieldSymmetricBoundaryCondition&&)                 = default;
    FieldSymmetricBoundaryCondition& operator=(FieldSymmetricBoundaryCondition&&)      = default;

    virtual ~FieldSymmetricBoundaryCondition() = default;


    /** @brief Implement getType to return Symmetric. */
    FieldBoundaryConditionType getType() const override
    {
        return FieldBoundaryConditionType::Symmetric;
    }


    /**
     * @brief Apply the symmetric condition using compile-time specialized parameters.
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
            if constexpr ((i != static_cast<size_t>(direction)) || is_scalar)
            // if the component is tangent to the boundary, or if we are handling a scalar
            {
                scalar_neumann_condition_.template apply_specialized<direction, side, centering>(
                    field, localGhostBox, gridLayout, time, fieldAccessor);
            }
            else
            // if the component is normal to the boundary
            {
                scalar_dirichlet_condition_.template apply_specialized<direction, side, centering>(
                    field, localGhostBox, gridLayout, time, fieldAccessor);
            }
        });
    }

private:
    using _scalar_neumann_condition_type = FieldNeumannBoundaryCondition<field_type, GridLayoutT>;
    using _scalar_dirichlet_condition_type
        = FieldDirichletBoundaryCondition<field_type, GridLayoutT>;

    _scalar_neumann_condition_type scalar_neumann_condition_{};
    _scalar_dirichlet_condition_type scalar_dirichlet_condition_{};

}; // class FieldSymmetricBoundaryCondition

} // namespace PHARE::core

#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_SYMMETRIC_BOUNDARY_CONDITION_HPP
