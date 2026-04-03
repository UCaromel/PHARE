#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIRICHLET_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIRICHLET_BOUNDARY_CONDITION_HPP

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition_dispatcher.hpp"

#include <cstddef>
#include <tuple>

namespace PHARE::core
{
/**
 * @brief Dirichlet boundary condition for scalar and vector fields.
 *
 * Impose a constajt value on the boundary by linearly extrapolating the (tensor) field in the ghost
 * cells.
 *
 * @tparam ScalarOrTensorFieldT Type of the field or tensor field.
 * @tparam GridLayoutT Grid layout configuration.
 *
 */
template<typename ScalarOrTensorFieldT, typename GridLayoutT>
class FieldDirichletBoundaryCondition
    : public FieldBoundaryConditionDispatcher<
          ScalarOrTensorFieldT, GridLayoutT,
          FieldDirichletBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT>>
{
public:
    using Super = FieldBoundaryConditionDispatcher<
        ScalarOrTensorFieldT, GridLayoutT,
        FieldDirichletBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT>>;
    using tensor_quantity_type = Super::tensor_quantity_type;
    using field_type             = Super::field_type;
    using value_type             = field_type::value_type;

    static constexpr size_t dimension = Super::dimension;
    static constexpr size_t N         = Super::N;
    static constexpr bool is_scalar   = Super::is_scalar;

    FieldDirichletBoundaryCondition() = default;

    FieldDirichletBoundaryCondition(value_type value)
        : value_{value} {};

    FieldDirichletBoundaryCondition(std::array<value_type, N> value)
        : value_{value} {};

    FieldDirichletBoundaryCondition(FieldDirichletBoundaryCondition const&)            = default;
    FieldDirichletBoundaryCondition& operator=(FieldDirichletBoundaryCondition const&) = default;
    FieldDirichletBoundaryCondition(FieldDirichletBoundaryCondition&&)                 = default;
    FieldDirichletBoundaryCondition& operator=(FieldDirichletBoundaryCondition&&)      = default;

    virtual ~FieldDirichletBoundaryCondition() = default;


    /** @brief Implement getType to return Dirichlet. */
    FieldBoundaryConditionType getType() const override
    {
        return FieldBoundaryConditionType::Dirichlet;
    }


    /**
     * @brief Apply the Dirichlet condition using compile-time specialized parameters.
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
            for (_index_type const& index : fieldBox)
            {
                _index_type mirrorIndex
                    = gridLayout.template boundaryMirrored<dimension, direction, side, centering>(
                        index);
                // if the ghost is on the boundary (possible if primal), set to value,
                // else set with a linear extrapolation
                constexpr size_t dir_i = static_cast<size_t>(direction);
                field(index) = (mirrorIndex[dir_i] == index[dir_i]) ? value_[i]
                                                            : 2.0 * value_[i] - field(mirrorIndex);
            }
        });
    }

private:
    using _index_type = Point<std::uint32_t, dimension>;

    std::array<value_type, N> value_{0}; /**< Value to impose on the boundary, zero by default. */

}; // class FieldDirichletBoundaryCondition

} // namespace PHARE::core

#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIRICHLET_BOUNDARY_CONDITION_HPP
