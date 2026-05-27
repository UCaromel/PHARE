#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIRICHLET_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIRICHLET_BOUNDARY_CONDITION_HPP

#include "core/boundary/boundary_defs.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition.hpp"

#include <cstddef>
#include <tuple>

namespace PHARE::core
{
/**
 * @brief Dirichlet boundary condition for scalar and vector fields.
 *
 * Impose a constant value on the boundary by linearly extrapolating the (tensor) field in the ghost
 * cells.
 *
 * @tparam ScalarOrTensorFieldT Type of the field or tensor field.
 * @tparam GridLayoutT Grid layout configuration.
 *
 */
template<typename ScalarOrTensorFieldT, typename GridLayoutT>
class FieldDirichletBoundaryCondition
    : public IFieldBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT>
{
public:
    using Super                    = IFieldBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT>;
    using tensor_quantity_type     = Super::tensor_quantity_type;
    using field_type               = Super::field_type;
    using value_type               = field_type::value_type;

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

    FieldBoundaryConditionType getType() const override
    {
        return FieldBoundaryConditionType::Dirichlet;
    }

    void apply(ScalarOrTensorFieldT& scalarOrTensorField,
               BoundaryLocation const boundaryLocation,
               Box<std::uint32_t, dimension> const& localGhostBox, GridLayoutT const& gridLayout,
               [[maybe_unused]] Super::boundary_condition_context_type const& ctx) override
    {
        Direction const direction = getDirection(boundaryLocation);
        Side const side           = getSide(boundaryLocation);

        if (static_cast<size_t>(direction) >= dimension)
            return;

        auto fields = [&]() {
            if constexpr (is_scalar)
                return std::make_tuple(scalarOrTensorField);
            else
                return scalarOrTensorField.components();
        }();

        for_N<N>([&](auto i) {
            field_type& field = std::get<i>(fields);
            QtyCentering const centering
                = GridLayoutT::centering(field.physicalQuantity())[static_cast<size_t>(direction)];
            auto fieldBox = gridLayout.toFieldBox(localGhostBox, field.physicalQuantity());
            for (_index_type const& index : fieldBox)
            {
                _index_type mirrorIndex
                    = gridLayout.boundaryMirrored(direction, side, centering, index);
                size_t const iDir = static_cast<size_t>(direction);
                field(index)      = (mirrorIndex[iDir] == index[iDir])
                                        ? value_[i]
                                        : 2.0 * value_[i] - field(mirrorIndex);
            }
        });
    }

private:
    using _index_type = Point<std::uint32_t, dimension>;

    std::array<value_type, N> value_{0};

}; // class FieldDirichletBoundaryCondition

} // namespace PHARE::core
#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIRICHLET_BOUNDARY_CONDITION_HPP
