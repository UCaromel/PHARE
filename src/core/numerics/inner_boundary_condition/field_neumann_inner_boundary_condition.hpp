#ifndef PHARE_CORE_NUMERICAN_INNER_BOUNDARY_CONDITION_FIELD_NEUMANN_INNER_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICAN_INNER_BOUNDARY_CONDITION_FIELD_NEUMANN_INNER_BOUNDARY_CONDITION_HPP

#include "core/numerics/inner_boundary_condition/field_inner_boundary_condition.hpp"

namespace PHARE::core
{
template<typename ScalarOrTensorFieldT, typename GridLayoutT, typename PhysicalStateT>
class FieldNeumannInnerBoundaryCondition
    : public FieldInnerBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, PhysicalStateT>
{
public:
    using Super = FieldInnerBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, PhysicalStateT>;

    using field_type                    = Super::field_type;
    using inner_boundary_mesh_data_type = Super::inner_boundary_mesh_data_type;
    using ghost_elem_data_type          = Super::ghost_elem_data_type;
    using inner_boundary_type           = Super::inner_boundary_type;
    using interpolator_type             = Super::interpolator_type;
    using state_type                    = Super::state_type;

    static constexpr size_t dimension = Super::dimension;
    static constexpr size_t N         = Super::N;
    static constexpr bool is_scalar   = Super::is_scalar;

    FieldNeumannInnerBoundaryCondition() = default;

    FieldInnerBoundaryConditionType getType() const override
    {
        return FieldInnerBoundaryConditionType::Neumann;
    }

    void apply(ScalarOrTensorFieldT& scalarOrTensorField, GridLayoutT const& layout,
               inner_boundary_mesh_data_type const& boundaryMeshData, state_type const& state,
               double const time) override
    {
        auto fields = [&]() {
            if constexpr (is_scalar)
                return std::make_tuple(scalarOrTensorField);
            else
                return scalarOrTensorField.components();
        }();

        for_N<N>([&](auto i) {
            auto& field            = std::get<i>(fields);
            auto const centering   = GridLayoutT::centering(field);
            auto const& ghostElems = boundaryMeshData.getGhostDataFromCentering(centering);

            for (ghost_elem_data_type const& ghostElem : ghostElems)
            {
                if (!ghostElem.mirrorIsInPatch)
                    continue;
                field(ghostElem.index) = this->interpolator_(layout, field, ghostElem.mirrorPoint);
            }
        });
    }
};

} // namespace PHARE::core
#endif // PHARE_CORE_NUMERICAN_INNER_BOUNDARY_CONDITION_FIELD_NEUMANN_INNER_BOUNDARY_CONDITION_HPP
