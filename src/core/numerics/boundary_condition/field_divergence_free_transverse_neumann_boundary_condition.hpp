#ifndef PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIVERGENCE_FREE_TRANSVERSE_NEUMANN_BOUNDARY_CONDITION_HPP
#define PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIVERGENCE_FREE_TRANSVERSE_NEUMANN_BOUNDARY_CONDITION_HPP

#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/numerics/boundary_condition/field_neumann_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_boundary_condition_dispatcher.hpp"

#include <cstddef>

namespace PHARE::core
{
/**
 * @brief Boundary condition implementation for vector fields, that enforces the normal derivative
 * of the tangential components to be zero, and sets the normal component in the ghost cells such
 * that its numerical divergence is zero.
 *
 * First tangential components are mirrored, then the normal component is filled on ghost cells to
 * have a null divergence.
 *
 * @warning This condition only makes sense for a vector field with same centerings than the
 * magnetic vector field.
 *
 * @tparam VecFieldT Type of the vector field.
 * @tparam GridLayoutT Grid layout configuration.
 *
 */
template<typename VecFieldT, typename GridLayoutT>
class FieldDivergenceFreeTransverseNeumannBoundaryCondition
    : public FieldBoundaryConditionDispatcher<
          VecFieldT, GridLayoutT,
          FieldDivergenceFreeTransverseNeumannBoundaryCondition<VecFieldT, GridLayoutT>>
{
public:
    using Super = FieldBoundaryConditionDispatcher<
        VecFieldT, GridLayoutT,
        FieldDivergenceFreeTransverseNeumannBoundaryCondition<VecFieldT, GridLayoutT>>;
    using tensor_quantity_type = Super::tensor_quantity_type;
    using field_type             = Super::field_type;

    static constexpr size_t dimension = Super::dimension;
    static constexpr size_t N         = Super::N;
    static_assert(
        N == 3,
        "Divergence-free transverse Neumann boundary condition only applies to vector fields.");

    FieldDivergenceFreeTransverseNeumannBoundaryCondition() = default;

    FieldDivergenceFreeTransverseNeumannBoundaryCondition(
        FieldDivergenceFreeTransverseNeumannBoundaryCondition const&)
        = default;
    FieldDivergenceFreeTransverseNeumannBoundaryCondition&
    operator=(FieldDivergenceFreeTransverseNeumannBoundaryCondition const&)
        = default;
    FieldDivergenceFreeTransverseNeumannBoundaryCondition(
        FieldDivergenceFreeTransverseNeumannBoundaryCondition&&)
        = default;
    FieldDivergenceFreeTransverseNeumannBoundaryCondition&
    operator=(FieldDivergenceFreeTransverseNeumannBoundaryCondition&&)
        = default;

    virtual ~FieldDivergenceFreeTransverseNeumannBoundaryCondition() = default;


    /** @brief Implements getType. */
    FieldBoundaryConditionType getType() const override
    {
        return FieldBoundaryConditionType::DivergenceFreeTransverseNeumann;
    }


    /**
     * @brief Apply the condition using compile-time specialized parameters.
     *
     * @tparam direction Normal direction of the boundary.
     * @tparam side Boundary side (Lower or Upper).
     * @tparam Centerings Component-wise centerings.
     *
     */
    template<Direction direction, Side side, QtyCentering... Centerings>
    void apply_specialized(VecFieldT& vecField, Box<std::uint32_t, dimension> const& localGhostBox,
                           GridLayoutT const& gridLayout, double const time,
                           [[maybe_unused]] Super::patch_field_accessor_type const&
                               fieldAccessor)
    {
        constexpr std::array centerings = {Centerings...};

        auto fields = vecField.components();

        // here we check the condition that the vector field has same staggering than the magnetic
        // field
        assert(gridLayout.centering(vecField) == gridLayout.centering(tensor_quantity_type::B));

        // handle transverse components
        for_N<N>([&](auto iTransverse) {
            if constexpr (iTransverse != static_cast<size_t>(direction))
            {
                constexpr QtyCentering centering = centerings[iTransverse];
                field_type& tField               = std::get<iTransverse>(fields);
                scalar_neumann_condition_.template apply_specialized<direction, side, centering>(
                    tField, localGhostBox, gridLayout, time, fieldAccessor);
            }
        });

        // handle normal component, by iterating on ghost cells from closest to farthest to the
        // physical domain.
        constexpr size_t iNormal = static_cast<size_t>(direction);
        field_type& nField       = std::get<iNormal>(fields);

        // define a lambda for iterating in different orders
        auto apply_loop_for_normal_component = [&](auto begin, auto end) {
            for (auto it = begin; it != end; ++it)
            {
                _index const& index = *it;

                // compute the "transverse divergence" in the cell
                double transverseDiv = 0.0;
                for_N<dimension>([&](auto iTransverse) {
                    if constexpr (iTransverse != iNormal)
                    {
                        field_type& tField        = std::get<iTransverse>(fields);
                        _index const& upper_index = index.template neighbor<iTransverse, 1>();
                        transverseDiv += tField(upper_index) - tField(index);
                    }
                });

                // set the last unset normal component using the disrete form of 'div = 0'
                if constexpr (side == Side::Upper)
                {
                    _index const& index_to_set      = index.template neighbor<iNormal, 1>();
                    _index const& index_already_set = index;
                    nField(index_to_set)            = nField(index_already_set) - transverseDiv;
                }
                else // if constexpr (side == Side::Lower)
                {
                    _index const& index_to_set      = index; // to continue
                    _index const& index_already_set = index.template neighbor<iNormal, 1>();
                    nField(index_to_set)            = nField(index_already_set) + transverseDiv;
                }
            }
        };

        // apply the loop in the required following which side of the box we are on
        if constexpr (side == Side::Upper)
        {
            apply_loop_for_normal_component(localGhostBox.begin(), localGhostBox.end());
        }
        else // if constexpr (side == Side::Lower)
        {
            apply_loop_for_normal_component(localGhostBox.rbegin(), localGhostBox.rend());
        }
    }

private:
    using _scalar_neumann_boundary_condition_type
        = FieldNeumannBoundaryCondition<field_type, GridLayoutT>;
    using _index = Point<std::uint32_t, dimension>;

    _scalar_neumann_boundary_condition_type scalar_neumann_condition_;

}; // class FieldDivergenceFreeTransverseNeumannBoundaryCondition

} // namespace PHARE::core

#endif // PHARE_CORE_NUMERICS_BOUNDARY_CONDITION_FIELD_DIVERGENCE_FREE_TRANSVERSE_NEUMANN_BOUNDARY_CONDITION_HPP
