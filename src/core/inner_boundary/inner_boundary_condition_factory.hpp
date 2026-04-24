#ifndef PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_CONDITION_FACTORY_HPP
#define PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_CONDITION_FACTORY_HPP

#include "core/data/field/field_traits.hpp"
#include "core/data/grid/gridlayout_traits.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/inner_boundary/field_antisymmetric_inner_boundary_condition.hpp"
#include "core/inner_boundary/field_neumann_inner_boundary_condition.hpp"
#include "core/inner_boundary/field_symmetric_inner_boundary_condition.hpp"
#include "core/inner_boundary/field_inner_boundary_condition.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace PHARE::core
{

/**
 * @brief Taxonomy of supported inner-boundary physics types.
 *
 * Each value maps to a set of per-quantity boundary conditions with physics-based
 * rules, analogous to BoundaryType for outer boundaries.
 *
 * - **Reflective** — specular reflection.
 *     scalars: Neumann, B: Symmetric, rhoV: Symmetric, E: Antisymmetric,
 *     others: Neumann.
 */
enum class InnerBoundaryConditionType { Reflective };

inline InnerBoundaryConditionType getInnerBoundaryConditionTypeFromString(std::string const& name)
{
    static std::unordered_map<std::string, InnerBoundaryConditionType> const typeMap{
        {"reflective", InnerBoundaryConditionType::Reflective},
    };
    auto it = typeMap.find(name);
    if (it == typeMap.end())
        throw std::runtime_error("Unknown inner boundary condition type: " + name);
    return it->second;
}


/**
 * @brief Factory that creates per-quantity inner boundary condition objects.
 *
 * Mirrors BoundaryFactory for outer boundaries. For a given InnerBoundaryConditionType
 * it registers the appropriate FieldInnerBoundaryCondition for every scalar and
 * vector quantity supplied by the caller.
 *
 * @tparam PhysicalQuantityT  Quantity traits (MHDQuantity, HybridQuantity, …).
 * @tparam FieldT             Scalar field type.
 * @tparam GridLayoutT        Grid layout type.
 * @tparam PhysicalStateT     Physical state type forwarded to BC appliers.
 */
template<typename PhysicalQuantityT, IsField FieldT, IsGridLayout GridLayoutT,
         typename PhysicalStateT>
class InnerBoundaryConditionFactory
{
public:
    using vecfield_type      = VecField<FieldT, PhysicalQuantityT>;
    using scalar_qty         = typename PhysicalQuantityT::Scalar;
    using vector_qty         = typename PhysicalQuantityT::Vector;
    using scalar_bc_type     = FieldInnerBoundaryCondition<FieldT, GridLayoutT, PhysicalStateT>;
    using vector_bc_type
        = FieldInnerBoundaryCondition<vecfield_type, GridLayoutT, PhysicalStateT>;
    using scalar_bc_map_type = std::unordered_map<scalar_qty, std::unique_ptr<scalar_bc_type>>;
    using vector_bc_map_type = std::unordered_map<vector_qty, std::unique_ptr<vector_bc_type>>;

    InnerBoundaryConditionFactory() = delete;

    /**
     * @brief Populate @p scalarBCs and @p vectorBCs following the rules for @p type.
     *
     * @param type             Inner boundary condition type.
     * @param scalarQuantities Scalar quantities that need a BC.
     * @param vectorQuantities Vector quantities that need a BC.
     * @param scalarBCs        Output map: scalar quantity → BC (written in place).
     * @param vectorBCs        Output map: vector quantity → BC (written in place).
     */
    static void create(InnerBoundaryConditionType type,
                       std::vector<scalar_qty> const& scalarQuantities,
                       std::vector<vector_qty> const& vectorQuantities,
                       scalar_bc_map_type& scalarBCs, vector_bc_map_type& vectorBCs)
    {
        switch (type)
        {
            case InnerBoundaryConditionType::Reflective:
                register_reflective_(scalarQuantities, vectorQuantities, scalarBCs, vectorBCs);
                break;
            default:
                throw std::runtime_error("InnerBoundaryConditionFactory: unknown type");
        }
    }

private:
    template<typename ScalarOrTensorFieldT>
    using Neumann
        = FieldNeumannInnerBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, PhysicalStateT>;

    template<typename ScalarOrTensorFieldT>
    using Symmetric
        = FieldSymmetricInnerBoundaryCondition<ScalarOrTensorFieldT, GridLayoutT, PhysicalStateT>;

    template<typename ScalarOrTensorFieldT>
    using Antisymmetric = FieldAntisymmetricInnerBoundaryCondition<ScalarOrTensorFieldT,
                                                                    GridLayoutT, PhysicalStateT>;

    /**
     * @brief Reflective body BC rules:
     *   scalars → Neumann
     *   B       → Symmetric      (specular reflection)
     *   rhoV    → Symmetric      (no normal flow, free-slip)
     *   E       → Antisymmetric  (tangential E = 0)
     *   others  → Neumann
     */
    static void register_reflective_(std::vector<scalar_qty> const& scalars,
                                     std::vector<vector_qty> const& vectors,
                                     scalar_bc_map_type& scalarBCs,
                                     vector_bc_map_type& vectorBCs)
    {
        for (auto const qty : scalars)
            scalarBCs[qty] = std::make_unique<Neumann<FieldT>>();

        for (auto const qty : vectors)
        {
            switch (qty)
            {
                case PhysicalQuantityT::Vector::B:
                    vectorBCs[qty] = std::make_unique<Symmetric<vecfield_type>>();
                    break;
                case PhysicalQuantityT::Vector::rhoV:
                    vectorBCs[qty] = std::make_unique<Symmetric<vecfield_type>>();
                    break;
                case PhysicalQuantityT::Vector::E:
                    vectorBCs[qty] = std::make_unique<Antisymmetric<vecfield_type>>();
                    break;
                default:
                    vectorBCs[qty] = std::make_unique<Neumann<vecfield_type>>();
                    break;
            }
        }
    }
};

} // namespace PHARE::core

#endif // PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_CONDITION_FACTORY_HPP
