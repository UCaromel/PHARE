#ifndef PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_MANAGER_HPP
#define PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_MANAGER_HPP

#include "core/data/field/field_traits.hpp"
#include "core/data/grid/gridlayout_traits.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/inner_boundary/inner_bc_context.hpp"
#include "core/inner_boundary/inner_boundary_condition_factory.hpp"
#include "core/inner_boundary/inner_boundary_factory.hpp"
#include "core/inner_boundary/inner_boundary_geometry.hpp"
#include "core/inner_boundary/inner_boundary_mesh_classifier.hpp"
#include "core/inner_boundary/inner_boundary_mesh_data.hpp"
#include "initializer/data_provider.hpp"

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace PHARE::core
{

/**
 * @brief Manages the lifecycle and application of inner boundary conditions.
 *
 * Owns the embedded boundary geometry, the per-patch mesh classification data,
 * and the per-quantity BC objects. Exposes:
 *   - classify(layout)                   — fill mesh data (call once per patch setup)
 *   - applyBC(Scalar, field, layout, ctx) — enforce BC on a scalar quantity
 *   - applyBC(Vector, field, layout, ctx) — enforce BC on a vector quantity
 *   - ResourcesUser interface             — expose mesh data to SAMRAI resource manager
 *
 * @tparam PhysicalQuantityT  Quantity traits (MHDQuantity, HybridQuantity, …).
 * @tparam FieldT             Scalar field type.
 * @tparam GridLayoutT        Grid layout type.
 * @tparam PhysicalStateT     Physical state type forwarded to BC appliers.
 */
template<typename PhysicalQuantityT, IsField FieldT, IsGridLayout GridLayoutT,
         typename PhysicalStateT>
class InnerBoundaryManager
{
public:
    static constexpr std::size_t dimension = GridLayoutT::dimension;

    using vecfield_type      = VecField<FieldT, PhysicalQuantityT>;
    using scalar_qty         = typename PhysicalQuantityT::Scalar;
    using vector_qty         = typename PhysicalQuantityT::Vector;
    using geometry_type      = InnerBoundaryGeometry<dimension>;
    using mesh_data_type     = InnerBoundaryMeshData<dimension, PhysicalQuantityT>;
    using classifier_type    = InnerBoundaryMeshClassifier<dimension, GridLayoutT, PhysicalQuantityT>;
    using context_type       = InnerBCContext<PhysicalStateT>;
    using scalar_bc_type     = FieldInnerBoundaryCondition<FieldT, GridLayoutT, PhysicalStateT>;
    using vector_bc_type
        = FieldInnerBoundaryCondition<vecfield_type, GridLayoutT, PhysicalStateT>;
    using scalar_bc_map_type = std::unordered_map<scalar_qty, std::unique_ptr<scalar_bc_type>>;
    using vector_bc_map_type = std::unordered_map<vector_qty, std::unique_ptr<vector_bc_type>>;
    using factory_type
        = InnerBoundaryConditionFactory<PhysicalQuantityT, FieldT, GridLayoutT, PhysicalStateT>;

    InnerBoundaryManager() = delete;

    /**
     * @brief Construct from pre-built geometry and BC condition type.
     *
     * @param geometry         Embedded boundary geometry (transferred ownership).
     * @param conditionType    Physics preset for per-quantity BC assignment.
     * @param scalarQuantities Scalar quantities that need a BC.
     * @param vectorQuantities Vector quantities that need a BC.
     */
    InnerBoundaryManager(std::unique_ptr<geometry_type> geometry,
                         InnerBoundaryConditionType conditionType,
                         std::vector<scalar_qty> const& scalarQuantities,
                         std::vector<vector_qty> const& vectorQuantities)
        : geometry_{std::move(geometry)}
        , meshData_{geometry_->name()}
    {
        factory_type::create(conditionType, scalarQuantities, vectorQuantities, scalarBCs_,
                             vectorBCs_);
    }

    /**
     * @brief Static factory: create a manager from a simulation dict, or return nullptr.
     *
     * Returns nullptr when the dict contains no "inner_boundary" key, matching
     * the behaviour of InnerBoundaryFactory::create().
     */
    static std::unique_ptr<InnerBoundaryManager>
    create(initializer::PHAREDict const& dict, std::vector<scalar_qty> const& scalarQuantities,
           std::vector<vector_qty> const& vectorQuantities)
    {
        if (!dict.contains("inner_boundary"))
            return nullptr;

        auto geometry = InnerBoundaryFactory<dimension>::create(dict);
        if (!geometry)
            return nullptr;

        auto const& ibDict    = dict["inner_boundary"];
        auto const typeName   = ibDict["condition_type"].template to<std::string>();
        auto const condType   = getInnerBoundaryConditionTypeFromString(typeName);

        return std::make_unique<InnerBoundaryManager>(std::move(geometry), condType,
                                                      scalarQuantities, vectorQuantities);
    }

    // -------------------------------------------------------------------------
    //  Classification
    // -------------------------------------------------------------------------

    /**
     * @brief Classify mesh elements relative to the embedded boundary.
     *
     * Must be called after setBuffer() has been invoked on the mesh data
     * (i.e., after SAMRAI has allocated the patch data).
     *
     * @param layout Grid layout of the current patch.
     */
    void classify(GridLayoutT const& layout)
    {
        auto classifier = classifier_type::withDefaults(*geometry_, layout);
        classifier(layout, meshData_);
    }

    // -------------------------------------------------------------------------
    //  BC application
    // -------------------------------------------------------------------------

    /**
     * @brief Apply the inner BC for a scalar quantity.
     *
     * @param qty    Scalar quantity identifying which BC to use.
     * @param field  Field whose ghost cells are updated in place.
     * @param layout Grid layout of the current patch.
     * @param ctx    State context (statenew, state, time, dt).
     */
    void applyBC(scalar_qty qty, FieldT& field, GridLayoutT const& layout,
                 context_type const& ctx)
    {
        auto it = scalarBCs_.find(qty);
        if (it == scalarBCs_.end())
            return; // quantity not registered — no-op
        it->second->apply(field, layout, meshData_, ctx);
    }

    /**
     * @brief Apply the inner BC for a vector quantity.
     *
     * @param qty      Vector quantity identifying which BC to use.
     * @param vecfield VecField whose ghost cells are updated in place.
     * @param layout   Grid layout of the current patch.
     * @param ctx      State context (statenew, state, time, dt).
     */
    void applyBC(vector_qty qty, vecfield_type& vecfield, GridLayoutT const& layout,
                 context_type const& ctx)
    {
        auto it = vectorBCs_.find(qty);
        if (it == vectorBCs_.end())
            return; // quantity not registered — no-op
        it->second->apply(vecfield, layout, meshData_, ctx);
    }

    // -------------------------------------------------------------------------
    //  Mesh data accessor
    // -------------------------------------------------------------------------

    NO_DISCARD mesh_data_type& getMeshData() { return meshData_; }

    NO_DISCARD mesh_data_type const& getMeshData() const { return meshData_; }

    // -------------------------------------------------------------------------
    //  ResourcesUser interface (exposes mesh data to SAMRAI resource manager)
    // -------------------------------------------------------------------------

    NO_DISCARD bool isUsable() const { return meshData_.isUsable(); }

    NO_DISCARD bool isSettable() const { return meshData_.isSettable(); }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(meshData_);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(meshData_);
    }

private:
    std::unique_ptr<geometry_type> geometry_;
    mesh_data_type                 meshData_;
    scalar_bc_map_type             scalarBCs_;
    vector_bc_map_type             vectorBCs_;
};

} // namespace PHARE::core

#endif // PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_MANAGER_HPP
