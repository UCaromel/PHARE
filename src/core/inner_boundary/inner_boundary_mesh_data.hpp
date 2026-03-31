#ifndef PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_MESH_DATA_HPP
#define PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_MESH_DATA_HPP

#include "core/data/field/field.hpp"
#include "core/data/vecfield/vecfield.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"
#include "core/utilities/point/point.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <tuple>
#include <vector>

namespace PHARE::core
{
/**
 * @brief Status of a mesh element (cell, face, or edge) relative to an inner boundary.
 *
 * - **Fluid**    — element lies entirely in the fluid domain.
 * - **Cut**      — element straddles the boundary surface.
 * - **Ghost**    — element lies inside the body but is used to enforce the boundary condition
 *                  by mirror-point interpolation from the fluid side.
 * - **Inactive** — element lies entirely inside the body and plays no role in the solver.
 */
enum class ElemStatus : std::uint8_t { Fluid, Cut, Ghost, Inactive };

/// Convert an ElemStatus value to its double encoding for field storage.
inline constexpr double toDouble(ElemStatus s)
{
    return static_cast<double>(static_cast<std::uint8_t>(s));
}


/**
 * @brief Precomputed per-ghost-element data used by the BC applier every time step.
 *
 * Storing these avoids recomputing expensive boundary queries (normal, symmetric)
 * once per field per ghost element per time step.
 *
 * @note `mirrorIsInPatch` is set to `false` when the mirror point lies outside
 * the current patch's AMR box. This can happen for ghost elements in the AMR
 * halo region. When `false`, the BC applier must skip the interpolation — the
 * ghost value will instead be filled by AMR communication.
 */
template<std::size_t dim>
struct GhostElemData
{
    Point<std::uint32_t, dim> index;          ///< Local array index of the ghost element.
    Point<double, dim>        mirrorPoint;    ///< Physical coords of the symmetric point in the fluid.
    Point<double, dim>        normal;         ///< Unit outward normal at the boundary (ghost → mirror).
    bool                      mirrorIsInPatch; ///< True iff the mirror point lies within this patch.
};


/**
 * @brief Bundle of node level-set values and mesh status data around an inner boundary.
 *
 * @tparam dim Spatial dimension.
 * @tparam PhysicalQuantityT Quantity traits used to define node/cell/face/edge
 * field types.
 */
template<std::size_t dim, typename PhysicalQuantityT>
struct InnerBoundaryMeshData
{
    using field_type    = Field<dim, typename PhysicalQuantityT::Scalar, double>;
    using vecfield_type = VecField<field_type, PhysicalQuantityT>;
    using ghost_elem_data_type    = GhostElemData<dim>;


    InnerBoundaryMeshData(std::string boundaryName = "")
        : signedDistanceAtNodes{boundaryName + "_signed_distance",
                                PhysicalQuantityT::Scalar::NodeCentered}
        , cellStatus{boundaryName + "_cell_status", PhysicalQuantityT::Scalar::CellCentered}
        , faceStatus{boundaryName + "_face_status", PhysicalQuantityT::Vector::FaceCentered}
        , edgeStatus{boundaryName + "_edge_status", PhysicalQuantityT::Vector::EdgeCentered}
    {
    }

    field_type signedDistanceAtNodes; ///< Signed distance to the boundary at nodes.
    field_type cellStatus;            ///< Per-cell fluid/cut/ghost/inactive classification.
    vecfield_type faceStatus;         ///< Per-face fluid/cut/ghost/inactive classification.
    vecfield_type edgeStatus;         ///< Per-edge fluid/cut/ghost/inactive classification.

    /// Ghost cells populated by the classifier; iterated by the BC applier every time step.
    std::vector<ghost_elem_data_type> ghostCellsData;
    /// Ghost faces per direction, populated by the classifier.
    std::array<std::vector<ghost_elem_data_type>, dim> ghostFacesData;
    /// Ghost edges per direction, populated by the classifier.
    std::array<std::vector<ghost_elem_data_type>, dim> ghostEdgesData;


    /**
     * @brief Return the status field for the mesh element type matching @p centering.
     *
     * - All dual  → cellStatus
     * - One primal → faceStatus[primal_dir]
     * - All primal (or one dual in 3D) → edgeStatus[dual_dir]
     *
     * @param centering Per-dimension centering of the field of interest.
     */
    field_type& getStatusFieldFromCentering(std::array<QtyCentering, dim> const& centering)
    {
        auto const n_primal
            = std::count(centering.begin(), centering.end(), QtyCentering::primal);

        if (n_primal == 0)
            return cellStatus;

        if (n_primal == 1)
        {
            auto const dir = std::distance(
                centering.begin(),
                std::find(centering.begin(), centering.end(), QtyCentering::primal));
            return faceStatus[dir];
        }

        auto const dir = std::distance(
            centering.begin(),
            std::find(centering.begin(), centering.end(), QtyCentering::dual));
        return edgeStatus[dir];
    }

    field_type const& getStatusFieldFromCentering(
        std::array<QtyCentering, dim> const& centering) const
    {
        auto const n_primal
            = std::count(centering.begin(), centering.end(), QtyCentering::primal);

        if (n_primal == 0)
            return cellStatus;

        if (n_primal == 1)
        {
            auto const dir = std::distance(
                centering.begin(),
                std::find(centering.begin(), centering.end(), QtyCentering::primal));
            return faceStatus[dir];
        }

        auto const dir = std::distance(
            centering.begin(),
            std::find(centering.begin(), centering.end(), QtyCentering::dual));
        return edgeStatus[dir];
    }

    /**
     * @brief Return a tuple of status field references matching the three centerings of a
     *        VecField or TensorField component triplet.
     *
     * @param centerings Three per-dimension centering arrays, one per component.
     */
    auto getStatusFieldsFromCenterings(
        std::array<std::array<QtyCentering, dim>, 3> const& centerings)
        -> std::tuple<field_type&, field_type&, field_type&>
    {
        return std::forward_as_tuple(getStatusFieldFromCentering(centerings[0]),
                                     getStatusFieldFromCentering(centerings[1]),
                                     getStatusFieldFromCentering(centerings[2]));
    }

    auto getStatusFieldsFromCenterings(
        std::array<std::array<QtyCentering, dim>, 3> const& centerings) const
        -> std::tuple<field_type const&, field_type const&, field_type const&>
    {
        return std::forward_as_tuple(getStatusFieldFromCentering(centerings[0]),
                                     getStatusFieldFromCentering(centerings[1]),
                                     getStatusFieldFromCentering(centerings[2]));
    }


    /**
     * @brief Return the ghost data list for the mesh element type matching @p centering.
     *
     * Mirrors the centering logic of getStatusFieldFromCentering:
     * - All dual  → ghostCellsData
     * - One primal → ghostFacesData[primal_dir]
     * - All primal (or one dual in 3D) → ghostEdgesData[dual_dir]
     *
     * @param centering Per-dimension centering of the field of interest.
     */
    std::vector<ghost_elem_data_type>&
    getGhostDataFromCentering(std::array<QtyCentering, dim> const& centering)
    {
        auto const n_primal
            = std::count(centering.begin(), centering.end(), QtyCentering::primal);

        if (n_primal == 0)
            return ghostCellsData;

        if (n_primal == 1)
        {
            auto const dir = std::distance(
                centering.begin(),
                std::find(centering.begin(), centering.end(), QtyCentering::primal));
            return ghostFacesData[dir];
        }

        auto const dir = std::distance(
            centering.begin(),
            std::find(centering.begin(), centering.end(), QtyCentering::dual));
        return ghostEdgesData[dir];
    }

    std::vector<ghost_elem_data_type> const&
    getGhostDataFromCentering(std::array<QtyCentering, dim> const& centering) const
    {
        auto const n_primal
            = std::count(centering.begin(), centering.end(), QtyCentering::primal);

        if (n_primal == 0)
            return ghostCellsData;

        if (n_primal == 1)
        {
            auto const dir = std::distance(
                centering.begin(),
                std::find(centering.begin(), centering.end(), QtyCentering::primal));
            return ghostFacesData[dir];
        }

        auto const dir = std::distance(
            centering.begin(),
            std::find(centering.begin(), centering.end(), QtyCentering::dual));
        return ghostEdgesData[dir];
    }

    /**
     * @brief Return a tuple of ghost data list references matching the three centerings of a
     *        VecField or TensorField component triplet.
     *
     * @param centerings Three per-dimension centering arrays, one per component.
     */
    auto getGhostDataFromCenterings(
        std::array<std::array<QtyCentering, dim>, 3> const& centerings)
        -> std::tuple<std::vector<ghost_elem_data_type>&, std::vector<ghost_elem_data_type>&,
                      std::vector<ghost_elem_data_type>&>
    {
        return std::forward_as_tuple(getGhostDataFromCentering(centerings[0]),
                                     getGhostDataFromCentering(centerings[1]),
                                     getGhostDataFromCentering(centerings[2]));
    }

    auto getGhostDataFromCenterings(
        std::array<std::array<QtyCentering, dim>, 3> const& centerings) const
        -> std::tuple<std::vector<ghost_elem_data_type> const&,
                      std::vector<ghost_elem_data_type> const&,
                      std::vector<ghost_elem_data_type> const&>
    {
        return std::forward_as_tuple(getGhostDataFromCentering(centerings[0]),
                                     getGhostDataFromCentering(centerings[1]),
                                     getGhostDataFromCentering(centerings[2]));
    }


    //-------------------------------------------------------------------------
    //                  start the ResourcesUser interface
    //-------------------------------------------------------------------------

    NO_DISCARD bool isUsable() const
    {
        return isUsable(signedDistanceAtNodes, cellStatus, faceStatus, edgeStatus);
    }

    NO_DISCARD bool isSettable() const
    {
        return isSettable(signedDistanceAtNodes, cellStatus, faceStatus, edgeStatus);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(signedDistanceAtNodes, cellStatus, faceStatus, edgeStatus);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::forward_as_tuple(signedDistanceAtNodes, cellStatus, faceStatus, edgeStatus);
    }

    //-------------------------------------------------------------------------
    //                  ends the ResourcesUser interface
    //-------------------------------------------------------------------------
};
} // namespace PHARE::core

#endif // PHARE_CORE_INNER_BOUNDARY_INNER_BOUNDARY_MESH_DATA_HPP
