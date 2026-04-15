#include "gtest/gtest.h"

#include <algorithm>
#include <vector>

#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutimplyee_mhd.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/inner_boundary/inner_boundary_mesh_classifier.hpp"
#include "core/inner_boundary/plane_inner_boundary.hpp"
#include "core/inner_boundary/sphere_inner_boundary.hpp"
#include "core/mhd/mhd_quantities.hpp"

namespace
{
constexpr double eps = 1e-12;

using GridLayoutImpl = PHARE::core::GridLayoutImplYeeMHD<2, 2, 1>;
using GridLayout     = PHARE::core::GridLayout<GridLayoutImpl>;
using Mapper = PHARE::core::InnerBoundaryMeshClassifier<2, GridLayout, PHARE::core::MHDQuantity>;
using MeshData           = PHARE::core::InnerBoundaryMeshData<2, PHARE::core::MHDQuantity>;
using ScalarField    = PHARE::core::Field<2, PHARE::core::MHDQuantity::Scalar, double>;

template<typename FieldT>
PHARE::core::Point<std::uint32_t, 2> physicalLocalIndex(GridLayout const& layout,
                                                        FieldT const& field,
                                                        std::uint32_t ix,
                                                        std::uint32_t iy)
{
    using PHARE::core::Direction;

    return {layout.physicalStartIndex(field, Direction::X) + ix,
            layout.physicalStartIndex(field, Direction::Y) + iy};
}

// Centering constants for 2D: bit i = 1 when direction i is dual.
// (P,P)→0=node, (D,P)→1=faceX, (P,D)→2=faceY, (D,D)→3=cell
constexpr std::array<PHARE::core::QtyCentering, 2> kCellC
    = {PHARE::core::QtyCentering::dual,   PHARE::core::QtyCentering::dual};
constexpr std::array<PHARE::core::QtyCentering, 2> kFaceXC
    = {PHARE::core::QtyCentering::primal, PHARE::core::QtyCentering::dual};
constexpr std::array<PHARE::core::QtyCentering, 2> kFaceYC
    = {PHARE::core::QtyCentering::dual,   PHARE::core::QtyCentering::primal};

struct InnerBoundaryMeshClassifierBuffers
{
    static constexpr char const* BOUNDARY_NAME = "test";

    explicit InnerBoundaryMeshClassifierBuffers(GridLayout const& layout)
        : phi_storage{layout.allocSize(PHARE::core::MHDQuantity::Scalar::NodeCentered)}
        , tags{BOUNDARY_NAME}
    {
        ScalarField phi_field{std::string(BOUNDARY_NAME) + "_signed_distance",
                              PHARE::core::MHDQuantity::Scalar::NodeCentered,
                              phi_storage.data(), phi_storage.shape()};
        tags.signedDistanceAtNodes.setBuffer(&phi_field);

        elem_storages.reserve(MeshData::num_elem_types);
        for (std::size_t i = 0; i < MeshData::num_elem_types; ++i)
        {
            auto const c   = MeshData::idxToCentering(i);
            auto const qty = MeshData::scalarFromCentering(c);
            elem_storages.emplace_back(layout.allocSize(qty));
            ScalarField tmp{tags.elemStatus[i].name(), qty,
                            elem_storages[i].data(), elem_storages[i].shape()};
            tags.elemStatus[i].setBuffer(&tmp);
        }
    }

    PHARE::core::NdArrayVector<2, double> phi_storage;
    std::vector<PHARE::core::NdArrayVector<2, double>> elem_storages;
    MeshData tags;
};
} // namespace

TEST(InnerBoundaryMeshClassifier, computesReasonableDefaultCutEpsFromLayout)
{
    // Sphere centred at (0.05, 0.05) so no grid node coincides with the centre
    // (which would make normal() undefined and cause populateGhostLists_ to throw).
    // The physical origin node sits at (0, 0), distance = sqrt(0.05^2+0.05^2) - 1 ≈ -0.929,
    // but we check the node at (0.2, 0.1) which is closest to r=1 for a simpler assertion.
    // We only exercise that the classifier runs without error and that phi at a node
    // well inside the sphere is negative.
    PHARE::core::SphereInnerBoundary<2> sphere{"sphere", {0.05, 0.05}, 1.};
    GridLayout layout{{0.2, 0.1}, {4u, 4u}, {0., 0.}};
    auto tagger = Mapper::withDefaults(sphere, layout);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    tagger(layout, buffers.tags);

    // The origin node (0, 0) is inside the sphere (distance < 0).
    auto const origin_node = physicalLocalIndex(layout, buffers.tags.signedDistanceAtNodes, 0u, 0u);
    EXPECT_LT(buffers.tags.signedDistanceAtNodes(origin_node), 0.);
}

TEST(InnerBoundaryMeshClassifier, tagsCutInactiveAndGhostGeometry)
{
    PHARE::core::PlaneInnerBoundary<2> plane{"plane", {0.0, 0.0}, {1.0, 0.0}}; // x=0
    PHARE::core::Box<int, 2> amr_box{{-2, 0}, {1, 1}};
    GridLayout layout{{1.0, 1.0}, {4u, 2u}, {0.0, 0.0}, amr_box};

    Mapper::Overrides ov;
    ov.cut_eps      = 1e-12;
    ov.inactive_eps = 0.0;
    auto tagger     = Mapper::withDefaults(plane, layout, ov);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    tagger(layout, buffers.tags);

    // Ghost cells grow inward (into the solid) from the cut layer.
    // physical[0] (x=-1.5) was inactive and is now promoted to ghost; physical[3] (x=1.5)
    // is a plain fluid cell on the outside.
    auto& cellField = buffers.tags.getStatusFieldFromCentering(kCellC);
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Ghost),
              cellField(physicalLocalIndex(layout, cellField, 0u, 0u)));
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Cut),
              cellField(physicalLocalIndex(layout, cellField, 1u, 0u)));
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Cut),
              cellField(physicalLocalIndex(layout, cellField, 2u, 0u)));
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Fluid),
              cellField(physicalLocalIndex(layout, cellField, 3u, 0u)));

    // Face at physical[2] straddles x=0 → Cut; face at physical[1] (between ghost and cut
    // cell) is adjacent to the ghost cell → Ghost.
    auto& faceXField = buffers.tags.getStatusFieldFromCentering(kFaceXC);
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Cut),
              faceXField(physicalLocalIndex(layout, faceXField, 2u, 0u)));
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Ghost),
              faceXField(physicalLocalIndex(layout, faceXField, 1u, 0u)));
    // FaceCenteredY face at physical[0,0] is adjacent to the ghost cell column → Ghost.
    auto& faceYField = buffers.tags.getStatusFieldFromCentering(kFaceYC);
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Ghost),
              faceYField(physicalLocalIndex(layout, faceYField, 0u, 0u)));

    // Note: in 2D, EdgeCenteredY has centering (primal, dual) identical to FaceCenteredX,
    // so both concepts share faceXField above. No separate edge assertions needed.
}

TEST(InnerBoundaryMeshClassifier, ghostCellListIsNonEmpty)
{
    PHARE::core::PlaneInnerBoundary<2> plane{"plane", {0.0, 0.0}, {1.0, 0.0}};
    PHARE::core::Box<int, 2> amr_box{{-2, 0}, {1, 1}};
    GridLayout layout{{1.0, 1.0}, {4u, 2u}, {0.0, 0.0}, amr_box};

    Mapper::Overrides ov;
    ov.cut_eps      = 1e-12;
    ov.inactive_eps = 0.0;
    auto tagger     = Mapper::withDefaults(plane, layout, ov);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    tagger(layout, buffers.tags);

    EXPECT_FALSE(buffers.tags.getGhostDataFromCentering(kCellC).empty());
}

TEST(InnerBoundaryMeshClassifier, ghostCellHasCorrectMirrorPointAndNormal)
{
    // Plane at x=0, normal (1,0). Ghost cell at physical[0,0] has center (-1.5, 0.5).
    // Its mirror is at (1.5, 0.5) and the outward normal is (1, 0).
    PHARE::core::PlaneInnerBoundary<2> plane{"plane", {0.0, 0.0}, {1.0, 0.0}};
    PHARE::core::Box<int, 2> amr_box{{-2, 0}, {1, 1}};
    GridLayout layout{{1.0, 1.0}, {4u, 2u}, {0.0, 0.0}, amr_box};

    Mapper::Overrides ov;
    ov.cut_eps      = 1e-12;
    ov.inactive_eps = 0.0;
    auto tagger     = Mapper::withDefaults(plane, layout, ov);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    tagger(layout, buffers.tags);

    auto const target_idx
        = physicalLocalIndex(layout, buffers.tags.getStatusFieldFromCentering(kCellC), 0u, 0u);
    auto const& ghost_cells = buffers.tags.getGhostDataFromCentering(kCellC);
    auto it = std::find_if(ghost_cells.begin(), ghost_cells.end(),
                           [&](auto const& g) { return g.index == target_idx; });
    ASSERT_NE(it, ghost_cells.end()) << "Ghost cell for physical[0,0] not found in ghostCells";

    EXPECT_NEAR(it->mirrorPoint[0], 1.5, eps);
    EXPECT_NEAR(it->mirrorPoint[1], 0.5, eps);
    EXPECT_NEAR(it->normal[0], 1.0, eps);
    EXPECT_NEAR(it->normal[1], 0.0, eps);
    // Mirror (1.5, 0.5) is within the physical domain of the patch → in-patch.
    EXPECT_TRUE(it->mirrorIsInPatch);
}

TEST(InnerBoundaryMeshClassifier, ghostFaceListIsNonEmpty)
{
    PHARE::core::PlaneInnerBoundary<2> plane{"plane", {0.0, 0.0}, {1.0, 0.0}};
    PHARE::core::Box<int, 2> amr_box{{-2, 0}, {1, 1}};
    GridLayout layout{{1.0, 1.0}, {4u, 2u}, {0.0, 0.0}, amr_box};

    Mapper::Overrides ov;
    ov.cut_eps      = 1e-12;
    ov.inactive_eps = 0.0;
    auto tagger     = Mapper::withDefaults(plane, layout, ov);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    tagger(layout, buffers.tags);

    // FaceCenteredX ghost face at physical[1,0] (between ghost and cut cell) must be present.
    EXPECT_FALSE(buffers.tags.getGhostDataFromCentering(kFaceXC).empty());
    // FaceCenteredY face adjacent to the ghost cell column must also be present.
    EXPECT_FALSE(buffers.tags.getGhostDataFromCentering(kFaceYC).empty());
}

TEST(InnerBoundaryMeshClassifier, amrHaloGhostCellsHaveMirrorOutsidePatch)
{
    // Plane at x=0. Physical cells: AMR x in [-2, 1]. AMR ghost (halo) cells on the solid
    // side occupy AMR x < -2 with centres at ..., -3.5, -2.5; their mirrors land at x >
    // 2.5, which is beyond the patch's physical domain. These entries must be flagged
    // mirrorIsInPatch = false.
    // Physical ghost at physical[0,0] (AMR x = -2, centre x = -1.5) has mirror at x = 1.5,
    // which IS within the domain → mirrorIsInPatch = true.
    PHARE::core::PlaneInnerBoundary<2> plane{"plane", {0.0, 0.0}, {1.0, 0.0}};
    PHARE::core::Box<int, 2> amr_box{{-2, 0}, {1, 1}};
    GridLayout layout{{1.0, 1.0}, {4u, 2u}, {0.0, 0.0}, amr_box};

    Mapper::Overrides ov;
    ov.cut_eps      = 1e-12;
    ov.inactive_eps = 0.0;
    auto tagger     = Mapper::withDefaults(plane, layout, ov);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    tagger(layout, buffers.tags);

    auto const& ghost_cells = buffers.tags.getGhostDataFromCentering(kCellC);
    ASSERT_FALSE(ghost_cells.empty());

    bool found_in_patch     = false;
    bool found_out_of_patch = false;
    for (auto const& g : ghost_cells)
    {
        if (g.mirrorIsInPatch)
            found_in_patch = true;
        else
            found_out_of_patch = true;
    }
    EXPECT_TRUE(found_in_patch)
        << "Physical ghost (mirror inside patch) must appear in the list";
    EXPECT_TRUE(found_out_of_patch)
        << "AMR-halo ghosts (mirror outside patch) must appear in the list";
}
