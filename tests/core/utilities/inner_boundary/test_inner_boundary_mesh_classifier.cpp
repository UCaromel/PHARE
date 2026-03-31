#include "gtest/gtest.h"

#include <algorithm>

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
using NodeField      = PHARE::core::Field<2, PHARE::core::MHDQuantity::Scalar, double>;
using CellField      = PHARE::core::Field<2, PHARE::core::MHDQuantity::Scalar, double>;
using FaceField      = PHARE::core::Field<2, PHARE::core::MHDQuantity::Scalar, double>;
using FaceVec        = PHARE::core::VecField<FaceField, PHARE::core::MHDQuantity>;
using EdgeField      = PHARE::core::Field<2, PHARE::core::MHDQuantity::Scalar, double>;
using EdgeVec        = PHARE::core::VecField<EdgeField, PHARE::core::MHDQuantity>;

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

struct InnerBoundaryMeshClassifierBuffers
{
    // InnerBoundaryMeshData names its fields as "<boundary>_<component>"; keep this in sync.
    static constexpr char const* BOUNDARY_NAME = "test";

    explicit InnerBoundaryMeshClassifierBuffers(GridLayout const& layout)
        : phi_storage{layout.allocSize(PHARE::core::MHDQuantity::Scalar::NodeCentered)}
        , cell_storage{layout.allocSize(PHARE::core::MHDQuantity::Scalar::CellCentered)}
        , face_x_storage{layout.allocSize(PHARE::core::MHDQuantity::Scalar::FaceCenteredX)}
        , face_y_storage{layout.allocSize(PHARE::core::MHDQuantity::Scalar::FaceCenteredY)}
        , face_z_storage{layout.allocSize(PHARE::core::MHDQuantity::Scalar::FaceCenteredZ)}
        , face_fields{FaceField{std::string(BOUNDARY_NAME) + "_face_status_x",
                                PHARE::core::MHDQuantity::Scalar::FaceCenteredX,
                                face_x_storage.data(), face_x_storage.shape()},
                      FaceField{std::string(BOUNDARY_NAME) + "_face_status_y",
                                PHARE::core::MHDQuantity::Scalar::FaceCenteredY,
                                face_y_storage.data(), face_y_storage.shape()},
                      FaceField{std::string(BOUNDARY_NAME) + "_face_status_z",
                                PHARE::core::MHDQuantity::Scalar::FaceCenteredZ,
                                face_z_storage.data(), face_z_storage.shape()}}
        , edge_x_storage{layout.allocSize(PHARE::core::MHDQuantity::Scalar::EdgeCenteredX)}
        , edge_y_storage{layout.allocSize(PHARE::core::MHDQuantity::Scalar::EdgeCenteredY)}
        , edge_z_storage{layout.allocSize(PHARE::core::MHDQuantity::Scalar::EdgeCenteredZ)}
        , edge_fields{EdgeField{std::string(BOUNDARY_NAME) + "_edge_status_x",
                                PHARE::core::MHDQuantity::Scalar::EdgeCenteredX,
                                edge_x_storage.data(), edge_x_storage.shape()},
                      EdgeField{std::string(BOUNDARY_NAME) + "_edge_status_y",
                                PHARE::core::MHDQuantity::Scalar::EdgeCenteredY,
                                edge_y_storage.data(), edge_y_storage.shape()},
                      EdgeField{std::string(BOUNDARY_NAME) + "_edge_status_z",
                                PHARE::core::MHDQuantity::Scalar::EdgeCenteredZ,
                                edge_z_storage.data(), edge_z_storage.shape()}}
        , tags{BOUNDARY_NAME}
    {
        NodeField phi_field{std::string(BOUNDARY_NAME) + "_signed_distance",
                            PHARE::core::MHDQuantity::Scalar::NodeCentered,
                            phi_storage.data(), phi_storage.shape()};
        tags.signedDistanceAtNodes.setBuffer(&phi_field);

        CellField cell_field{std::string(BOUNDARY_NAME) + "_cell_status",
                             PHARE::core::MHDQuantity::Scalar::CellCentered,
                             cell_storage.data(), cell_storage.shape()};
        tags.cellStatus.setBuffer(&cell_field);

        tags.faceStatus.setBuffer(&face_fields);
        tags.edgeStatus.setBuffer(&edge_fields);
    }

    PHARE::core::NdArrayVector<2, double> phi_storage;

    PHARE::core::NdArrayVector<2, double> cell_storage;

    PHARE::core::NdArrayVector<2, double> face_x_storage;
    PHARE::core::NdArrayVector<2, double> face_y_storage;
    PHARE::core::NdArrayVector<2, double> face_z_storage;
    std::array<FaceField, 3> face_fields;

    PHARE::core::NdArrayVector<2, double> edge_x_storage;
    PHARE::core::NdArrayVector<2, double> edge_y_storage;
    PHARE::core::NdArrayVector<2, double> edge_z_storage;
    std::array<EdgeField, 3> edge_fields;
    MeshData tags;
};
} // namespace

TEST(InnerBoundaryMeshClassifier, computesReasonableDefaultCutEpsFromLayout)
{
    PHARE::core::SphereInnerBoundary<2> sphere{"sphere", {0., 0.}, 1.};
    GridLayout layout{{0.2, 0.1}, {4u, 4u}, {0., 0.}};
    auto tagger = Mapper::withDefaults(sphere, layout);

    InnerBoundaryMeshClassifierBuffers buffers{layout};
    tagger(layout, buffers.tags);

    auto const origin_node = physicalLocalIndex(layout, buffers.tags.signedDistanceAtNodes, 0u, 0u);
    EXPECT_NEAR(buffers.tags.signedDistanceAtNodes(origin_node), -1., eps);
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
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Ghost),
              buffers.tags.cellStatus(physicalLocalIndex(layout, buffers.tags.cellStatus, 0u, 0u)));
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Cut),
              buffers.tags.cellStatus(physicalLocalIndex(layout, buffers.tags.cellStatus, 1u, 0u)));
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Cut),
              buffers.tags.cellStatus(physicalLocalIndex(layout, buffers.tags.cellStatus, 2u, 0u)));
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Fluid),
              buffers.tags.cellStatus(physicalLocalIndex(layout, buffers.tags.cellStatus, 3u, 0u)));

    // Face at physical[2] straddles x=0 → Cut; face at physical[1] (between ghost and cut
    // cell) is adjacent to the ghost cell → Ghost.
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Cut),
              buffers.tags.faceStatus[0](physicalLocalIndex(layout, buffers.tags.faceStatus[0], 2u, 0u)));
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Ghost),
              buffers.tags.faceStatus[0](physicalLocalIndex(layout, buffers.tags.faceStatus[0], 1u, 0u)));
    // FaceCenteredY face at physical[0,0] is adjacent to the ghost cell column → Ghost.
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Ghost),
              buffers.tags.faceStatus[1](physicalLocalIndex(layout, buffers.tags.faceStatus[1], 0u, 0u)));

    // EdgeCenteredY edge at physical[2] straddles x=0 → Cut; edge at physical[0] borders
    // the ghost cell → Ghost.
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Cut),
              buffers.tags.edgeStatus[1](physicalLocalIndex(layout, buffers.tags.edgeStatus[1], 2u, 0u)));
    EXPECT_EQ(PHARE::core::toDouble(PHARE::core::ElemStatus::Ghost),
              buffers.tags.edgeStatus[1](physicalLocalIndex(layout, buffers.tags.edgeStatus[1], 0u, 0u)));
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

    EXPECT_FALSE(buffers.tags.ghostCellsData.empty());
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
        = physicalLocalIndex(layout, buffers.tags.cellStatus, 0u, 0u);
    auto const& ghost_cells = buffers.tags.ghostCellsData;
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
    EXPECT_FALSE(buffers.tags.ghostFacesData[0].empty());
    // FaceCenteredY face adjacent to the ghost cell column must also be present.
    EXPECT_FALSE(buffers.tags.ghostFacesData[1].empty());
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

    auto const& ghost_cells = buffers.tags.ghostCellsData;
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
