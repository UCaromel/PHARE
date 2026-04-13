#include "gtest/gtest.h"

#include <cstdint>

#include "core/data/field/field.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutimplyee_mhd.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/inner_boundary/inner_boundary_mesh_classifier.hpp"
#include "core/inner_boundary/plane_inner_boundary.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "core/numerics/inner_boundary_condition/field_dirichlet_inner_boundary_condition.hpp"
#include "core/utilities/box/box.hpp"

namespace
{
constexpr double eps = 1e-12;

using GridLayoutImpl = PHARE::core::GridLayoutImplYeeMHD<2, 2, 1>;
using GridLayout     = PHARE::core::GridLayout<GridLayoutImpl>;
using Classifier     = PHARE::core::InnerBoundaryMeshClassifier<2, GridLayout, PHARE::core::MHDQuantity>;
using MeshData       = PHARE::core::InnerBoundaryMeshData<2, PHARE::core::MHDQuantity>;
using ScalarField    = PHARE::core::Field<2, PHARE::core::MHDQuantity::Scalar, double>;
using FaceField      = PHARE::core::Field<2, PHARE::core::MHDQuantity::Scalar, double>;
using EdgeField      = PHARE::core::Field<2, PHARE::core::MHDQuantity::Scalar, double>;
using FaceVec        = PHARE::core::VecField<FaceField, PHARE::core::MHDQuantity>;
using EdgeVec        = PHARE::core::VecField<EdgeField, PHARE::core::MHDQuantity>;

struct DummyState
{
};

struct MeshDataBuffers
{
    static constexpr char const* BOUNDARY_NAME = "test";

    explicit MeshDataBuffers(GridLayout const& layout)
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
        ScalarField phi_field{std::string(BOUNDARY_NAME) + "_signed_distance",
                              PHARE::core::MHDQuantity::Scalar::NodeCentered,
                              phi_storage.data(), phi_storage.shape()};
        tags.signedDistanceAtNodes.setBuffer(&phi_field);

        ScalarField cell_field{std::string(BOUNDARY_NAME) + "_cell_status",
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

struct DirichletBCFixture
{
    PHARE::core::PlaneInnerBoundary<2> plane{"plane", {0.0, 0.0}, {1.0, 0.0}};
    PHARE::core::Box<int, 2> amr_box{{-2, 0}, {1, 1}};
    GridLayout layout{{1.0, 1.0}, {4u, 2u}, {0.0, 0.0}, amr_box};

    MeshDataBuffers buffers{layout};

    DirichletBCFixture()
    {
        Classifier::Overrides ov;
        ov.cut_eps      = 1e-12;
        ov.inactive_eps = 0.0;
        auto classifier = Classifier::withDefaults(plane, layout, ov);
        classifier(layout, buffers.tags);
    }
};

} // namespace

TEST(FieldDirichletInnerBoundaryCondition, constantFieldMatchingBoundaryValueIsUnchanged)
{
    DirichletBCFixture fix;
    auto const& layout   = fix.layout;
    auto const& meshData = fix.buffers.tags;

    constexpr double C = 7.0;

    PHARE::core::NdArrayVector<2, double> storage{
        layout.allocSize(PHARE::core::MHDQuantity::Scalar::CellCentered)};
    ScalarField field{"rho", PHARE::core::MHDQuantity::Scalar::CellCentered,
                      storage.data(), storage.shape()};

    for (auto i = 0u; i < field.shape()[0]; ++i)
        for (auto j = 0u; j < field.shape()[1]; ++j)
            field(i, j) = C;

    PHARE::core::FieldDirichletInnerBoundaryCondition<ScalarField, GridLayout, DummyState> bc{C};
    DummyState state;
    bc.apply(field, layout, meshData, state, 0.0);

    for (auto i = 0u; i < field.shape()[0]; ++i)
        for (auto j = 0u; j < field.shape()[1]; ++j)
            EXPECT_NEAR(field(i, j), C, eps) << "cell (" << i << ", " << j << ") changed";
}

TEST(FieldDirichletInnerBoundaryCondition, ghostCellReceivesExtrapolatedBoundaryValue)
{
    DirichletBCFixture fix;
    auto const& layout   = fix.layout;
    auto const& meshData = fix.buffers.tags;

    constexpr double boundaryValue = 5.0;

    PHARE::core::NdArrayVector<2, double> storage{
        layout.allocSize(PHARE::core::MHDQuantity::Scalar::CellCentered)};
    ScalarField field{"rho", PHARE::core::MHDQuantity::Scalar::CellCentered,
                      storage.data(), storage.shape()};

    for (auto i = 0u; i < field.shape()[0]; ++i)
        for (auto j = 0u; j < field.shape()[1]; ++j)
        {
            auto amr_pos = layout.localToAMR(PHARE::core::Point<std::uint32_t, 2>{i, j});
            auto amr_ij  = PHARE::core::Point<int, 2>{static_cast<int>(amr_pos[0]),
                                                      static_cast<int>(amr_pos[1])};
            auto pos     = layout.fieldNodeCoordinates(field, amr_ij);
            field(i, j)  = pos[0] + pos[1];
        }

    for (auto const& g : meshData.ghostCellsData)
        field(g.index) = 0.0;

    PHARE::core::FieldDirichletInnerBoundaryCondition<ScalarField, GridLayout, DummyState> bc{
        boundaryValue};
    DummyState state;
    bc.apply(field, layout, meshData, state, 0.0);

    auto const& ghostCells = meshData.ghostCellsData;
    ASSERT_FALSE(ghostCells.empty());

    bool foundInPatch = false;
    for (auto const& g : ghostCells)
    {
        if (!g.mirrorIsInPatch)
        {
            EXPECT_NEAR(field(g.index), 0.0, eps)
                << "out-of-patch ghost at (" << g.index[0] << ", " << g.index[1]
                << ") must not be touched by the BC";
            continue;
        }

        foundInPatch          = true;
        double const expected = 2.0 * boundaryValue - (g.mirrorPoint[0] + g.mirrorPoint[1]);
        EXPECT_NEAR(field(g.index), expected, 1e-10)
            << "ghost at (" << g.index[0] << ", " << g.index[1] << ")";
    }
    EXPECT_TRUE(foundInPatch) << "at least one in-patch ghost must exist";
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
