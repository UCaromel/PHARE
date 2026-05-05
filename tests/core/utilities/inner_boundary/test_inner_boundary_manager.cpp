#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/data/field/field.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutimplyee_mhd.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/inner_boundary/inner_boundary_manager.hpp"
#include "core/inner_boundary/inner_boundary_mesh_classifier.hpp"
#include "core/inner_boundary/plane_inner_boundary.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "core/utilities/box/box.hpp"
#include "initializer/data_provider.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"

namespace
{
constexpr double eps = 1e-10;

using GridLayoutImpl = PHARE::core::GridLayoutImplYeeMHD<2, 2, 1>;
using GridLayout     = PHARE::core::GridLayout<GridLayoutImpl>;
using ScalarField  = PHARE::core::Field<2, PHARE::core::MHDQuantity::Scalar, double>;
using VecFieldMHD2 = PHARE::core::UsableVecFieldMHD<2>;
using MeshData     = PHARE::core::InnerBoundaryMeshData<2, PHARE::core::MHDQuantity>;
using Classifier = PHARE::core::InnerBoundaryMeshClassifier<2, GridLayout, PHARE::core::MHDQuantity>;

/// Minimal physical-state stub.
struct DummyState
{
};

using Manager = PHARE::core::InnerBoundaryManager<PHARE::core::MHDQuantity, ScalarField,
                                                   GridLayout, DummyState>;

// ---------------------------------------------------------------------------
//  ManagerFixture — constructs a Manager directly (no dict needed)
//
//  Plane at x = 0, normal (1, 0).  Grid: 4 × 2 physical cells, dx=dy=1,
//  AMR box {{-2,0},{1,1}}.  Same geometry as all per-BC tests.
//
//  Storage arrays are owned by the fixture and wired into the manager's
//  MeshData via setBuffer(), exactly as MeshDataBuffers does for the
//  per-BC test fixtures.
// ---------------------------------------------------------------------------
struct ManagerFixture
{
    static constexpr char const* BOUNDARY_NAME = "test";

    PHARE::core::Box<int, 2> amr_box{{-2, 0}, {1, 1}};
    GridLayout layout{{1.0, 1.0}, {4u, 2u}, {0.0, 0.0}, amr_box};

    // Storage arrays — must be declared before manager so they outlive it.
    PHARE::core::NdArrayVector<2, double> phi_storage{
        layout.allocSize(PHARE::core::MHDQuantity::Scalar::NodeCentered)};
    std::vector<PHARE::core::NdArrayVector<2, double>> elem_storages;

    std::vector<PHARE::core::MHDQuantity::Scalar> scalarQtys{
        PHARE::core::MHDQuantity::Scalar::rho,
        PHARE::core::MHDQuantity::Scalar::Etot,
    };
    std::vector<PHARE::core::MHDQuantity::Vector> vectorQtys{
        PHARE::core::MHDQuantity::Vector::B,
        PHARE::core::MHDQuantity::Vector::rhoV,
        PHARE::core::MHDQuantity::Vector::E,
    };

    Manager manager{
        std::make_unique<PHARE::core::PlaneInnerBoundary<2>>(
            BOUNDARY_NAME,
            PHARE::core::Point<double, 2>{0.0, 0.0},
            PHARE::core::Point<double, 2>{1.0, 0.0}),
        PHARE::core::InnerBoundaryConditionType::Reflective,
        scalarQtys,
        vectorQtys};

    ManagerFixture()
    {
        // Wire storage into the manager's mesh data via temporary Field objects.
        // setBuffer() only retains the raw data pointer, so the temporaries can be
        // destroyed after the call; the NdArrayVector storage members keep the memory alive.
        auto& md = manager.getMeshData();

        std::string const bn{BOUNDARY_NAME};

        ScalarField phi_field{bn + "_signed_distance",
                              PHARE::core::MHDQuantity::Scalar::NodeCentered,
                              phi_storage.data(), phi_storage.shape()};
        md.signedDistanceAtNodes.setBuffer(&phi_field);

        elem_storages.reserve(MeshData::num_elem_types);
        for (std::size_t i = 0; i < MeshData::num_elem_types; ++i)
        {
            auto const c   = MeshData::idxToCentering(i);
            auto const qty = MeshData::scalarFromCentering(c);
            elem_storages.emplace_back(layout.allocSize(qty));
            ScalarField tmp{md.elemStatus[i].name(), qty,
                            elem_storages[i].data(), elem_storages[i].shape()};
            md.elemStatus[i].setBuffer(&tmp);
        }

        // Classify mesh elements relative to the plane at x = 0.
        PHARE::core::PlaneInnerBoundary<2> plane{BOUNDARY_NAME, {0.0, 0.0}, {1.0, 0.0}};
        Classifier::Overrides ov;
        ov.cut_eps      = 1e-12;
        ov.inactive_eps = 0.0;
        auto classifier = Classifier::withDefaults(plane, layout, ov);
        classifier(layout, md);
    }
};

} // namespace


// ---------------------------------------------------------------------------
// Static factory: returns nullptr when no inner_boundary key in dict
// ---------------------------------------------------------------------------

TEST(InnerBoundaryManager, createReturnsNullptrWithoutInnerBoundaryKey)
{
    PHARE::initializer::PHAREDict dict;
    // dict has no "inner_boundary" key

    auto mgr = Manager::create(dict, {}, {});
    EXPECT_EQ(mgr, nullptr);
}


// ---------------------------------------------------------------------------
// Factory wiring: Reflective assigns the right BC type to each quantity
//
// We verify observable behaviour rather than implementation internals:
//   rho  (scalar) → Neumann   : ghost receives mirror value (constant field unchanged)
//   rhoV (vector) → Symmetric : normal component reversed, tangential preserved
//   E    (vector) → Antisymmetric : tangential component reversed, normal preserved
// ---------------------------------------------------------------------------

/**
 * @brief Scalar rho under Reflective gets Neumann: a constant field stays constant.
 */
TEST(InnerBoundaryManager, reflectiveScalarIsNeumann)
{
    ManagerFixture fix;
    auto const& layout = fix.layout;

    constexpr double C = 5.0;

    PHARE::core::NdArrayVector<2, double> storage{
        layout.allocSize(PHARE::core::MHDQuantity::Scalar::CellCentered)};
    ScalarField rho{"rho", PHARE::core::MHDQuantity::Scalar::CellCentered,
                    storage.data(), storage.shape()};

    for (auto i = 0u; i < rho.shape()[0]; ++i)
        for (auto j = 0u; j < rho.shape()[1]; ++j)
            rho(i, j) = C;

    DummyState state;
    PHARE::core::InnerBCContext<DummyState> ctx{state, state, 0.0};
    fix.manager.applyBC(PHARE::core::MHDQuantity::Scalar::rho, rho, layout, ctx);

    for (auto i = 0u; i < rho.shape()[0]; ++i)
        for (auto j = 0u; j < rho.shape()[1]; ++j)
            EXPECT_NEAR(rho(i, j), C, eps) << "cell (" << i << ", " << j << ") changed";
}

/**
 * @brief Vector rhoV under Reflective gets Symmetric: normal component is reversed.
 *
 * Constant field (Cx, Cy, Cz), plane normal n=(1,0,0).
 * Expected ghost: (-Cx, Cy, Cz).
 */
TEST(InnerBoundaryManager, reflectiveRhoVIsSymmetric_normalComponentFlips)
{
    ManagerFixture fix;
    auto const& layout   = fix.layout;
    auto const& meshData = fix.manager.getMeshData();

    constexpr double Cx = 3.0, Cy = 4.0, Cz = 5.0;

    VecFieldMHD2 rhoV{"rhoV", layout, PHARE::core::MHDQuantity::Vector::rhoV};
    auto& rhoVx = rhoV.getComponent(PHARE::core::Component::X);
    auto& rhoVy = rhoV.getComponent(PHARE::core::Component::Y);
    auto& rhoVz = rhoV.getComponent(PHARE::core::Component::Z);

    for (auto i = 0u; i < rhoVx.shape()[0]; ++i)
        for (auto j = 0u; j < rhoVx.shape()[1]; ++j)
        {
            rhoVx(i, j) = Cx;
            rhoVy(i, j) = Cy;
            rhoVz(i, j) = Cz;
        }

    DummyState state;
    PHARE::core::InnerBCContext<DummyState> ctx{state, state, 0.0};
    fix.manager.applyBC(PHARE::core::MHDQuantity::Vector::rhoV, rhoV, layout, ctx);

    constexpr std::array<PHARE::core::QtyCentering, 2> kCellC
        = {PHARE::core::QtyCentering::dual, PHARE::core::QtyCentering::dual};

    bool foundInPatch = false;
    for (auto const& g : meshData.getGhostDataFromCentering(kCellC))
    {
        if (!g.mirrorIsInPatch)
            continue;
        foundInPatch = true;

        EXPECT_NEAR(rhoVx(g.index), -Cx, eps)
            << "rhoVx ghost at (" << g.index[0] << "," << g.index[1] << ")";
        EXPECT_NEAR(rhoVy(g.index), Cy, eps)
            << "rhoVy ghost at (" << g.index[0] << "," << g.index[1] << ")";
        EXPECT_NEAR(rhoVz(g.index), Cz, eps)
            << "rhoVz ghost at (" << g.index[0] << "," << g.index[1] << ")";
    }
    EXPECT_TRUE(foundInPatch) << "at least one in-patch ghost must exist";
}

/**
 * @brief Vector E under Reflective gets Antisymmetric: tangential component is reversed.
 *
 * Constant field (Ex, Ey, Ez), plane normal n=(1,0,0).
 * Antisymmetric: keep normal (Ex), reverse tangential → ghost = (Ex, -Ey, -Ez).
 *
 * But E components are edge-centred, so the ghost elements come from ghostEdgesData,
 * not ghostCellsData.  We verify the x-component (EdgeCenteredX centering = (dual,primal,primal)
 * → ghostEdgesData[0]) separately from y/z (EdgeCenteredY/Z).
 */
TEST(InnerBoundaryManager, reflectiveEIsAntisymmetric_tangentialComponentFlips)
{
    ManagerFixture fix;
    auto const& layout   = fix.layout;
    auto const& meshData = fix.manager.getMeshData();

    constexpr double Ex = 2.0, Ey = 3.0, Ez = -1.0;

    VecFieldMHD2 E{"E", layout, PHARE::core::MHDQuantity::Vector::E};
    auto& Ex_field = E.getComponent(PHARE::core::Component::X);
    auto& Ey_field = E.getComponent(PHARE::core::Component::Y);
    auto& Ez_field = E.getComponent(PHARE::core::Component::Z);

    for (auto i = 0u; i < Ex_field.shape()[0]; ++i)
        for (auto j = 0u; j < Ex_field.shape()[1]; ++j)
            Ex_field(i, j) = Ex;

    for (auto i = 0u; i < Ey_field.shape()[0]; ++i)
        for (auto j = 0u; j < Ey_field.shape()[1]; ++j)
            Ey_field(i, j) = Ey;

    // Ez is the out-of-plane component; fill it but its ghost list has no separate entry in 2D.
    for (auto i = 0u; i < Ez_field.shape()[0]; ++i)
        for (auto j = 0u; j < Ez_field.shape()[1]; ++j)
            Ez_field(i, j) = Ez;

    DummyState state;
    PHARE::core::InnerBCContext<DummyState> ctx{state, state, 0.0};
    fix.manager.applyBC(PHARE::core::MHDQuantity::Vector::E, E, layout, ctx);

    // In 2D, EdgeCenteredX has centering (dual,primal) and EdgeCenteredY has (primal,dual).
    // EdgeCenteredZ collapses to all-primal = node centering, shared with Ez.
    // Normal n = (1,0,0):
    //   Ex (dual,primal): antisymmetric keeps normal component → Ex preserved.
    //   Ey (primal,dual): antisymmetric reverses tangential → Ey negated.
    constexpr std::array<PHARE::core::QtyCentering, 2> kEdgeXC // EdgeCenteredX
        = {PHARE::core::QtyCentering::dual,   PHARE::core::QtyCentering::primal};
    constexpr std::array<PHARE::core::QtyCentering, 2> kEdgeYC // EdgeCenteredY
        = {PHARE::core::QtyCentering::primal, PHARE::core::QtyCentering::dual};

    bool foundAnyEdge = false;
    for (auto const& g : meshData.getGhostDataFromCentering(kEdgeXC))
    {
        if (!g.mirrorIsInPatch)
            continue;
        foundAnyEdge = true;
        EXPECT_NEAR(Ex_field(g.index), Ex, eps)
            << "Ex (normal component) should be preserved";
    }
    for (auto const& g : meshData.getGhostDataFromCentering(kEdgeYC))
    {
        if (!g.mirrorIsInPatch)
            continue;
        foundAnyEdge = true;
        EXPECT_NEAR(Ey_field(g.index), -Ey, eps)
            << "Ey (tangential component) should be negated";
    }

    EXPECT_TRUE(foundAnyEdge) << "no in-patch edge ghost found — check classifier / geometry";
}

/**
 * @brief applyBC is a no-op for a quantity that was not registered.
 */
TEST(InnerBoundaryManager, applyBCIsNoopForUnregisteredScalar)
{
    ManagerFixture fix;
    auto const& layout = fix.layout;

    PHARE::core::NdArrayVector<2, double> storage{
        layout.allocSize(PHARE::core::MHDQuantity::Scalar::CellCentered)};
    ScalarField field{"P", PHARE::core::MHDQuantity::Scalar::CellCentered,
                      storage.data(), storage.shape()};

    constexpr double C = 42.0;
    for (auto i = 0u; i < field.shape()[0]; ++i)
        for (auto j = 0u; j < field.shape()[1]; ++j)
            field(i, j) = C;

    DummyState state;
    PHARE::core::InnerBCContext<DummyState> ctx{state, state, 0.0};
    // P was not registered — should be a no-op, not a crash
    fix.manager.applyBC(PHARE::core::MHDQuantity::Scalar::P, field, layout, ctx);

    for (auto i = 0u; i < field.shape()[0]; ++i)
        for (auto j = 0u; j < field.shape()[1]; ++j)
            EXPECT_NEAR(field(i, j), C, eps) << "unregistered scalar field was modified";
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
