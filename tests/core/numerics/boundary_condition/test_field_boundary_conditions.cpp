#include "gtest/gtest.h"

#include "core/boundary/boundary_defs.hpp"
#include "core/data/grid/grid.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutimplyee.hpp"
#include "core/data/grid/gridlayoutimplyee_mhd.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/data/patch_field_accessor.hpp"
#include "core/numerics/boundary_condition/field_antisymmetric_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_divergence_free_transverse_dirichlet_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_divergence_free_transverse_neumann_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_dirichlet_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_neumann_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_none_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_symmetric_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_total_energy_from_pressure_boundary_condition.hpp"
#include "core/numerics/thermo/ideal_gas_thermo.hpp"
#include "core/utilities/box/box.hpp"
#include "tests/core/data/tensorfield/test_tensorfield_fixtures.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"

using namespace PHARE::core;

// 1D, interpolation order 1
static constexpr std::uint32_t nCells    = 10;
static constexpr std::uint32_t ghostWidth = GridLayoutImplYee<1, 1>::ghost_width;
static constexpr double interiorValue    = 5.0;
static constexpr double ghostSentinel    = 99.0;

using GridLayout1D = GridLayout<GridLayoutImplYee<1, 1>>;
using NdArray1D    = NdArrayVector<1, double>;
using Grid1D       = Grid<NdArray1D, HybridQuantity::Scalar>;
using Field1D      = Grid1D::field_type;

/**
 * @brief No-op accessor: throws if called, verifying BCs don't use cross-field access.
 */
template<typename FieldT>
struct NullFieldAccessorT : IPatchFieldAccessor<FieldT, HybridQuantity>
{
    FieldT& getField(HybridQuantity::Scalar) const override
    {
        throw std::runtime_error("NullFieldAccessorT: getField() should not be called");
    }
    VecField<FieldT, HybridQuantity> getVecField(HybridQuantity::Vector) const override
    {
        throw std::runtime_error("NullFieldAccessorT: getVecField() should not be called");
    }
};
using NullFieldAccessor = NullFieldAccessorT<Field1D>;

// Local cell-boxes for the ghost regions (passed to bc.apply())
Box<std::uint32_t, 1> lowerGhostCellBox()
{
    return {Point<std::uint32_t, 1>{0u}, Point<std::uint32_t, 1>{ghostWidth - 1}};
}
Box<std::uint32_t, 1> upperGhostCellBox()
{
    return {Point<std::uint32_t, 1>{ghostWidth + nCells},
            Point<std::uint32_t, 1>{2 * ghostWidth + nCells - 1}};
}


/**
 * @brief 1D scalar field BC fixture.
 *
 * Interior cells are filled with @p interiorValue, ghost cells with @p ghostSentinel.
 * Uses HybridQuantity::Scalar::rho (cell-centred / dual in 1D Yee).
 */
struct FieldBC1D : testing::Test
{
    GridLayout1D layout{{0.1}, {nCells}, {0.0}};
    NullFieldAccessor acc;

    static constexpr auto qty = HybridQuantity::Scalar::rho;
    Grid1D grid{"rho", qty, layout.allocSize(qty)};
    Field1D& field{*(&grid)};

    std::uint32_t physStart{layout.physicalStartIndex(qty, Direction::X)};
    std::uint32_t physEnd{layout.physicalEndIndex(qty, Direction::X)};

    FieldBC1D()
    {
        for (std::uint32_t i = 0; i < grid.shape()[0]; ++i)
            field(i) = ghostSentinel;
        for (std::uint32_t i = physStart; i <= physEnd; ++i)
            field(i) = interiorValue;
    }
};


// ─── None ─────────────────────────────────────────────────────────────────────

TEST_F(FieldBC1D, NoneDoesNotModifyGhostCells)
{
    FieldNoneBoundaryCondition<Field1D, GridLayout1D> bc;
    bc.apply(field, BoundaryLocation::XLower, lowerGhostCellBox(), layout, 0.0, acc);
    bc.apply(field, BoundaryLocation::XUpper, upperGhostCellBox(), layout, 0.0, acc);

    for (std::uint32_t g = 0; g < ghostWidth; ++g)
    {
        EXPECT_DOUBLE_EQ(field(g), ghostSentinel);
        EXPECT_DOUBLE_EQ(field(grid.shape()[0] - 1 - g), ghostSentinel);
    }
}


// ─── Neumann ──────────────────────────────────────────────────────────────────

TEST_F(FieldBC1D, NeumannSetsLowerGhostToInteriorValue)
{
    FieldNeumannBoundaryCondition<Field1D, GridLayout1D> bc;
    bc.apply(field, BoundaryLocation::XLower, lowerGhostCellBox(), layout, 0.0, acc);

    // Mirror of any lower ghost points into a constant interior → ghost = interiorValue
    for (std::uint32_t g = 0; g < ghostWidth; ++g)
        EXPECT_DOUBLE_EQ(field(g), interiorValue);
}

TEST_F(FieldBC1D, NeumannSetsUpperGhostToInteriorValue)
{
    FieldNeumannBoundaryCondition<Field1D, GridLayout1D> bc;
    bc.apply(field, BoundaryLocation::XUpper, upperGhostCellBox(), layout, 0.0, acc);

    std::uint32_t allocSz = grid.shape()[0];
    for (std::uint32_t g = 0; g < ghostWidth; ++g)
        EXPECT_DOUBLE_EQ(field(allocSz - 1 - g), interiorValue);
}


// ─── Dirichlet ────────────────────────────────────────────────────────────────

TEST_F(FieldBC1D, DirichletSetsLowerGhostByLinearExtrapolation)
{
    double const value = 3.0;
    FieldDirichletBoundaryCondition<Field1D, GridLayout1D> bc{value};
    bc.apply(field, BoundaryLocation::XLower, lowerGhostCellBox(), layout, 0.0, acc);

    // Interior is constant = interiorValue, so ghost = 2*value - interiorValue
    double expected = 2.0 * value - interiorValue;
    for (std::uint32_t g = 0; g < ghostWidth; ++g)
        EXPECT_DOUBLE_EQ(field(g), expected);
}

TEST_F(FieldBC1D, DirichletSetsUpperGhostByLinearExtrapolation)
{
    double const value = 3.0;
    FieldDirichletBoundaryCondition<Field1D, GridLayout1D> bc{value};
    bc.apply(field, BoundaryLocation::XUpper, upperGhostCellBox(), layout, 0.0, acc);

    double expected   = 2.0 * value - interiorValue;
    std::uint32_t allocSz = grid.shape()[0];
    for (std::uint32_t g = 0; g < ghostWidth; ++g)
        EXPECT_DOUBLE_EQ(field(allocSz - 1 - g), expected);
}


// ─── Symmetric scalar (= Neumann) ─────────────────────────────────────────────

TEST_F(FieldBC1D, SymmetricScalarEquivalentToNeumann)
{
    // Reference: apply Neumann on a copy
    Grid1D refGrid{"rho_ref", qty, layout.allocSize(qty)};
    Field1D& refField{*(&refGrid)};
    for (std::uint32_t i = 0; i < refGrid.shape()[0]; ++i)
        refField(i) = (i >= physStart && i <= physEnd) ? interiorValue : ghostSentinel;
    FieldNeumannBoundaryCondition<Field1D, GridLayout1D> neumann;
    neumann.apply(refField, BoundaryLocation::XLower, lowerGhostCellBox(), layout, 0.0, acc);
    neumann.apply(refField, BoundaryLocation::XUpper, upperGhostCellBox(), layout, 0.0, acc);

    // Apply Symmetric
    FieldSymmetricBoundaryCondition<Field1D, GridLayout1D> sym;
    sym.apply(field, BoundaryLocation::XLower, lowerGhostCellBox(), layout, 0.0, acc);
    sym.apply(field, BoundaryLocation::XUpper, upperGhostCellBox(), layout, 0.0, acc);

    for (std::uint32_t i = 0; i < grid.shape()[0]; ++i)
        EXPECT_DOUBLE_EQ(field(i), refField(i)) << "at index " << i;
}


// ─── AntiSymmetric scalar (= Dirichlet 0) ─────────────────────────────────────

TEST_F(FieldBC1D, AntiSymmetricScalarEquivalentToDirichletZero)
{
    Grid1D refGrid{"rho_ref", qty, layout.allocSize(qty)};
    Field1D& refField{*(&refGrid)};
    for (std::uint32_t i = 0; i < refGrid.shape()[0]; ++i)
        refField(i) = (i >= physStart && i <= physEnd) ? interiorValue : ghostSentinel;
    FieldDirichletBoundaryCondition<Field1D, GridLayout1D> dirichlet{0.0};
    dirichlet.apply(refField, BoundaryLocation::XLower, lowerGhostCellBox(), layout, 0.0, acc);
    dirichlet.apply(refField, BoundaryLocation::XUpper, upperGhostCellBox(), layout, 0.0, acc);

    FieldAntiSymmetricBoundaryCondition<Field1D, GridLayout1D> antisym;
    antisym.apply(field, BoundaryLocation::XLower, lowerGhostCellBox(), layout, 0.0, acc);
    antisym.apply(field, BoundaryLocation::XUpper, upperGhostCellBox(), layout, 0.0, acc);

    for (std::uint32_t i = 0; i < grid.shape()[0]; ++i)
        EXPECT_DOUBLE_EQ(field(i), refField(i)) << "at index " << i;
}


// ─── VecField (B) fixture ─────────────────────────────────────────────────────
//
// 1D, order 1, 10 cells. HybridQuantity::Vector::B.
// ghostWidth = GridLayoutImplYee<1,1>::ghost_width = 2
//   Bx: primal → allocSize=15, physStart=2, physEnd=12
//               lower ghosts: [0,1],  upper ghosts: [13,14]
//   By: dual   → allocSize=14, physStart=2, physEnd=11
//               lower ghosts: [0,1],  upper ghosts: [12,13]
//   Bz: dual   → same as By
//
// Interior is filled with interiorValue, ghosts with ghostSentinel.

using VecField1D = VecField<Field1D, HybridQuantity>;

struct VecFieldBC1D : testing::Test
{
    GridLayout1D layout{{0.1}, {nCells}, {0.0}};
    NullFieldAccessor acc;

    static constexpr auto vecQty = HybridQuantity::Vector::B;
    UsableTensorField<1, 1> B{"B", layout, vecQty};

    VecFieldBC1D()
    {
        for (std::size_t comp = 0; comp < 3; ++comp)
        {
            auto& f = B[comp];
            for (std::uint32_t i = 0; i < f.shape()[0]; ++i)
                f(i) = ghostSentinel;
            auto qty        = HybridQuantity::componentsQuantities(vecQty)[comp];
            std::uint32_t s = layout.physicalStartIndex(qty, Direction::X);
            std::uint32_t e = layout.physicalEndIndex(qty, Direction::X);
            for (std::uint32_t i = s; i <= e; ++i)
                f(i) = interiorValue;
        }
    }
};


// ─── Symmetric VecField ───────────────────────────────────────────────────────

TEST_F(VecFieldBC1D, SymmetricNormalComponentBxSetToDirichletZero)
{
    FieldSymmetricBoundaryCondition<VecField1D, GridLayout1D> bc;
    bc.apply(B, BoundaryLocation::XLower, lowerGhostCellBox(), layout, 0.0, acc);
    bc.apply(B, BoundaryLocation::XUpper, upperGhostCellBox(), layout, 0.0, acc);

    // Bx is primal: physStart=2 (lower boundary node), physEnd=12 (upper boundary node).
    // Boundary nodes are set to the Dirichlet value = 0.
    // Ghost nodes closest to the boundary: ghost = 2*0 - interior = -interiorValue.
    auto& Bx = B[0];
    auto bxQty = HybridQuantity::Scalar::Bx;
    std::uint32_t bxPhysStart = layout.physicalStartIndex(bxQty, Direction::X);
    std::uint32_t bxPhysEnd   = layout.physicalEndIndex(bxQty, Direction::X);
    EXPECT_DOUBLE_EQ(Bx(bxPhysStart - 1), -interiorValue); // lower ghost nearest boundary
    EXPECT_DOUBLE_EQ(Bx(bxPhysStart),     0.0);            // lower boundary node = Dirichlet value
    EXPECT_DOUBLE_EQ(Bx(bxPhysEnd),       0.0);            // upper boundary node = Dirichlet value
    EXPECT_DOUBLE_EQ(Bx(bxPhysEnd + 1),   -interiorValue); // upper ghost nearest boundary
}

TEST_F(VecFieldBC1D, SymmetricTangentialComponentsByBzSetToNeumann)
{
    FieldSymmetricBoundaryCondition<VecField1D, GridLayout1D> bc;
    bc.apply(B, BoundaryLocation::XLower, lowerGhostCellBox(), layout, 0.0, acc);
    bc.apply(B, BoundaryLocation::XUpper, upperGhostCellBox(), layout, 0.0, acc);

    // By and Bz are dual: physEnd=11, upper ghosts start at physEnd+1=12.
    // Neumann mirrors a flat interior → ghost = interiorValue.
    auto byQty = HybridQuantity::Scalar::By;
    std::uint32_t byPhysEnd = layout.physicalEndIndex(byQty, Direction::X);
    for (std::size_t comp : {1u, 2u})
    {
        auto& f = B[comp];
        EXPECT_DOUBLE_EQ(f(0),            interiorValue) << "component " << comp << " lower ghost";
        EXPECT_DOUBLE_EQ(f(byPhysEnd + 1), interiorValue) << "component " << comp << " upper ghost";
    }
}


// ─── AntiSymmetric VecField ───────────────────────────────────────────────────

TEST_F(VecFieldBC1D, AntiSymmetricNormalComponentBxSetToNeumann)
{
    FieldAntiSymmetricBoundaryCondition<VecField1D, GridLayout1D> bc;
    bc.apply(B, BoundaryLocation::XLower, lowerGhostCellBox(), layout, 0.0, acc);
    bc.apply(B, BoundaryLocation::XUpper, upperGhostCellBox(), layout, 0.0, acc);

    // Bx is primal: Neumann mirrors a flat interior → all ghost nodes = interiorValue.
    auto& Bx = B[0];
    auto bxQty = HybridQuantity::Scalar::Bx;
    std::uint32_t bxPhysStart = layout.physicalStartIndex(bxQty, Direction::X);
    std::uint32_t bxPhysEnd   = layout.physicalEndIndex(bxQty, Direction::X);
    EXPECT_DOUBLE_EQ(Bx(bxPhysStart - 1), interiorValue); // lower ghost nearest boundary
    EXPECT_DOUBLE_EQ(Bx(bxPhysEnd + 1),   interiorValue); // upper ghost nearest boundary
}

TEST_F(VecFieldBC1D, AntiSymmetricTangentialComponentsByBzSetToDirichletZero)
{
    FieldAntiSymmetricBoundaryCondition<VecField1D, GridLayout1D> bc;
    bc.apply(B, BoundaryLocation::XLower, lowerGhostCellBox(), layout, 0.0, acc);
    bc.apply(B, BoundaryLocation::XUpper, upperGhostCellBox(), layout, 0.0, acc);

    // By and Bz are dual: physEnd=11, upper ghosts start at physEnd+1=12.
    // Dirichlet(0) → ghost = 2*0 - interior = -interiorValue.
    auto byQty = HybridQuantity::Scalar::By;
    std::uint32_t byPhysEnd = layout.physicalEndIndex(byQty, Direction::X);
    for (std::size_t comp : {1u, 2u})
    {
        auto& f = B[comp];
        EXPECT_DOUBLE_EQ(f(0),            -interiorValue) << "component " << comp << " lower ghost";
        EXPECT_DOUBLE_EQ(f(byPhysEnd + 1), -interiorValue) << "component " << comp << " upper ghost";
    }
}


// ─── DivergenceFreeTransverseDirichlet VecField ───────────────────────────────

TEST_F(VecFieldBC1D, DivergenceFreeTransverseDirichletAtXBoundaries)
{
    std::array values{123.0, 7.0, 11.0};
    FieldDivergenceFreeTransverseDirichletBoundaryCondition<VecField1D, GridLayout1D> bc{values};
    bc.apply(B, BoundaryLocation::XLower, lowerGhostCellBox(), layout, 0.0, acc);
    bc.apply(B, BoundaryLocation::XUpper, upperGhostCellBox(), layout, 0.0, acc);

    auto& Bx = B[0];
    auto bxQty = HybridQuantity::Scalar::Bx;
    std::uint32_t bxPhysStart = layout.physicalStartIndex(bxQty, Direction::X);
    std::uint32_t bxPhysEnd   = layout.physicalEndIndex(bxQty, Direction::X);
    EXPECT_DOUBLE_EQ(Bx(bxPhysStart - 2), interiorValue);
    EXPECT_DOUBLE_EQ(Bx(bxPhysStart - 1), interiorValue);
    EXPECT_DOUBLE_EQ(Bx(bxPhysEnd + 1), interiorValue);
    EXPECT_DOUBLE_EQ(Bx(bxPhysEnd + 2), interiorValue);

    for (std::size_t comp : {1u, 2u})
    {
        auto& f              = B[comp];
        auto qty             = HybridQuantity::componentsQuantities(vecQty)[comp];
        std::uint32_t psx    = layout.physicalStartIndex(qty, Direction::X);
        std::uint32_t pex    = layout.physicalEndIndex(qty, Direction::X);
        double expectedGhost = 2.0 * values[comp] - interiorValue;
        EXPECT_DOUBLE_EQ(f(psx - 2), expectedGhost) << "component " << comp << " lower far ghost";
        EXPECT_DOUBLE_EQ(f(psx - 1), expectedGhost) << "component " << comp << " lower near ghost";
        EXPECT_DOUBLE_EQ(f(pex + 1), expectedGhost) << "component " << comp << " upper near ghost";
        EXPECT_DOUBLE_EQ(f(pex + 2), expectedGhost) << "component " << comp << " upper far ghost";
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// 2D tests
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr std::uint32_t nCellsX2D = 10u;
static constexpr std::uint32_t nCellsY2D = 8u;

using GridLayout2D = GridLayout<GridLayoutImplYee<2, 1>>;
using NdArray2D    = NdArrayVector<2, double>;
using Grid2D       = Grid<NdArray2D, HybridQuantity::Scalar>;
using Field2D      = Grid2D::field_type;
using VecField2D   = VecField<Field2D, HybridQuantity>;

// Ghost cell boxes: normal-direction strip × full transverse extent.
Box<std::uint32_t, 2> xLowerGhostCellBox2D()
{
    return {{0u, 0u}, {ghostWidth - 1u, nCellsY2D + 2u * ghostWidth - 1u}};
}
Box<std::uint32_t, 2> xUpperGhostCellBox2D()
{
    return {{ghostWidth + nCellsX2D, 0u},
            {2u * ghostWidth + nCellsX2D - 1u, nCellsY2D + 2u * ghostWidth - 1u}};
}
Box<std::uint32_t, 2> yLowerGhostCellBox2D()
{
    return {{0u, 0u}, {nCellsX2D + 2u * ghostWidth - 1u, ghostWidth - 1u}};
}
Box<std::uint32_t, 2> yUpperGhostCellBox2D()
{
    return {{0u, ghostWidth + nCellsY2D},
            {nCellsX2D + 2u * ghostWidth - 1u, 2u * ghostWidth + nCellsY2D - 1u}};
}

/**
 * @brief 2D VecField (B) fixture. ghostWidth=2 (order 1).
 *
 * B field centerings in Yee-2D:
 *   Bx: primal X, dual   Y
 *   By: dual   X, primal Y
 *   Bz: dual   X, dual   Y
 *
 * Interior cells are filled with @p interiorValue, ghost cells with @p ghostSentinel.
 */
struct VecFieldBC2D : testing::Test
{
    GridLayout2D layout{{0.1, 0.1}, {nCellsX2D, nCellsY2D}, {0.0, 0.0}};
    NullFieldAccessorT<Field2D> acc;

    static constexpr auto vecQty = HybridQuantity::Vector::B;
    UsableTensorField<2, 1> B{"B", layout, vecQty};

    VecFieldBC2D()
    {
        for (std::size_t comp = 0; comp < 3; ++comp)
        {
            auto& f    = B[comp];
            auto shape = f.shape();
            for (std::uint32_t ix = 0; ix < shape[0]; ++ix)
                for (std::uint32_t iy = 0; iy < shape[1]; ++iy)
                    f(ix, iy) = ghostSentinel;
            auto qty         = HybridQuantity::componentsQuantities(vecQty)[comp];
            std::uint32_t sx = layout.physicalStartIndex(qty, Direction::X);
            std::uint32_t ex = layout.physicalEndIndex(qty, Direction::X);
            std::uint32_t sy = layout.physicalStartIndex(qty, Direction::Y);
            std::uint32_t ey = layout.physicalEndIndex(qty, Direction::Y);
            for (std::uint32_t ix = sx; ix <= ex; ++ix)
                for (std::uint32_t iy = sy; iy <= ey; ++iy)
                    f(ix, iy) = interiorValue;
        }
    }
};

struct VecFieldBC2DNonUniformBy : testing::Test
{
    GridLayout2D layout{{0.1, 0.1}, {nCellsX2D, nCellsY2D}, {0.0, 0.0}};
    NullFieldAccessorT<Field2D> acc;

    static constexpr auto vecQty = HybridQuantity::Vector::B;
    UsableTensorField<2, 1> B{"B", layout, vecQty};

    VecFieldBC2DNonUniformBy()
    {
        for (std::size_t comp = 0; comp < 3; ++comp)
        {
            auto& f    = B[comp];
            auto shape = f.shape();
            for (std::uint32_t ix = 0; ix < shape[0]; ++ix)
                for (std::uint32_t iy = 0; iy < shape[1]; ++iy)
                    f(ix, iy) = ghostSentinel;

            auto qty         = HybridQuantity::componentsQuantities(vecQty)[comp];
            std::uint32_t sx = layout.physicalStartIndex(qty, Direction::X);
            std::uint32_t ex = layout.physicalEndIndex(qty, Direction::X);
            std::uint32_t sy = layout.physicalStartIndex(qty, Direction::Y);
            std::uint32_t ey = layout.physicalEndIndex(qty, Direction::Y);

            for (std::uint32_t ix = sx; ix <= ex; ++ix)
                for (std::uint32_t iy = sy; iy <= ey; ++iy)
                    f(ix, iy) = comp == 1 ? static_cast<double>(iy) : interiorValue;
        }
    }
};


// ─── Symmetric VecField 2D ───────────────────────────────────────────────────

TEST_F(VecFieldBC2D, SymmetricAtXBoundaries)
{
    FieldSymmetricBoundaryCondition<VecField2D, GridLayout2D> bc;
    bc.apply(B, BoundaryLocation::XLower, xLowerGhostCellBox2D(), layout, 0.0, acc);
    bc.apply(B, BoundaryLocation::XUpper, xUpperGhostCellBox2D(), layout, 0.0, acc);

    // Bx: primal in X (normal to X boundary) → Dirichlet(0)
    {
        auto& Bx          = B[0];
        auto bxQty        = HybridQuantity::Scalar::Bx;
        std::uint32_t psx = layout.physicalStartIndex(bxQty, Direction::X);
        std::uint32_t pex = layout.physicalEndIndex(bxQty, Direction::X);
        std::uint32_t sy  = layout.physicalStartIndex(bxQty, Direction::Y);
        std::uint32_t ey  = layout.physicalEndIndex(bxQty, Direction::Y);
        for (std::uint32_t iy = sy; iy <= ey; ++iy)
        {
            EXPECT_DOUBLE_EQ(Bx(psx - 1, iy), -interiorValue) << "Bx lower ghost iy=" << iy;
            EXPECT_DOUBLE_EQ(Bx(psx,     iy), 0.0)            << "Bx lower boundary iy=" << iy;
            EXPECT_DOUBLE_EQ(Bx(pex,     iy), 0.0)            << "Bx upper boundary iy=" << iy;
            EXPECT_DOUBLE_EQ(Bx(pex + 1, iy), -interiorValue) << "Bx upper ghost iy=" << iy;
        }
    }

    // By, Bz: dual in X (tangential to X boundary) → Neumann
    for (std::size_t comp : {1u, 2u})
    {
        auto& f           = B[comp];
        auto qty          = HybridQuantity::componentsQuantities(vecQty)[comp];
        std::uint32_t psx = layout.physicalStartIndex(qty, Direction::X);
        std::uint32_t pex = layout.physicalEndIndex(qty, Direction::X);
        std::uint32_t sy  = layout.physicalStartIndex(qty, Direction::Y);
        std::uint32_t ey  = layout.physicalEndIndex(qty, Direction::Y);
        for (std::uint32_t iy = sy; iy <= ey; ++iy)
        {
            EXPECT_DOUBLE_EQ(f(psx - 1, iy), interiorValue)
                << "comp=" << comp << " lower ghost iy=" << iy;
            EXPECT_DOUBLE_EQ(f(pex + 1, iy), interiorValue)
                << "comp=" << comp << " upper ghost iy=" << iy;
        }
    }
}

TEST_F(VecFieldBC2D, SymmetricAtYBoundaries)
{
    FieldSymmetricBoundaryCondition<VecField2D, GridLayout2D> bc;
    bc.apply(B, BoundaryLocation::YLower, yLowerGhostCellBox2D(), layout, 0.0, acc);
    bc.apply(B, BoundaryLocation::YUpper, yUpperGhostCellBox2D(), layout, 0.0, acc);

    // By: primal in Y (normal to Y boundary) → Dirichlet(0)
    {
        auto& By          = B[1];
        auto byQty        = HybridQuantity::Scalar::By;
        std::uint32_t psy = layout.physicalStartIndex(byQty, Direction::Y);
        std::uint32_t pey = layout.physicalEndIndex(byQty, Direction::Y);
        std::uint32_t sx  = layout.physicalStartIndex(byQty, Direction::X);
        std::uint32_t ex  = layout.physicalEndIndex(byQty, Direction::X);
        for (std::uint32_t ix = sx; ix <= ex; ++ix)
        {
            EXPECT_DOUBLE_EQ(By(ix, psy - 1), -interiorValue) << "By lower ghost ix=" << ix;
            EXPECT_DOUBLE_EQ(By(ix, psy),     0.0)            << "By lower boundary ix=" << ix;
            EXPECT_DOUBLE_EQ(By(ix, pey),     0.0)            << "By upper boundary ix=" << ix;
            EXPECT_DOUBLE_EQ(By(ix, pey + 1), -interiorValue) << "By upper ghost ix=" << ix;
        }
    }

    // Bx, Bz: dual in Y (tangential to Y boundary) → Neumann
    for (std::size_t comp : {0u, 2u})
    {
        auto& f           = B[comp];
        auto qty          = HybridQuantity::componentsQuantities(vecQty)[comp];
        std::uint32_t psy = layout.physicalStartIndex(qty, Direction::Y);
        std::uint32_t pey = layout.physicalEndIndex(qty, Direction::Y);
        std::uint32_t sx  = layout.physicalStartIndex(qty, Direction::X);
        std::uint32_t ex  = layout.physicalEndIndex(qty, Direction::X);
        for (std::uint32_t ix = sx; ix <= ex; ++ix)
        {
            EXPECT_DOUBLE_EQ(f(ix, psy - 1), interiorValue)
                << "comp=" << comp << " lower ghost ix=" << ix;
            EXPECT_DOUBLE_EQ(f(ix, pey + 1), interiorValue)
                << "comp=" << comp << " upper ghost ix=" << ix;
        }
    }
}


// ─── AntiSymmetric VecField 2D ───────────────────────────────────────────────

TEST_F(VecFieldBC2D, AntiSymmetricAtXBoundaries)
{
    FieldAntiSymmetricBoundaryCondition<VecField2D, GridLayout2D> bc;
    bc.apply(B, BoundaryLocation::XLower, xLowerGhostCellBox2D(), layout, 0.0, acc);
    bc.apply(B, BoundaryLocation::XUpper, xUpperGhostCellBox2D(), layout, 0.0, acc);

    // Bx: primal in X (normal to X boundary) → Neumann
    {
        auto& Bx          = B[0];
        auto bxQty        = HybridQuantity::Scalar::Bx;
        std::uint32_t psx = layout.physicalStartIndex(bxQty, Direction::X);
        std::uint32_t pex = layout.physicalEndIndex(bxQty, Direction::X);
        std::uint32_t sy  = layout.physicalStartIndex(bxQty, Direction::Y);
        std::uint32_t ey  = layout.physicalEndIndex(bxQty, Direction::Y);
        for (std::uint32_t iy = sy; iy <= ey; ++iy)
        {
            EXPECT_DOUBLE_EQ(Bx(psx - 1, iy), interiorValue) << "Bx lower ghost iy=" << iy;
            EXPECT_DOUBLE_EQ(Bx(pex + 1, iy), interiorValue) << "Bx upper ghost iy=" << iy;
        }
    }

    // By, Bz: dual in X (tangential to X boundary) → Dirichlet(0)
    for (std::size_t comp : {1u, 2u})
    {
        auto& f           = B[comp];
        auto qty          = HybridQuantity::componentsQuantities(vecQty)[comp];
        std::uint32_t psx = layout.physicalStartIndex(qty, Direction::X);
        std::uint32_t pex = layout.physicalEndIndex(qty, Direction::X);
        std::uint32_t sy  = layout.physicalStartIndex(qty, Direction::Y);
        std::uint32_t ey  = layout.physicalEndIndex(qty, Direction::Y);
        for (std::uint32_t iy = sy; iy <= ey; ++iy)
        {
            EXPECT_DOUBLE_EQ(f(psx - 1, iy), -interiorValue)
                << "comp=" << comp << " lower ghost iy=" << iy;
            EXPECT_DOUBLE_EQ(f(pex + 1, iy), -interiorValue)
                << "comp=" << comp << " upper ghost iy=" << iy;
        }
    }
}

TEST_F(VecFieldBC2D, AntiSymmetricAtYBoundaries)
{
    FieldAntiSymmetricBoundaryCondition<VecField2D, GridLayout2D> bc;
    bc.apply(B, BoundaryLocation::YLower, yLowerGhostCellBox2D(), layout, 0.0, acc);
    bc.apply(B, BoundaryLocation::YUpper, yUpperGhostCellBox2D(), layout, 0.0, acc);

    // By: primal in Y (normal to Y boundary) → Neumann
    {
        auto& By          = B[1];
        auto byQty        = HybridQuantity::Scalar::By;
        std::uint32_t psy = layout.physicalStartIndex(byQty, Direction::Y);
        std::uint32_t pey = layout.physicalEndIndex(byQty, Direction::Y);
        std::uint32_t sx  = layout.physicalStartIndex(byQty, Direction::X);
        std::uint32_t ex  = layout.physicalEndIndex(byQty, Direction::X);
        for (std::uint32_t ix = sx; ix <= ex; ++ix)
        {
            EXPECT_DOUBLE_EQ(By(ix, psy - 1), interiorValue) << "By lower ghost ix=" << ix;
            EXPECT_DOUBLE_EQ(By(ix, pey + 1), interiorValue) << "By upper ghost ix=" << ix;
        }
    }

    // Bx, Bz: dual in Y (tangential to Y boundary) → Dirichlet(0)
    for (std::size_t comp : {0u, 2u})
    {
        auto& f           = B[comp];
        auto qty          = HybridQuantity::componentsQuantities(vecQty)[comp];
        std::uint32_t psy = layout.physicalStartIndex(qty, Direction::Y);
        std::uint32_t pey = layout.physicalEndIndex(qty, Direction::Y);
        std::uint32_t sx  = layout.physicalStartIndex(qty, Direction::X);
        std::uint32_t ex  = layout.physicalEndIndex(qty, Direction::X);
        for (std::uint32_t ix = sx; ix <= ex; ++ix)
        {
            EXPECT_DOUBLE_EQ(f(ix, psy - 1), -interiorValue)
                << "comp=" << comp << " lower ghost ix=" << ix;
            EXPECT_DOUBLE_EQ(f(ix, pey + 1), -interiorValue)
                << "comp=" << comp << " upper ghost ix=" << ix;
        }
    }
}


// ─── DivergenceFreeTransverseDirichlet VecField 2D ────────────────────────────

TEST_F(VecFieldBC2D, DivergenceFreeTransverseDirichletAtXBoundaries)
{
    std::array values{123.0, 7.0, 11.0};
    FieldDivergenceFreeTransverseDirichletBoundaryCondition<VecField2D, GridLayout2D> bc{values};
    bc.apply(B, BoundaryLocation::XLower, xLowerGhostCellBox2D(), layout, 0.0, acc);
    bc.apply(B, BoundaryLocation::XUpper, xUpperGhostCellBox2D(), layout, 0.0, acc);

    auto& Bx          = B[0];
    auto bxQty        = HybridQuantity::Scalar::Bx;
    std::uint32_t psx = layout.physicalStartIndex(bxQty, Direction::X);
    std::uint32_t pex = layout.physicalEndIndex(bxQty, Direction::X);
    std::uint32_t sy  = layout.physicalStartIndex(bxQty, Direction::Y);
    std::uint32_t ey  = layout.physicalEndIndex(bxQty, Direction::Y);
    for (std::uint32_t iy = sy; iy <= ey; ++iy)
    {
        EXPECT_DOUBLE_EQ(Bx(psx - 2, iy), interiorValue) << "Bx lower far ghost iy=" << iy;
        EXPECT_DOUBLE_EQ(Bx(psx - 1, iy), interiorValue) << "Bx lower near ghost iy=" << iy;
        EXPECT_DOUBLE_EQ(Bx(pex + 1, iy), interiorValue) << "Bx upper near ghost iy=" << iy;
        EXPECT_DOUBLE_EQ(Bx(pex + 2, iy), interiorValue) << "Bx upper far ghost iy=" << iy;
    }

    for (std::size_t comp : {1u, 2u})
    {
        auto& f           = B[comp];
        auto qty          = HybridQuantity::componentsQuantities(vecQty)[comp];
        std::uint32_t psx = layout.physicalStartIndex(qty, Direction::X);
        std::uint32_t pex = layout.physicalEndIndex(qty, Direction::X);
        std::uint32_t sy  = layout.physicalStartIndex(qty, Direction::Y);
        std::uint32_t ey  = layout.physicalEndIndex(qty, Direction::Y);
        double expectedGhost = 2.0 * values[comp] - interiorValue;
        for (std::uint32_t iy = sy; iy <= ey; ++iy)
        {
            EXPECT_DOUBLE_EQ(f(psx - 2, iy), expectedGhost)
                << "comp=" << comp << " lower far ghost iy=" << iy;
            EXPECT_DOUBLE_EQ(f(psx - 1, iy), expectedGhost)
                << "comp=" << comp << " lower near ghost iy=" << iy;
            EXPECT_DOUBLE_EQ(f(pex + 1, iy), expectedGhost)
                << "comp=" << comp << " upper near ghost iy=" << iy;
            EXPECT_DOUBLE_EQ(f(pex + 2, iy), expectedGhost)
                << "comp=" << comp << " upper far ghost iy=" << iy;
        }
    }
}

TEST_F(VecFieldBC2D, DivergenceFreeTransverseDirichletAtYBoundaries)
{
    std::array values{3.0, 123.0, 11.0};
    FieldDivergenceFreeTransverseDirichletBoundaryCondition<VecField2D, GridLayout2D> bc{values};
    bc.apply(B, BoundaryLocation::YLower, yLowerGhostCellBox2D(), layout, 0.0, acc);
    bc.apply(B, BoundaryLocation::YUpper, yUpperGhostCellBox2D(), layout, 0.0, acc);

    auto& By          = B[1];
    auto byQty        = HybridQuantity::Scalar::By;
    std::uint32_t psy = layout.physicalStartIndex(byQty, Direction::Y);
    std::uint32_t pey = layout.physicalEndIndex(byQty, Direction::Y);
    std::uint32_t sx  = layout.physicalStartIndex(byQty, Direction::X);
    std::uint32_t ex  = layout.physicalEndIndex(byQty, Direction::X);
    for (std::uint32_t ix = sx; ix <= ex; ++ix)
    {
        EXPECT_DOUBLE_EQ(By(ix, psy - 2), interiorValue) << "By lower far ghost ix=" << ix;
        EXPECT_DOUBLE_EQ(By(ix, psy - 1), interiorValue) << "By lower near ghost ix=" << ix;
        EXPECT_DOUBLE_EQ(By(ix, pey + 1), interiorValue) << "By upper near ghost ix=" << ix;
        EXPECT_DOUBLE_EQ(By(ix, pey + 2), interiorValue) << "By upper far ghost ix=" << ix;
    }

    for (std::size_t comp : {0u, 2u})
    {
        auto& f           = B[comp];
        auto qty          = HybridQuantity::componentsQuantities(vecQty)[comp];
        std::uint32_t psy = layout.physicalStartIndex(qty, Direction::Y);
        std::uint32_t pey = layout.physicalEndIndex(qty, Direction::Y);
        std::uint32_t sx  = layout.physicalStartIndex(qty, Direction::X);
        std::uint32_t ex  = layout.physicalEndIndex(qty, Direction::X);
        double expectedGhost = 2.0 * values[comp] - interiorValue;
        for (std::uint32_t ix = sx; ix <= ex; ++ix)
        {
            EXPECT_DOUBLE_EQ(f(ix, psy - 2), expectedGhost)
                << "comp=" << comp << " lower far ghost ix=" << ix;
            EXPECT_DOUBLE_EQ(f(ix, psy - 1), expectedGhost)
                << "comp=" << comp << " lower near ghost ix=" << ix;
            EXPECT_DOUBLE_EQ(f(ix, pey + 1), expectedGhost)
                << "comp=" << comp << " upper near ghost ix=" << ix;
            EXPECT_DOUBLE_EQ(f(ix, pey + 2), expectedGhost)
                << "comp=" << comp << " upper far ghost ix=" << ix;
        }
    }
}

TEST_F(VecFieldBC2DNonUniformBy, DivergenceFreeTransverseDirichletKeepsXGhostDivergenceZero)
{
    FieldDivergenceFreeTransverseDirichletBoundaryCondition<VecField2D, GridLayout2D> bc{
        std::array{123.0, 0.0, 11.0}};
    bc.apply(B, BoundaryLocation::XLower, xLowerGhostCellBox2D(), layout, 0.0, acc);
    bc.apply(B, BoundaryLocation::XUpper, xUpperGhostCellBox2D(), layout, 0.0, acc);

    auto& Bx = B[0];
    auto& By = B[1];
    for (auto const& index : xLowerGhostCellBox2D())
    {
        EXPECT_DOUBLE_EQ(Bx(index.template neighbor<0, 1>()) - Bx(index)
                             + By(index.template neighbor<1, 1>()) - By(index),
                         0.0)
            << "lower divergence at (" << index[0] << ", " << index[1] << ")";
    }
    for (auto const& index : xUpperGhostCellBox2D())
    {
        EXPECT_DOUBLE_EQ(Bx(index.template neighbor<0, 1>()) - Bx(index)
                             + By(index.template neighbor<1, 1>()) - By(index),
                         0.0)
            << "upper divergence at (" << index[0] << ", " << index[1] << ")";
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// 3D tests
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr std::uint32_t nCellsX3D = 10u;
static constexpr std::uint32_t nCellsY3D = 8u;
static constexpr std::uint32_t nCellsZ3D = 6u;

using GridLayout3D = GridLayout<GridLayoutImplYee<3, 1>>;
using NdArray3D    = NdArrayVector<3, double>;
using Grid3D       = Grid<NdArray3D, HybridQuantity::Scalar>;
using Field3D      = Grid3D::field_type;
using VecField3D   = VecField<Field3D, HybridQuantity>;

// Ghost cell boxes for Z boundaries: full XY plane × normal-direction strip.
Box<std::uint32_t, 3> zLowerGhostCellBox3D()
{
    return {{0u, 0u, 0u},
            {nCellsX3D + 2u * ghostWidth - 1u, nCellsY3D + 2u * ghostWidth - 1u,
             ghostWidth - 1u}};
}
Box<std::uint32_t, 3> zUpperGhostCellBox3D()
{
    return {{0u, 0u, ghostWidth + nCellsZ3D},
            {nCellsX3D + 2u * ghostWidth - 1u, nCellsY3D + 2u * ghostWidth - 1u,
             2u * ghostWidth + nCellsZ3D - 1u}};
}

/**
 * @brief 3D VecField (B) fixture. ghostWidth=2 (order 1).
 *
 * B field centerings in Yee-3D:
 *   Bx: primal X, dual   Y, dual   Z
 *   By: dual   X, primal Y, dual   Z
 *   Bz: dual   X, dual   Y, primal Z
 *
 * Interior cells are filled with @p interiorValue, ghost cells with @p ghostSentinel.
 */
struct VecFieldBC3D : testing::Test
{
    GridLayout3D layout{{0.1, 0.1, 0.1}, {nCellsX3D, nCellsY3D, nCellsZ3D}, {0.0, 0.0, 0.0}};
    NullFieldAccessorT<Field3D> acc;

    static constexpr auto vecQty = HybridQuantity::Vector::B;
    UsableTensorField<3, 1> B{"B", layout, vecQty};

    VecFieldBC3D()
    {
        for (std::size_t comp = 0; comp < 3; ++comp)
        {
            auto& f    = B[comp];
            auto shape = f.shape();
            for (std::uint32_t ix = 0; ix < shape[0]; ++ix)
                for (std::uint32_t iy = 0; iy < shape[1]; ++iy)
                    for (std::uint32_t iz = 0; iz < shape[2]; ++iz)
                        f(ix, iy, iz) = ghostSentinel;
            auto qty         = HybridQuantity::componentsQuantities(vecQty)[comp];
            std::uint32_t sx = layout.physicalStartIndex(qty, Direction::X);
            std::uint32_t ex = layout.physicalEndIndex(qty, Direction::X);
            std::uint32_t sy = layout.physicalStartIndex(qty, Direction::Y);
            std::uint32_t ey = layout.physicalEndIndex(qty, Direction::Y);
            std::uint32_t sz = layout.physicalStartIndex(qty, Direction::Z);
            std::uint32_t ez = layout.physicalEndIndex(qty, Direction::Z);
            for (std::uint32_t ix = sx; ix <= ex; ++ix)
                for (std::uint32_t iy = sy; iy <= ey; ++iy)
                    for (std::uint32_t iz = sz; iz <= ez; ++iz)
                        f(ix, iy, iz) = interiorValue;
        }
    }
};


// ─── Symmetric VecField 3D ───────────────────────────────────────────────────

TEST_F(VecFieldBC3D, SymmetricAtZBoundaries)
{
    FieldSymmetricBoundaryCondition<VecField3D, GridLayout3D> bc;
    bc.apply(B, BoundaryLocation::ZLower, zLowerGhostCellBox3D(), layout, 0.0, acc);
    bc.apply(B, BoundaryLocation::ZUpper, zUpperGhostCellBox3D(), layout, 0.0, acc);

    // Bz: primal in Z (normal to Z boundary) → Dirichlet(0)
    {
        auto& Bz          = B[2];
        auto bzQty        = HybridQuantity::Scalar::Bz;
        std::uint32_t psz = layout.physicalStartIndex(bzQty, Direction::Z);
        std::uint32_t pez = layout.physicalEndIndex(bzQty, Direction::Z);
        std::uint32_t sx  = layout.physicalStartIndex(bzQty, Direction::X);
        std::uint32_t ex  = layout.physicalEndIndex(bzQty, Direction::X);
        std::uint32_t sy  = layout.physicalStartIndex(bzQty, Direction::Y);
        std::uint32_t ey  = layout.physicalEndIndex(bzQty, Direction::Y);
        for (std::uint32_t ix = sx; ix <= ex; ++ix)
            for (std::uint32_t iy = sy; iy <= ey; ++iy)
            {
                EXPECT_DOUBLE_EQ(Bz(ix, iy, psz - 1), -interiorValue)
                    << "Bz lower ghost ix=" << ix << " iy=" << iy;
                EXPECT_DOUBLE_EQ(Bz(ix, iy, psz),     0.0)
                    << "Bz lower boundary ix=" << ix << " iy=" << iy;
                EXPECT_DOUBLE_EQ(Bz(ix, iy, pez),     0.0)
                    << "Bz upper boundary ix=" << ix << " iy=" << iy;
                EXPECT_DOUBLE_EQ(Bz(ix, iy, pez + 1), -interiorValue)
                    << "Bz upper ghost ix=" << ix << " iy=" << iy;
            }
    }

    // Bx, By: dual in Z (tangential to Z boundary) → Neumann
    for (std::size_t comp : {0u, 1u})
    {
        auto& f           = B[comp];
        auto qty          = HybridQuantity::componentsQuantities(vecQty)[comp];
        std::uint32_t psz = layout.physicalStartIndex(qty, Direction::Z);
        std::uint32_t pez = layout.physicalEndIndex(qty, Direction::Z);
        std::uint32_t sx  = layout.physicalStartIndex(qty, Direction::X);
        std::uint32_t ex  = layout.physicalEndIndex(qty, Direction::X);
        std::uint32_t sy  = layout.physicalStartIndex(qty, Direction::Y);
        std::uint32_t ey  = layout.physicalEndIndex(qty, Direction::Y);
        for (std::uint32_t ix = sx; ix <= ex; ++ix)
            for (std::uint32_t iy = sy; iy <= ey; ++iy)
            {
                EXPECT_DOUBLE_EQ(f(ix, iy, psz - 1), interiorValue)
                    << "comp=" << comp << " lower ghost ix=" << ix << " iy=" << iy;
                EXPECT_DOUBLE_EQ(f(ix, iy, pez + 1), interiorValue)
                    << "comp=" << comp << " upper ghost ix=" << ix << " iy=" << iy;
            }
    }
}


// ─── AntiSymmetric VecField 3D ───────────────────────────────────────────────

TEST_F(VecFieldBC3D, AntiSymmetricAtZBoundaries)
{
    FieldAntiSymmetricBoundaryCondition<VecField3D, GridLayout3D> bc;
    bc.apply(B, BoundaryLocation::ZLower, zLowerGhostCellBox3D(), layout, 0.0, acc);
    bc.apply(B, BoundaryLocation::ZUpper, zUpperGhostCellBox3D(), layout, 0.0, acc);

    // Bz: primal in Z (normal to Z boundary) → Neumann
    {
        auto& Bz          = B[2];
        auto bzQty        = HybridQuantity::Scalar::Bz;
        std::uint32_t psz = layout.physicalStartIndex(bzQty, Direction::Z);
        std::uint32_t pez = layout.physicalEndIndex(bzQty, Direction::Z);
        std::uint32_t sx  = layout.physicalStartIndex(bzQty, Direction::X);
        std::uint32_t ex  = layout.physicalEndIndex(bzQty, Direction::X);
        std::uint32_t sy  = layout.physicalStartIndex(bzQty, Direction::Y);
        std::uint32_t ey  = layout.physicalEndIndex(bzQty, Direction::Y);
        for (std::uint32_t ix = sx; ix <= ex; ++ix)
            for (std::uint32_t iy = sy; iy <= ey; ++iy)
            {
                EXPECT_DOUBLE_EQ(Bz(ix, iy, psz - 1), interiorValue)
                    << "Bz lower ghost ix=" << ix << " iy=" << iy;
                EXPECT_DOUBLE_EQ(Bz(ix, iy, pez + 1), interiorValue)
                    << "Bz upper ghost ix=" << ix << " iy=" << iy;
            }
    }

    // Bx, By: dual in Z (tangential to Z boundary) → Dirichlet(0)
    for (std::size_t comp : {0u, 1u})
    {
        auto& f           = B[comp];
        auto qty          = HybridQuantity::componentsQuantities(vecQty)[comp];
        std::uint32_t psz = layout.physicalStartIndex(qty, Direction::Z);
        std::uint32_t pez = layout.physicalEndIndex(qty, Direction::Z);
        std::uint32_t sx  = layout.physicalStartIndex(qty, Direction::X);
        std::uint32_t ex  = layout.physicalEndIndex(qty, Direction::X);
        std::uint32_t sy  = layout.physicalStartIndex(qty, Direction::Y);
        std::uint32_t ey  = layout.physicalEndIndex(qty, Direction::Y);
        for (std::uint32_t ix = sx; ix <= ex; ++ix)
            for (std::uint32_t iy = sy; iy <= ey; ++iy)
            {
                EXPECT_DOUBLE_EQ(f(ix, iy, psz - 1), -interiorValue)
                    << "comp=" << comp << " lower ghost ix=" << ix << " iy=" << iy;
                EXPECT_DOUBLE_EQ(f(ix, iy, pez + 1), -interiorValue)
                    << "comp=" << comp << " upper ghost ix=" << ix << " iy=" << iy;
        }
    }
}


// ─── DivergenceFreeTransverseDirichlet VecField 3D ────────────────────────────

TEST_F(VecFieldBC3D, DivergenceFreeTransverseDirichletAtZBoundaries)
{
    std::array values{3.0, 7.0, 123.0};
    FieldDivergenceFreeTransverseDirichletBoundaryCondition<VecField3D, GridLayout3D> bc{values};
    bc.apply(B, BoundaryLocation::ZLower, zLowerGhostCellBox3D(), layout, 0.0, acc);
    bc.apply(B, BoundaryLocation::ZUpper, zUpperGhostCellBox3D(), layout, 0.0, acc);

    auto& Bz          = B[2];
    auto bzQty        = HybridQuantity::Scalar::Bz;
    std::uint32_t psz = layout.physicalStartIndex(bzQty, Direction::Z);
    std::uint32_t pez = layout.physicalEndIndex(bzQty, Direction::Z);
    std::uint32_t sx  = layout.physicalStartIndex(bzQty, Direction::X);
    std::uint32_t ex  = layout.physicalEndIndex(bzQty, Direction::X);
    std::uint32_t sy  = layout.physicalStartIndex(bzQty, Direction::Y);
    std::uint32_t ey  = layout.physicalEndIndex(bzQty, Direction::Y);
    for (std::uint32_t ix = sx; ix <= ex; ++ix)
        for (std::uint32_t iy = sy; iy <= ey; ++iy)
        {
            EXPECT_DOUBLE_EQ(Bz(ix, iy, psz - 2), interiorValue)
                << "Bz lower far ghost ix=" << ix << " iy=" << iy;
            EXPECT_DOUBLE_EQ(Bz(ix, iy, psz - 1), interiorValue)
                << "Bz lower near ghost ix=" << ix << " iy=" << iy;
            EXPECT_DOUBLE_EQ(Bz(ix, iy, pez + 1), interiorValue)
                << "Bz upper near ghost ix=" << ix << " iy=" << iy;
            EXPECT_DOUBLE_EQ(Bz(ix, iy, pez + 2), interiorValue)
                << "Bz upper far ghost ix=" << ix << " iy=" << iy;
        }

    for (std::size_t comp : {0u, 1u})
    {
        auto& f           = B[comp];
        auto qty          = HybridQuantity::componentsQuantities(vecQty)[comp];
        std::uint32_t psz = layout.physicalStartIndex(qty, Direction::Z);
        std::uint32_t pez = layout.physicalEndIndex(qty, Direction::Z);
        std::uint32_t sx  = layout.physicalStartIndex(qty, Direction::X);
        std::uint32_t ex  = layout.physicalEndIndex(qty, Direction::X);
        std::uint32_t sy  = layout.physicalStartIndex(qty, Direction::Y);
        std::uint32_t ey  = layout.physicalEndIndex(qty, Direction::Y);
        double expectedGhost = 2.0 * values[comp] - interiorValue;
        for (std::uint32_t ix = sx; ix <= ex; ++ix)
            for (std::uint32_t iy = sy; iy <= ey; ++iy)
            {
                EXPECT_DOUBLE_EQ(f(ix, iy, psz - 2), expectedGhost)
                    << "comp=" << comp << " lower far ghost ix=" << ix << " iy=" << iy;
                EXPECT_DOUBLE_EQ(f(ix, iy, psz - 1), expectedGhost)
                    << "comp=" << comp << " lower near ghost ix=" << ix << " iy=" << iy;
                EXPECT_DOUBLE_EQ(f(ix, iy, pez + 1), expectedGhost)
                    << "comp=" << comp << " upper near ghost ix=" << ix << " iy=" << iy;
                EXPECT_DOUBLE_EQ(f(ix, iy, pez + 2), expectedGhost)
                    << "comp=" << comp << " upper far ghost ix=" << ix << " iy=" << iy;
            }
    }
}




// ═══════════════════════════════════════════════════════════════════════════════
// FieldTotalEnergyFromPressureBoundaryCondition tests (MHD, 1D/2D/3D)
// ═══════════════════════════════════════════════════════════════════════════════

// MHD Yee layout with reconstruction_nghosts=1:
//   ghost_width = roundUpToEven(1+2) = 4  (same for all spatial dims)
static constexpr std::uint32_t mhdGhostWidth = GridLayoutImplYeeMHD<1, 1, 1>::ghost_width;

/**
 * @brief Concrete patch-field accessor for MHD unit tests (dim-templated).
 *
 * Holds references to the scalar grids and vector field fixtures that the
 * FieldTotalEnergyFromPressureBoundaryCondition needs at runtime.
 * getVecField() copy-constructs a VecField view from the underlying
 * UsableVecFieldMHD; buffer pointers are preserved so writes through the
 * returned view affect the actual data.
 */
template<std::size_t dim>
struct MHDPatchFieldAccessorTest : IPatchFieldAccessor<FieldMHD<dim>, MHDQuantity>
{
    using GridMHDd     = Grid<NdArrayVector<dim, double>, MHDQuantity::Scalar>;
    using VecFieldMHDd = VecFieldMHD<dim>;

    GridMHDd&              rho;
    GridMHDd&              P;
    GridMHDd&              Etot;
    UsableVecFieldMHD<dim>& rhoV;
    UsableVecFieldMHD<dim>& Bvec;

    MHDPatchFieldAccessorTest(GridMHDd& rho_, GridMHDd& P_, GridMHDd& Etot_,
                              UsableVecFieldMHD<dim>& rhoV_, UsableVecFieldMHD<dim>& Bvec_)
        : rho{rho_}
        , P{P_}
        , Etot{Etot_}
        , rhoV{rhoV_}
        , Bvec{Bvec_}
    {
    }

    FieldMHD<dim>& getField(MHDQuantity::Scalar qty) const override
    {
        switch (qty)
        {
            case MHDQuantity::Scalar::rho:  return *(&rho);
            case MHDQuantity::Scalar::P:    return *(&P);
            case MHDQuantity::Scalar::Etot: return *(&Etot);
            default: throw std::runtime_error("MHDPatchFieldAccessorTest: unsupported scalar qty");
        }
    }

    VecFieldMHDd getVecField(MHDQuantity::Vector qty) const override
    {
        switch (qty)
        {
            case MHDQuantity::Vector::rhoV: return rhoV.super();
            case MHDQuantity::Vector::B:    return Bvec.super();
            default:
                throw std::runtime_error("MHDPatchFieldAccessorTest: unsupported vector qty");
        }
    }
};

// ─── 1D MHD ──────────────────────────────────────────────────────────────────

using GridLayoutMHD1D = GridLayout<GridLayoutImplYeeMHD<1, 1, 1>>;
using GridMHD1D       = Grid<NdArrayVector<1, double>, MHDQuantity::Scalar>;

static constexpr std::uint32_t nCellsMHD = 10u;

Box<std::uint32_t, 1> mhdLowerGhostCellBox()
{
    return {Point<std::uint32_t, 1>{0u}, Point<std::uint32_t, 1>{mhdGhostWidth - 1u}};
}
Box<std::uint32_t, 1> mhdUpperGhostCellBox()
{
    return {Point<std::uint32_t, 1>{mhdGhostWidth + nCellsMHD},
            Point<std::uint32_t, 1>{2u * mhdGhostWidth + nCellsMHD - 1u}};
}


/**
 * @brief 1D MHD fixture for FieldTotalEnergyFromPressureBoundaryCondition tests.
 *
 * Sets a uniform thermodynamic state (ρ, vx, vy, vz, Bx, By, Bz, P) over the
 * interior, derives Etot from it via the ideal-gas EOS, and leaves ghost cells
 * at the sentinel value.
 *
 * Physical state used (γ = 5/3):
 *   ρ=2, vx=vy=vz=1, Bx=By=Bz=0.5, P=1
 *   e_int (volumetric) = ρ·P/(ρ·(γ-1)) = 1.5
 *   Etot = 1.5 + ½ρv² + ½B² = 1.5 + 3.0 + 0.375 = 4.875
 */
struct EtotFromPressureBC1D : testing::Test
{
    static constexpr double gamma    = 5.0 / 3.0;
    static constexpr double rho_val  = 2.0;
    static constexpr double vx_val   = 1.0;
    static constexpr double vy_val   = 1.0;
    static constexpr double vz_val   = 1.0;
    static constexpr double Bx_val   = 0.5;
    static constexpr double By_val   = 0.5;
    static constexpr double Bz_val   = 0.5;
    static constexpr double P_val    = 1.0;
    static constexpr double sentinel = -999.0;

    // Etot = ρ·u + ½ρ|v|² + ½|B|²  with u = P/(ρ(γ-1))
    static constexpr double u_specific = P_val / (rho_val * (gamma - 1.0));
    static constexpr double etot_val
        = rho_val * u_specific
          + 0.5 * rho_val * (vx_val * vx_val + vy_val * vy_val + vz_val * vz_val)
          + 0.5 * (Bx_val * Bx_val + By_val * By_val + Bz_val * Bz_val);

    GridLayoutMHD1D layout{{0.1}, {nCellsMHD}, {0.0}};

    GridMHD1D rhoGrid {"rho",  MHDQuantity::Scalar::rho,  layout.allocSize(MHDQuantity::Scalar::rho)};
    GridMHD1D PGrid   {"P",    MHDQuantity::Scalar::P,    layout.allocSize(MHDQuantity::Scalar::P)};
    GridMHD1D EtotGrid{"Etot", MHDQuantity::Scalar::Etot, layout.allocSize(MHDQuantity::Scalar::Etot)};

    UsableVecFieldMHD<1> rhoV{"rhoV", layout, MHDQuantity::Vector::rhoV};
    UsableVecFieldMHD<1> Bvec{"B",    layout, MHDQuantity::Vector::B};

    MHDPatchFieldAccessorTest<1> acc{rhoGrid, PGrid, EtotGrid, rhoV, Bvec};

    FieldMHD<1>& rhoField  {*(&rhoGrid)};
    FieldMHD<1>& PField    {*(&PGrid)};
    FieldMHD<1>& EtotField {*(&EtotGrid)};

    EtotFromPressureBC1D()
    {
        auto fill_scalar = [&](auto& f, MHDQuantity::Scalar qty, double interior_val) {
            for (std::uint32_t i = 0; i < f.shape()[0]; ++i)
                f(i) = sentinel;
            std::uint32_t ps = layout.physicalStartIndex(qty, Direction::X);
            std::uint32_t pe = layout.physicalEndIndex(qty, Direction::X);
            for (std::uint32_t i = ps; i <= pe; ++i)
                f(i) = interior_val;
        };

        fill_scalar(rhoField,  MHDQuantity::Scalar::rho,  rho_val);
        fill_scalar(PField,    MHDQuantity::Scalar::P,    P_val);
        fill_scalar(EtotField, MHDQuantity::Scalar::Etot, etot_val);

        fill_scalar(rhoV[0], MHDQuantity::Scalar::rhoVx, rho_val * vx_val);
        fill_scalar(rhoV[1], MHDQuantity::Scalar::rhoVy, rho_val * vy_val);
        fill_scalar(rhoV[2], MHDQuantity::Scalar::rhoVz, rho_val * vz_val);

        // Bx is primal in X; By and Bz are dual
        fill_scalar(Bvec[0], MHDQuantity::Scalar::Bx, Bx_val);
        fill_scalar(Bvec[1], MHDQuantity::Scalar::By, By_val);
        fill_scalar(Bvec[2], MHDQuantity::Scalar::Bz, Bz_val);
    }
};


// ─── TotalEnergyFromPressure BC tests ─────────────────────────────────────────

/**
 * @brief Neumann on ρ, ρv, B, P → ghost Etot must equal the interior mirror Etot.
 *
 * With a uniform interior state all Neumann sub-BCs simply copy the interior
 * value into the ghost layer.  Step 1 recovers P from Etot+conservative vars
 * (same as the stored P), step 2 fills ghosts with the same uniform values, and
 * step 3 reconstructs Etot, which must equal the original interior Etot.
 */
TEST_F(EtotFromPressureBC1D, NeumannSubBCsGhostEtotEqualsInteriorEtot)
{
    auto rho_bc  = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<1>, GridLayoutMHD1D>>();
    auto P_bc    = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<1>, GridLayoutMHD1D>>();
    auto rhoV_bc = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<1>, GridLayoutMHD1D>>();
    auto B_bc    = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<1>, GridLayoutMHD1D>>();
    auto thermo  = std::make_shared<IdealGasThermo>(gamma);

    FieldTotalEnergyFromPressureBoundaryCondition<FieldMHD<1>, GridLayoutMHD1D> bc{
        rho_bc, rhoV_bc, B_bc, P_bc, thermo};

    bc.apply(EtotField, BoundaryLocation::XLower, mhdLowerGhostCellBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::XUpper, mhdUpperGhostCellBox(), layout, 0.0, acc);

    auto etotQty     = MHDQuantity::Scalar::Etot;
    std::uint32_t ps = layout.physicalStartIndex(etotQty, Direction::X);
    std::uint32_t pe = layout.physicalEndIndex(etotQty, Direction::X);

    for (std::uint32_t g = 0; g < mhdGhostWidth; ++g)
    {
        EXPECT_NEAR(EtotField(ps - 1u - g), etot_val, 1e-12) << "lower ghost g=" << g;
        EXPECT_NEAR(EtotField(pe + 1u + g), etot_val, 1e-12) << "upper ghost g=" << g;
    }
}

/**
 * @brief Interior Etot must remain unchanged after the BC is applied.
 *
 * The BC only writes ghost cells; it must leave the physical domain intact.
 */
TEST_F(EtotFromPressureBC1D, InteriorEtotUnchangedAfterBC)
{
    auto rho_bc  = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<1>, GridLayoutMHD1D>>();
    auto P_bc    = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<1>, GridLayoutMHD1D>>();
    auto rhoV_bc = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<1>, GridLayoutMHD1D>>();
    auto B_bc    = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<1>, GridLayoutMHD1D>>();
    auto thermo  = std::make_shared<IdealGasThermo>(gamma);

    FieldTotalEnergyFromPressureBoundaryCondition<FieldMHD<1>, GridLayoutMHD1D> bc{
        rho_bc, rhoV_bc, B_bc, P_bc, thermo};

    bc.apply(EtotField, BoundaryLocation::XLower, mhdLowerGhostCellBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::XUpper, mhdUpperGhostCellBox(), layout, 0.0, acc);

    auto etotQty     = MHDQuantity::Scalar::Etot;
    std::uint32_t ps = layout.physicalStartIndex(etotQty, Direction::X);
    std::uint32_t pe = layout.physicalEndIndex(etotQty, Direction::X);

    for (std::uint32_t i = ps; i <= pe; ++i)
        EXPECT_DOUBLE_EQ(EtotField(i), etot_val) << "interior index i=" << i;
}


// ─── 2D MHD ──────────────────────────────────────────────────────────────────

using GridLayoutMHD2D = GridLayout<GridLayoutImplYeeMHD<2, 1, 1>>;
using GridMHD2D       = Grid<NdArrayVector<2, double>, MHDQuantity::Scalar>;

static constexpr std::uint32_t nCellsMHDX2D = 10u;
static constexpr std::uint32_t nCellsMHDY2D = 8u;

Box<std::uint32_t, 2> mhd2DXLowerGhostBox()
{
    return {{0u, 0u}, {mhdGhostWidth - 1u, nCellsMHDY2D + 2u * mhdGhostWidth - 1u}};
}
Box<std::uint32_t, 2> mhd2DXUpperGhostBox()
{
    return {{mhdGhostWidth + nCellsMHDX2D, 0u},
            {2u * mhdGhostWidth + nCellsMHDX2D - 1u, nCellsMHDY2D + 2u * mhdGhostWidth - 1u}};
}
Box<std::uint32_t, 2> mhd2DYLowerGhostBox()
{
    return {{0u, 0u}, {nCellsMHDX2D + 2u * mhdGhostWidth - 1u, mhdGhostWidth - 1u}};
}
Box<std::uint32_t, 2> mhd2DYUpperGhostBox()
{
    return {{0u, mhdGhostWidth + nCellsMHDY2D},
            {nCellsMHDX2D + 2u * mhdGhostWidth - 1u, 2u * mhdGhostWidth + nCellsMHDY2D - 1u}};
}

/**
 * @brief 2D MHD fixture for FieldTotalEnergyFromPressureBoundaryCondition tests.
 *
 * Same uniform physical state as the 1D fixture (γ=5/3, ρ=2, v=1, B=0.5, P=1,
 * Etot=4.875), extended to a 2D nCellsMHDX2D×nCellsMHDY2D grid.
 */
struct EtotFromPressureBC2D : testing::Test
{
    static constexpr double gamma    = 5.0 / 3.0;
    static constexpr double rho_val  = 2.0;
    static constexpr double vx_val   = 1.0;
    static constexpr double vy_val   = 1.0;
    static constexpr double vz_val   = 1.0;
    static constexpr double Bx_val   = 0.5;
    static constexpr double By_val   = 0.5;
    static constexpr double Bz_val   = 0.5;
    static constexpr double P_val    = 1.0;
    static constexpr double sentinel = -999.0;

    static constexpr double u_specific = P_val / (rho_val * (gamma - 1.0));
    static constexpr double etot_val
        = rho_val * u_specific
          + 0.5 * rho_val * (vx_val * vx_val + vy_val * vy_val + vz_val * vz_val)
          + 0.5 * (Bx_val * Bx_val + By_val * By_val + Bz_val * Bz_val);

    GridLayoutMHD2D layout{{0.1, 0.1}, {nCellsMHDX2D, nCellsMHDY2D}, {0.0, 0.0}};

    GridMHD2D rhoGrid {"rho",  MHDQuantity::Scalar::rho,  layout.allocSize(MHDQuantity::Scalar::rho)};
    GridMHD2D PGrid   {"P",    MHDQuantity::Scalar::P,    layout.allocSize(MHDQuantity::Scalar::P)};
    GridMHD2D EtotGrid{"Etot", MHDQuantity::Scalar::Etot, layout.allocSize(MHDQuantity::Scalar::Etot)};

    UsableVecFieldMHD<2> rhoV{"rhoV", layout, MHDQuantity::Vector::rhoV};
    UsableVecFieldMHD<2> Bvec{"B",    layout, MHDQuantity::Vector::B};

    MHDPatchFieldAccessorTest<2> acc{rhoGrid, PGrid, EtotGrid, rhoV, Bvec};

    FieldMHD<2>& rhoField  {*(&rhoGrid)};
    FieldMHD<2>& PField    {*(&PGrid)};
    FieldMHD<2>& EtotField {*(&EtotGrid)};

    EtotFromPressureBC2D()
    {
        auto fill_scalar = [&](auto& f, MHDQuantity::Scalar qty, double interior_val) {
            auto const sz = f.shape();
            for (std::uint32_t ix = 0; ix < sz[0]; ++ix)
                for (std::uint32_t iy = 0; iy < sz[1]; ++iy)
                    f(ix, iy) = sentinel;
            std::uint32_t ps_x = layout.physicalStartIndex(qty, Direction::X);
            std::uint32_t pe_x = layout.physicalEndIndex(qty, Direction::X);
            std::uint32_t ps_y = layout.physicalStartIndex(qty, Direction::Y);
            std::uint32_t pe_y = layout.physicalEndIndex(qty, Direction::Y);
            for (std::uint32_t ix = ps_x; ix <= pe_x; ++ix)
                for (std::uint32_t iy = ps_y; iy <= pe_y; ++iy)
                    f(ix, iy) = interior_val;
        };

        fill_scalar(rhoField,  MHDQuantity::Scalar::rho,  rho_val);
        fill_scalar(PField,    MHDQuantity::Scalar::P,    P_val);
        fill_scalar(EtotField, MHDQuantity::Scalar::Etot, etot_val);

        fill_scalar(rhoV[0], MHDQuantity::Scalar::rhoVx, rho_val * vx_val);
        fill_scalar(rhoV[1], MHDQuantity::Scalar::rhoVy, rho_val * vy_val);
        fill_scalar(rhoV[2], MHDQuantity::Scalar::rhoVz, rho_val * vz_val);

        fill_scalar(Bvec[0], MHDQuantity::Scalar::Bx, Bx_val);
        fill_scalar(Bvec[1], MHDQuantity::Scalar::By, By_val);
        fill_scalar(Bvec[2], MHDQuantity::Scalar::Bz, Bz_val);
    }
};

/**
 * @brief Neumann sub-BCs on a uniform 2D state → ghost Etot equals interior Etot
 *        on all four patch boundaries.
 */
TEST_F(EtotFromPressureBC2D, NeumannSubBCsGhostEtotEqualsInteriorEtot)
{
    auto rho_bc  = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<2>, GridLayoutMHD2D>>();
    auto P_bc    = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<2>, GridLayoutMHD2D>>();
    auto rhoV_bc = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<2>, GridLayoutMHD2D>>();
    auto B_bc    = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<2>, GridLayoutMHD2D>>();
    auto thermo  = std::make_shared<IdealGasThermo>(gamma);

    FieldTotalEnergyFromPressureBoundaryCondition<FieldMHD<2>, GridLayoutMHD2D> bc{
        rho_bc, rhoV_bc, B_bc, P_bc, thermo};

    bc.apply(EtotField, BoundaryLocation::XLower, mhd2DXLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::XUpper, mhd2DXUpperGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::YLower, mhd2DYLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::YUpper, mhd2DYUpperGhostBox(), layout, 0.0, acc);

    for (auto const& idx : mhd2DXLowerGhostBox())
        EXPECT_NEAR(EtotField(idx), etot_val, 1e-12) << "XLower ghost (" << idx[0] << "," << idx[1] << ")";
    for (auto const& idx : mhd2DXUpperGhostBox())
        EXPECT_NEAR(EtotField(idx), etot_val, 1e-12) << "XUpper ghost (" << idx[0] << "," << idx[1] << ")";
    for (auto const& idx : mhd2DYLowerGhostBox())
        EXPECT_NEAR(EtotField(idx), etot_val, 1e-12) << "YLower ghost (" << idx[0] << "," << idx[1] << ")";
    for (auto const& idx : mhd2DYUpperGhostBox())
        EXPECT_NEAR(EtotField(idx), etot_val, 1e-12) << "YUpper ghost (" << idx[0] << "," << idx[1] << ")";
}

/**
 * @brief Interior Etot must remain unchanged after applying all 2D boundaries.
 */
TEST_F(EtotFromPressureBC2D, InteriorEtotUnchangedAfterBC)
{
    auto rho_bc  = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<2>, GridLayoutMHD2D>>();
    auto P_bc    = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<2>, GridLayoutMHD2D>>();
    auto rhoV_bc = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<2>, GridLayoutMHD2D>>();
    auto B_bc    = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<2>, GridLayoutMHD2D>>();
    auto thermo  = std::make_shared<IdealGasThermo>(gamma);

    FieldTotalEnergyFromPressureBoundaryCondition<FieldMHD<2>, GridLayoutMHD2D> bc{
        rho_bc, rhoV_bc, B_bc, P_bc, thermo};

    bc.apply(EtotField, BoundaryLocation::XLower, mhd2DXLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::XUpper, mhd2DXUpperGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::YLower, mhd2DYLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::YUpper, mhd2DYUpperGhostBox(), layout, 0.0, acc);

    auto etotQty     = MHDQuantity::Scalar::Etot;
    std::uint32_t ps_x = layout.physicalStartIndex(etotQty, Direction::X);
    std::uint32_t pe_x = layout.physicalEndIndex(etotQty, Direction::X);
    std::uint32_t ps_y = layout.physicalStartIndex(etotQty, Direction::Y);
    std::uint32_t pe_y = layout.physicalEndIndex(etotQty, Direction::Y);

    for (std::uint32_t ix = ps_x; ix <= pe_x; ++ix)
        for (std::uint32_t iy = ps_y; iy <= pe_y; ++iy)
            EXPECT_DOUBLE_EQ(EtotField(ix, iy), etot_val)
                << "interior index (" << ix << "," << iy << ")";
}


// ─── 3D MHD ──────────────────────────────────────────────────────────────────

using GridLayoutMHD3D = GridLayout<GridLayoutImplYeeMHD<3, 1, 1>>;
using GridMHD3D       = Grid<NdArrayVector<3, double>, MHDQuantity::Scalar>;

static constexpr std::uint32_t nCellsMHDX3D = 10u;
static constexpr std::uint32_t nCellsMHDY3D = 8u;
static constexpr std::uint32_t nCellsMHDZ3D = 6u;

Box<std::uint32_t, 3> mhd3DXLowerGhostBox()
{
    return {{0u, 0u, 0u},
            {mhdGhostWidth - 1u, nCellsMHDY3D + 2u * mhdGhostWidth - 1u,
             nCellsMHDZ3D + 2u * mhdGhostWidth - 1u}};
}
Box<std::uint32_t, 3> mhd3DXUpperGhostBox()
{
    return {{mhdGhostWidth + nCellsMHDX3D, 0u, 0u},
            {2u * mhdGhostWidth + nCellsMHDX3D - 1u, nCellsMHDY3D + 2u * mhdGhostWidth - 1u,
             nCellsMHDZ3D + 2u * mhdGhostWidth - 1u}};
}
Box<std::uint32_t, 3> mhd3DYLowerGhostBox()
{
    return {{0u, 0u, 0u},
            {nCellsMHDX3D + 2u * mhdGhostWidth - 1u, mhdGhostWidth - 1u,
             nCellsMHDZ3D + 2u * mhdGhostWidth - 1u}};
}
Box<std::uint32_t, 3> mhd3DYUpperGhostBox()
{
    return {{0u, mhdGhostWidth + nCellsMHDY3D, 0u},
            {nCellsMHDX3D + 2u * mhdGhostWidth - 1u, 2u * mhdGhostWidth + nCellsMHDY3D - 1u,
             nCellsMHDZ3D + 2u * mhdGhostWidth - 1u}};
}
Box<std::uint32_t, 3> mhd3DZLowerGhostBox()
{
    return {{0u, 0u, 0u},
            {nCellsMHDX3D + 2u * mhdGhostWidth - 1u, nCellsMHDY3D + 2u * mhdGhostWidth - 1u,
             mhdGhostWidth - 1u}};
}
Box<std::uint32_t, 3> mhd3DZUpperGhostBox()
{
    return {{0u, 0u, mhdGhostWidth + nCellsMHDZ3D},
            {nCellsMHDX3D + 2u * mhdGhostWidth - 1u, nCellsMHDY3D + 2u * mhdGhostWidth - 1u,
             2u * mhdGhostWidth + nCellsMHDZ3D - 1u}};
}

/**
 * @brief 3D MHD fixture for FieldTotalEnergyFromPressureBoundaryCondition tests.
 *
 * Same uniform physical state as the 1D/2D fixtures, extended to a 3D grid.
 */
struct EtotFromPressureBC3D : testing::Test
{
    static constexpr double gamma    = 5.0 / 3.0;
    static constexpr double rho_val  = 2.0;
    static constexpr double vx_val   = 1.0;
    static constexpr double vy_val   = 1.0;
    static constexpr double vz_val   = 1.0;
    static constexpr double Bx_val   = 0.5;
    static constexpr double By_val   = 0.5;
    static constexpr double Bz_val   = 0.5;
    static constexpr double P_val    = 1.0;
    static constexpr double sentinel = -999.0;

    static constexpr double u_specific = P_val / (rho_val * (gamma - 1.0));
    static constexpr double etot_val
        = rho_val * u_specific
          + 0.5 * rho_val * (vx_val * vx_val + vy_val * vy_val + vz_val * vz_val)
          + 0.5 * (Bx_val * Bx_val + By_val * By_val + Bz_val * Bz_val);

    GridLayoutMHD3D layout{
        {0.1, 0.1, 0.1}, {nCellsMHDX3D, nCellsMHDY3D, nCellsMHDZ3D}, {0.0, 0.0, 0.0}};

    GridMHD3D rhoGrid {"rho",  MHDQuantity::Scalar::rho,  layout.allocSize(MHDQuantity::Scalar::rho)};
    GridMHD3D PGrid   {"P",    MHDQuantity::Scalar::P,    layout.allocSize(MHDQuantity::Scalar::P)};
    GridMHD3D EtotGrid{"Etot", MHDQuantity::Scalar::Etot, layout.allocSize(MHDQuantity::Scalar::Etot)};

    UsableVecFieldMHD<3> rhoV{"rhoV", layout, MHDQuantity::Vector::rhoV};
    UsableVecFieldMHD<3> Bvec{"B",    layout, MHDQuantity::Vector::B};

    MHDPatchFieldAccessorTest<3> acc{rhoGrid, PGrid, EtotGrid, rhoV, Bvec};

    FieldMHD<3>& rhoField  {*(&rhoGrid)};
    FieldMHD<3>& PField    {*(&PGrid)};
    FieldMHD<3>& EtotField {*(&EtotGrid)};

    EtotFromPressureBC3D()
    {
        auto fill_scalar = [&](auto& f, MHDQuantity::Scalar qty, double interior_val) {
            auto const sz = f.shape();
            for (std::uint32_t ix = 0; ix < sz[0]; ++ix)
                for (std::uint32_t iy = 0; iy < sz[1]; ++iy)
                    for (std::uint32_t iz = 0; iz < sz[2]; ++iz)
                        f(ix, iy, iz) = sentinel;
            std::uint32_t ps_x = layout.physicalStartIndex(qty, Direction::X);
            std::uint32_t pe_x = layout.physicalEndIndex(qty, Direction::X);
            std::uint32_t ps_y = layout.physicalStartIndex(qty, Direction::Y);
            std::uint32_t pe_y = layout.physicalEndIndex(qty, Direction::Y);
            std::uint32_t ps_z = layout.physicalStartIndex(qty, Direction::Z);
            std::uint32_t pe_z = layout.physicalEndIndex(qty, Direction::Z);
            for (std::uint32_t ix = ps_x; ix <= pe_x; ++ix)
                for (std::uint32_t iy = ps_y; iy <= pe_y; ++iy)
                    for (std::uint32_t iz = ps_z; iz <= pe_z; ++iz)
                        f(ix, iy, iz) = interior_val;
        };

        fill_scalar(rhoField,  MHDQuantity::Scalar::rho,  rho_val);
        fill_scalar(PField,    MHDQuantity::Scalar::P,    P_val);
        fill_scalar(EtotField, MHDQuantity::Scalar::Etot, etot_val);

        fill_scalar(rhoV[0], MHDQuantity::Scalar::rhoVx, rho_val * vx_val);
        fill_scalar(rhoV[1], MHDQuantity::Scalar::rhoVy, rho_val * vy_val);
        fill_scalar(rhoV[2], MHDQuantity::Scalar::rhoVz, rho_val * vz_val);

        fill_scalar(Bvec[0], MHDQuantity::Scalar::Bx, Bx_val);
        fill_scalar(Bvec[1], MHDQuantity::Scalar::By, By_val);
        fill_scalar(Bvec[2], MHDQuantity::Scalar::Bz, Bz_val);
    }
};

/**
 * @brief Neumann sub-BCs on a uniform 3D state → ghost Etot equals interior Etot
 *        on all six patch boundaries.
 */
TEST_F(EtotFromPressureBC3D, NeumannSubBCsGhostEtotEqualsInteriorEtot)
{
    auto rho_bc  = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<3>, GridLayoutMHD3D>>();
    auto P_bc    = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<3>, GridLayoutMHD3D>>();
    auto rhoV_bc = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<3>, GridLayoutMHD3D>>();
    auto B_bc    = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<3>, GridLayoutMHD3D>>();
    auto thermo  = std::make_shared<IdealGasThermo>(gamma);

    FieldTotalEnergyFromPressureBoundaryCondition<FieldMHD<3>, GridLayoutMHD3D> bc{
        rho_bc, rhoV_bc, B_bc, P_bc, thermo};

    bc.apply(EtotField, BoundaryLocation::XLower, mhd3DXLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::XUpper, mhd3DXUpperGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::YLower, mhd3DYLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::YUpper, mhd3DYUpperGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::ZLower, mhd3DZLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::ZUpper, mhd3DZUpperGhostBox(), layout, 0.0, acc);

    for (auto const& idx : mhd3DXLowerGhostBox())
        EXPECT_NEAR(EtotField(idx), etot_val, 1e-12)
            << "XLower ghost (" << idx[0] << "," << idx[1] << "," << idx[2] << ")";
    for (auto const& idx : mhd3DXUpperGhostBox())
        EXPECT_NEAR(EtotField(idx), etot_val, 1e-12)
            << "XUpper ghost (" << idx[0] << "," << idx[1] << "," << idx[2] << ")";
    for (auto const& idx : mhd3DYLowerGhostBox())
        EXPECT_NEAR(EtotField(idx), etot_val, 1e-12)
            << "YLower ghost (" << idx[0] << "," << idx[1] << "," << idx[2] << ")";
    for (auto const& idx : mhd3DYUpperGhostBox())
        EXPECT_NEAR(EtotField(idx), etot_val, 1e-12)
            << "YUpper ghost (" << idx[0] << "," << idx[1] << "," << idx[2] << ")";
    for (auto const& idx : mhd3DZLowerGhostBox())
        EXPECT_NEAR(EtotField(idx), etot_val, 1e-12)
            << "ZLower ghost (" << idx[0] << "," << idx[1] << "," << idx[2] << ")";
    for (auto const& idx : mhd3DZUpperGhostBox())
        EXPECT_NEAR(EtotField(idx), etot_val, 1e-12)
            << "ZUpper ghost (" << idx[0] << "," << idx[1] << "," << idx[2] << ")";
}

/**
 * @brief Interior Etot must remain unchanged after applying all 3D boundaries.
 */
TEST_F(EtotFromPressureBC3D, InteriorEtotUnchangedAfterBC)
{
    auto rho_bc  = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<3>, GridLayoutMHD3D>>();
    auto P_bc    = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<3>, GridLayoutMHD3D>>();
    auto rhoV_bc = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<3>, GridLayoutMHD3D>>();
    auto B_bc    = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<3>, GridLayoutMHD3D>>();
    auto thermo  = std::make_shared<IdealGasThermo>(gamma);

    FieldTotalEnergyFromPressureBoundaryCondition<FieldMHD<3>, GridLayoutMHD3D> bc{
        rho_bc, rhoV_bc, B_bc, P_bc, thermo};

    bc.apply(EtotField, BoundaryLocation::XLower, mhd3DXLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::XUpper, mhd3DXUpperGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::YLower, mhd3DYLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::YUpper, mhd3DYUpperGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::ZLower, mhd3DZLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::ZUpper, mhd3DZUpperGhostBox(), layout, 0.0, acc);

    auto etotQty       = MHDQuantity::Scalar::Etot;
    std::uint32_t ps_x = layout.physicalStartIndex(etotQty, Direction::X);
    std::uint32_t pe_x = layout.physicalEndIndex(etotQty, Direction::X);
    std::uint32_t ps_y = layout.physicalStartIndex(etotQty, Direction::Y);
    std::uint32_t pe_y = layout.physicalEndIndex(etotQty, Direction::Y);
    std::uint32_t ps_z = layout.physicalStartIndex(etotQty, Direction::Z);
    std::uint32_t pe_z = layout.physicalEndIndex(etotQty, Direction::Z);

    for (std::uint32_t ix = ps_x; ix <= pe_x; ++ix)
        for (std::uint32_t iy = ps_y; iy <= pe_y; ++iy)
            for (std::uint32_t iz = ps_z; iz <= pe_z; ++iz)
                EXPECT_DOUBLE_EQ(EtotField(ix, iy, iz), etot_val)
                    << "interior index (" << ix << "," << iy << "," << iz << ")";
}


// ════════════════════════════════════════════════════════════════════════════
// FixedPressureOutflow BC tests
//
// The BC uses Neumann for ρ, ρv, B and Dirichlet(P_fixed) for pressure.
// Ghost Etot is reconstructed from the fixed ghost pressure + the Neumann
// (interior-mirror) density, momentum, and magnetic field.
// ════════════════════════════════════════════════════════════════════════════

// ─── 1D ─────────────────────────────────────────────────────────────────────

/**
 * @brief 1D fixture for FixedPressureOutflow BC tests.
 *
 * Same interior state as EtotFromPressureBC1D (γ=5/3, ρ=2, v=1, B=0.5,
 * P_interior=1, Etot_interior=4.875), but the pressure boundary uses a
 * Dirichlet condition with face value P_fixed=0.5. For this uniform state,
 * the ghost pressure becomes 2*P_fixed - P_interior = 0.
 */
struct FixedPressureOutflowBC1D : testing::Test
{
    static constexpr double gamma    = 5.0 / 3.0;
    static constexpr double rho_val  = 2.0;
    static constexpr double vx_val   = 1.0;
    static constexpr double vy_val   = 1.0;
    static constexpr double vz_val   = 1.0;
    static constexpr double Bx_val   = 0.5;
    static constexpr double By_val   = 0.5;
    static constexpr double Bz_val   = 0.5;
    static constexpr double P_val    = 1.0;
    static constexpr double P_fixed  = 0.5;
    static constexpr double sentinel = -999.0;

    static constexpr double u_specific_interior = P_val / (rho_val * (gamma - 1.0));
    static constexpr double etot_val
        = rho_val * u_specific_interior
          + 0.5 * rho_val * (vx_val * vx_val + vy_val * vy_val + vz_val * vz_val)
          + 0.5 * (Bx_val * Bx_val + By_val * By_val + Bz_val * Bz_val);

    static constexpr double P_ghost = 2.0 * P_fixed - P_val;
    static constexpr double u_specific_ghost = P_ghost / (rho_val * (gamma - 1.0));
    static constexpr double etot_ghost
        = rho_val * u_specific_ghost
          + 0.5 * rho_val * (vx_val * vx_val + vy_val * vy_val + vz_val * vz_val)
          + 0.5 * (Bx_val * Bx_val + By_val * By_val + Bz_val * Bz_val);

    GridLayoutMHD1D layout{{0.1}, {nCellsMHD}, {0.0}};

    GridMHD1D rhoGrid {"rho",  MHDQuantity::Scalar::rho,  layout.allocSize(MHDQuantity::Scalar::rho)};
    GridMHD1D PGrid   {"P",    MHDQuantity::Scalar::P,    layout.allocSize(MHDQuantity::Scalar::P)};
    GridMHD1D EtotGrid{"Etot", MHDQuantity::Scalar::Etot, layout.allocSize(MHDQuantity::Scalar::Etot)};

    UsableVecFieldMHD<1> rhoV{"rhoV", layout, MHDQuantity::Vector::rhoV};
    UsableVecFieldMHD<1> Bvec{"B",    layout, MHDQuantity::Vector::B};

    MHDPatchFieldAccessorTest<1> acc{rhoGrid, PGrid, EtotGrid, rhoV, Bvec};

    FieldMHD<1>& rhoField  {*(&rhoGrid)};
    FieldMHD<1>& PField    {*(&PGrid)};
    FieldMHD<1>& EtotField {*(&EtotGrid)};

    FixedPressureOutflowBC1D()
    {
        auto fill_scalar = [&](auto& f, MHDQuantity::Scalar qty, double interior_val) {
            for (std::uint32_t i = 0; i < f.shape()[0]; ++i)
                f(i) = sentinel;
            std::uint32_t ps = layout.physicalStartIndex(qty, Direction::X);
            std::uint32_t pe = layout.physicalEndIndex(qty, Direction::X);
            for (std::uint32_t i = ps; i <= pe; ++i)
                f(i) = interior_val;
        };

        fill_scalar(rhoField,  MHDQuantity::Scalar::rho,  rho_val);
        fill_scalar(PField,    MHDQuantity::Scalar::P,    P_val);
        fill_scalar(EtotField, MHDQuantity::Scalar::Etot, etot_val);

        fill_scalar(rhoV[0], MHDQuantity::Scalar::rhoVx, rho_val * vx_val);
        fill_scalar(rhoV[1], MHDQuantity::Scalar::rhoVy, rho_val * vy_val);
        fill_scalar(rhoV[2], MHDQuantity::Scalar::rhoVz, rho_val * vz_val);

        fill_scalar(Bvec[0], MHDQuantity::Scalar::Bx, Bx_val);
        fill_scalar(Bvec[1], MHDQuantity::Scalar::By, By_val);
        fill_scalar(Bvec[2], MHDQuantity::Scalar::Bz, Bz_val);
    }
};


/**
 * @brief Ghost Etot must match the value reconstructed from the fixed pressure
 * and the Neumann-extrapolated ρ, ρv, B (equal to interior values for a uniform state).
 */
TEST_F(FixedPressureOutflowBC1D, DirichletPressureGhostEtotMatchesExpected)
{
    auto rho_bc  = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<1>, GridLayoutMHD1D>>();
    auto P_bc    = std::make_shared<FieldDirichletBoundaryCondition<FieldMHD<1>, GridLayoutMHD1D>>(P_fixed);
    auto rhoV_bc = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<1>, GridLayoutMHD1D>>();
    auto B_bc    = std::make_shared<FieldDivergenceFreeTransverseNeumannBoundaryCondition<VecFieldMHD<1>, GridLayoutMHD1D>>();
    auto thermo  = std::make_shared<IdealGasThermo>(gamma);

    FieldTotalEnergyFromPressureBoundaryCondition<FieldMHD<1>, GridLayoutMHD1D> bc{
        rho_bc, rhoV_bc, B_bc, P_bc, thermo};

    bc.apply(EtotField, BoundaryLocation::XLower, mhdLowerGhostCellBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::XUpper, mhdUpperGhostCellBox(), layout, 0.0, acc);

    auto etotQty     = MHDQuantity::Scalar::Etot;
    std::uint32_t ps = layout.physicalStartIndex(etotQty, Direction::X);
    std::uint32_t pe = layout.physicalEndIndex(etotQty, Direction::X);

    for (std::uint32_t g = 0; g < mhdGhostWidth; ++g)
    {
        EXPECT_NEAR(EtotField(ps - 1u - g), etot_ghost, 1e-12) << "lower ghost g=" << g;
        EXPECT_NEAR(EtotField(pe + 1u + g), etot_ghost, 1e-12) << "upper ghost g=" << g;
    }
}

/**
 * @brief Interior Etot must remain unchanged after the FixedPressureOutflow BC is applied.
 */
TEST_F(FixedPressureOutflowBC1D, InteriorEtotUnchangedAfterBC)
{
    auto rho_bc  = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<1>, GridLayoutMHD1D>>();
    auto P_bc    = std::make_shared<FieldDirichletBoundaryCondition<FieldMHD<1>, GridLayoutMHD1D>>(P_fixed);
    auto rhoV_bc = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<1>, GridLayoutMHD1D>>();
    auto B_bc    = std::make_shared<FieldDivergenceFreeTransverseNeumannBoundaryCondition<VecFieldMHD<1>, GridLayoutMHD1D>>();
    auto thermo  = std::make_shared<IdealGasThermo>(gamma);

    FieldTotalEnergyFromPressureBoundaryCondition<FieldMHD<1>, GridLayoutMHD1D> bc{
        rho_bc, rhoV_bc, B_bc, P_bc, thermo};

    bc.apply(EtotField, BoundaryLocation::XLower, mhdLowerGhostCellBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::XUpper, mhdUpperGhostCellBox(), layout, 0.0, acc);

    auto etotQty     = MHDQuantity::Scalar::Etot;
    std::uint32_t ps = layout.physicalStartIndex(etotQty, Direction::X);
    std::uint32_t pe = layout.physicalEndIndex(etotQty, Direction::X);

    for (std::uint32_t i = ps; i <= pe; ++i)
        EXPECT_DOUBLE_EQ(EtotField(i), etot_val) << "interior index i=" << i;
}


// ─── 2D ─────────────────────────────────────────────────────────────────────

/**
 * @brief 2D fixture for FixedPressureOutflow BC tests.
 *
 * Same interior state as EtotFromPressureBC2D, with a Dirichlet pressure
 * boundary condition using face value P_fixed=0.5.
 */
struct FixedPressureOutflowBC2D : testing::Test
{
    static constexpr double gamma    = 5.0 / 3.0;
    static constexpr double rho_val  = 2.0;
    static constexpr double vx_val   = 1.0;
    static constexpr double vy_val   = 1.0;
    static constexpr double vz_val   = 1.0;
    static constexpr double Bx_val   = 0.5;
    static constexpr double By_val   = 0.5;
    static constexpr double Bz_val   = 0.5;
    static constexpr double P_val    = 1.0;
    static constexpr double P_fixed  = 0.5;
    static constexpr double sentinel = -999.0;

    static constexpr double u_specific_interior = P_val / (rho_val * (gamma - 1.0));
    static constexpr double etot_val
        = rho_val * u_specific_interior
          + 0.5 * rho_val * (vx_val * vx_val + vy_val * vy_val + vz_val * vz_val)
          + 0.5 * (Bx_val * Bx_val + By_val * By_val + Bz_val * Bz_val);

    static constexpr double P_ghost = 2.0 * P_fixed - P_val;
    static constexpr double u_specific_ghost = P_ghost / (rho_val * (gamma - 1.0));
    static constexpr double etot_ghost
        = rho_val * u_specific_ghost
          + 0.5 * rho_val * (vx_val * vx_val + vy_val * vy_val + vz_val * vz_val)
          + 0.5 * (Bx_val * Bx_val + By_val * By_val + Bz_val * Bz_val);

    GridLayoutMHD2D layout{{0.1, 0.1}, {nCellsMHDX2D, nCellsMHDY2D}, {0.0, 0.0}};

    GridMHD2D rhoGrid {"rho",  MHDQuantity::Scalar::rho,  layout.allocSize(MHDQuantity::Scalar::rho)};
    GridMHD2D PGrid   {"P",    MHDQuantity::Scalar::P,    layout.allocSize(MHDQuantity::Scalar::P)};
    GridMHD2D EtotGrid{"Etot", MHDQuantity::Scalar::Etot, layout.allocSize(MHDQuantity::Scalar::Etot)};

    UsableVecFieldMHD<2> rhoV{"rhoV", layout, MHDQuantity::Vector::rhoV};
    UsableVecFieldMHD<2> Bvec{"B",    layout, MHDQuantity::Vector::B};

    MHDPatchFieldAccessorTest<2> acc{rhoGrid, PGrid, EtotGrid, rhoV, Bvec};

    FieldMHD<2>& rhoField  {*(&rhoGrid)};
    FieldMHD<2>& PField    {*(&PGrid)};
    FieldMHD<2>& EtotField {*(&EtotGrid)};

    FixedPressureOutflowBC2D()
    {
        auto fill_scalar = [&](auto& f, MHDQuantity::Scalar qty, double interior_val) {
            auto [nx, ny] = f.shape();
            for (std::uint32_t ix = 0; ix < nx; ++ix)
                for (std::uint32_t iy = 0; iy < ny; ++iy)
                    f(ix, iy) = sentinel;
            std::uint32_t ps_x = layout.physicalStartIndex(qty, Direction::X);
            std::uint32_t pe_x = layout.physicalEndIndex(qty, Direction::X);
            std::uint32_t ps_y = layout.physicalStartIndex(qty, Direction::Y);
            std::uint32_t pe_y = layout.physicalEndIndex(qty, Direction::Y);
            for (std::uint32_t ix = ps_x; ix <= pe_x; ++ix)
                for (std::uint32_t iy = ps_y; iy <= pe_y; ++iy)
                    f(ix, iy) = interior_val;
        };

        fill_scalar(rhoField,  MHDQuantity::Scalar::rho,  rho_val);
        fill_scalar(PField,    MHDQuantity::Scalar::P,    P_val);
        fill_scalar(EtotField, MHDQuantity::Scalar::Etot, etot_val);

        fill_scalar(rhoV[0], MHDQuantity::Scalar::rhoVx, rho_val * vx_val);
        fill_scalar(rhoV[1], MHDQuantity::Scalar::rhoVy, rho_val * vy_val);
        fill_scalar(rhoV[2], MHDQuantity::Scalar::rhoVz, rho_val * vz_val);

        fill_scalar(Bvec[0], MHDQuantity::Scalar::Bx, Bx_val);
        fill_scalar(Bvec[1], MHDQuantity::Scalar::By, By_val);
        fill_scalar(Bvec[2], MHDQuantity::Scalar::Bz, Bz_val);
    }
};


/**
 * @brief Ghost Etot on all four 2D boundaries must match the value reconstructed
 * from the fixed pressure and the Neumann-extrapolated ρ, ρv, B.
 */
TEST_F(FixedPressureOutflowBC2D, DirichletPressureGhostEtotMatchesExpected)
{
    auto rho_bc  = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<2>, GridLayoutMHD2D>>();
    auto P_bc    = std::make_shared<FieldDirichletBoundaryCondition<FieldMHD<2>, GridLayoutMHD2D>>(P_fixed);
    auto rhoV_bc = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<2>, GridLayoutMHD2D>>();
    auto B_bc    = std::make_shared<FieldDivergenceFreeTransverseNeumannBoundaryCondition<VecFieldMHD<2>, GridLayoutMHD2D>>();
    auto thermo  = std::make_shared<IdealGasThermo>(gamma);

    FieldTotalEnergyFromPressureBoundaryCondition<FieldMHD<2>, GridLayoutMHD2D> bc{
        rho_bc, rhoV_bc, B_bc, P_bc, thermo};

    bc.apply(EtotField, BoundaryLocation::XLower, mhd2DXLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::XUpper, mhd2DXUpperGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::YLower, mhd2DYLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::YUpper, mhd2DYUpperGhostBox(), layout, 0.0, acc);

    auto etotQty     = MHDQuantity::Scalar::Etot;
    std::uint32_t ps_x = layout.physicalStartIndex(etotQty, Direction::X);
    std::uint32_t pe_x = layout.physicalEndIndex(etotQty, Direction::X);
    std::uint32_t ps_y = layout.physicalStartIndex(etotQty, Direction::Y);
    std::uint32_t pe_y = layout.physicalEndIndex(etotQty, Direction::Y);

    for (std::uint32_t g = 0; g < mhdGhostWidth; ++g)
    {
        for (std::uint32_t iy = ps_y; iy <= pe_y; ++iy)
        {
            EXPECT_NEAR(EtotField(ps_x - 1u - g, iy), etot_ghost, 1e-12)
                << "XLower ghost g=" << g << " iy=" << iy;
            EXPECT_NEAR(EtotField(pe_x + 1u + g, iy), etot_ghost, 1e-12)
                << "XUpper ghost g=" << g << " iy=" << iy;
        }
        for (std::uint32_t ix = ps_x; ix <= pe_x; ++ix)
        {
            EXPECT_NEAR(EtotField(ix, ps_y - 1u - g), etot_ghost, 1e-12)
                << "YLower ghost g=" << g << " ix=" << ix;
            EXPECT_NEAR(EtotField(ix, pe_y + 1u + g), etot_ghost, 1e-12)
                << "YUpper ghost g=" << g << " ix=" << ix;
        }
    }
}

/**
 * @brief Interior Etot must remain unchanged after the 2D FixedPressureOutflow BC is applied.
 */
TEST_F(FixedPressureOutflowBC2D, InteriorEtotUnchangedAfterBC)
{
    auto rho_bc  = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<2>, GridLayoutMHD2D>>();
    auto P_bc    = std::make_shared<FieldDirichletBoundaryCondition<FieldMHD<2>, GridLayoutMHD2D>>(P_fixed);
    auto rhoV_bc = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<2>, GridLayoutMHD2D>>();
    auto B_bc    = std::make_shared<FieldDivergenceFreeTransverseNeumannBoundaryCondition<VecFieldMHD<2>, GridLayoutMHD2D>>();
    auto thermo  = std::make_shared<IdealGasThermo>(gamma);

    FieldTotalEnergyFromPressureBoundaryCondition<FieldMHD<2>, GridLayoutMHD2D> bc{
        rho_bc, rhoV_bc, B_bc, P_bc, thermo};

    bc.apply(EtotField, BoundaryLocation::XLower, mhd2DXLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::XUpper, mhd2DXUpperGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::YLower, mhd2DYLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::YUpper, mhd2DYUpperGhostBox(), layout, 0.0, acc);

    auto etotQty       = MHDQuantity::Scalar::Etot;
    std::uint32_t ps_x = layout.physicalStartIndex(etotQty, Direction::X);
    std::uint32_t pe_x = layout.physicalEndIndex(etotQty, Direction::X);
    std::uint32_t ps_y = layout.physicalStartIndex(etotQty, Direction::Y);
    std::uint32_t pe_y = layout.physicalEndIndex(etotQty, Direction::Y);

    for (std::uint32_t ix = ps_x; ix <= pe_x; ++ix)
        for (std::uint32_t iy = ps_y; iy <= pe_y; ++iy)
            EXPECT_DOUBLE_EQ(EtotField(ix, iy), etot_val)
                << "interior index (" << ix << "," << iy << ")";
}


// ─── 3D ─────────────────────────────────────────────────────────────────────

/**
 * @brief 3D fixture for FixedPressureOutflow BC tests.
 *
 * Same interior state as EtotFromPressureBC3D, with a Dirichlet pressure
 * boundary condition using face value P_fixed=0.5.
 */
struct FixedPressureOutflowBC3D : testing::Test
{
    static constexpr double gamma    = 5.0 / 3.0;
    static constexpr double rho_val  = 2.0;
    static constexpr double vx_val   = 1.0;
    static constexpr double vy_val   = 1.0;
    static constexpr double vz_val   = 1.0;
    static constexpr double Bx_val   = 0.5;
    static constexpr double By_val   = 0.5;
    static constexpr double Bz_val   = 0.5;
    static constexpr double P_val    = 1.0;
    static constexpr double P_fixed  = 0.5;
    static constexpr double sentinel = -999.0;

    static constexpr double u_specific_interior = P_val / (rho_val * (gamma - 1.0));
    static constexpr double etot_val
        = rho_val * u_specific_interior
          + 0.5 * rho_val * (vx_val * vx_val + vy_val * vy_val + vz_val * vz_val)
          + 0.5 * (Bx_val * Bx_val + By_val * By_val + Bz_val * Bz_val);

    static constexpr double P_ghost = 2.0 * P_fixed - P_val;
    static constexpr double u_specific_ghost = P_ghost / (rho_val * (gamma - 1.0));
    static constexpr double etot_ghost
        = rho_val * u_specific_ghost
          + 0.5 * rho_val * (vx_val * vx_val + vy_val * vy_val + vz_val * vz_val)
          + 0.5 * (Bx_val * Bx_val + By_val * By_val + Bz_val * Bz_val);

    GridLayoutMHD3D layout{{0.1, 0.1, 0.1}, {nCellsMHDX3D, nCellsMHDY3D, nCellsMHDZ3D}, {0.0, 0.0, 0.0}};

    GridMHD3D rhoGrid {"rho",  MHDQuantity::Scalar::rho,  layout.allocSize(MHDQuantity::Scalar::rho)};
    GridMHD3D PGrid   {"P",    MHDQuantity::Scalar::P,    layout.allocSize(MHDQuantity::Scalar::P)};
    GridMHD3D EtotGrid{"Etot", MHDQuantity::Scalar::Etot, layout.allocSize(MHDQuantity::Scalar::Etot)};

    UsableVecFieldMHD<3> rhoV{"rhoV", layout, MHDQuantity::Vector::rhoV};
    UsableVecFieldMHD<3> Bvec{"B",    layout, MHDQuantity::Vector::B};

    MHDPatchFieldAccessorTest<3> acc{rhoGrid, PGrid, EtotGrid, rhoV, Bvec};

    FieldMHD<3>& rhoField  {*(&rhoGrid)};
    FieldMHD<3>& PField    {*(&PGrid)};
    FieldMHD<3>& EtotField {*(&EtotGrid)};

    FixedPressureOutflowBC3D()
    {
        auto fill_scalar = [&](auto& f, MHDQuantity::Scalar qty, double interior_val) {
            auto [nx, ny, nz] = f.shape();
            for (std::uint32_t ix = 0; ix < nx; ++ix)
                for (std::uint32_t iy = 0; iy < ny; ++iy)
                    for (std::uint32_t iz = 0; iz < nz; ++iz)
                        f(ix, iy, iz) = sentinel;
            std::uint32_t ps_x = layout.physicalStartIndex(qty, Direction::X);
            std::uint32_t pe_x = layout.physicalEndIndex(qty, Direction::X);
            std::uint32_t ps_y = layout.physicalStartIndex(qty, Direction::Y);
            std::uint32_t pe_y = layout.physicalEndIndex(qty, Direction::Y);
            std::uint32_t ps_z = layout.physicalStartIndex(qty, Direction::Z);
            std::uint32_t pe_z = layout.physicalEndIndex(qty, Direction::Z);
            for (std::uint32_t ix = ps_x; ix <= pe_x; ++ix)
                for (std::uint32_t iy = ps_y; iy <= pe_y; ++iy)
                    for (std::uint32_t iz = ps_z; iz <= pe_z; ++iz)
                        f(ix, iy, iz) = interior_val;
        };

        fill_scalar(rhoField,  MHDQuantity::Scalar::rho,  rho_val);
        fill_scalar(PField,    MHDQuantity::Scalar::P,    P_val);
        fill_scalar(EtotField, MHDQuantity::Scalar::Etot, etot_val);

        fill_scalar(rhoV[0], MHDQuantity::Scalar::rhoVx, rho_val * vx_val);
        fill_scalar(rhoV[1], MHDQuantity::Scalar::rhoVy, rho_val * vy_val);
        fill_scalar(rhoV[2], MHDQuantity::Scalar::rhoVz, rho_val * vz_val);

        fill_scalar(Bvec[0], MHDQuantity::Scalar::Bx, Bx_val);
        fill_scalar(Bvec[1], MHDQuantity::Scalar::By, By_val);
        fill_scalar(Bvec[2], MHDQuantity::Scalar::Bz, Bz_val);
    }
};


/**
 * @brief Ghost Etot on all six 3D boundaries must match the value reconstructed
 * from the fixed pressure and the Neumann-extrapolated ρ, ρv, B.
 */
TEST_F(FixedPressureOutflowBC3D, DirichletPressureGhostEtotMatchesExpected)
{
    auto rho_bc  = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<3>, GridLayoutMHD3D>>();
    auto P_bc    = std::make_shared<FieldDirichletBoundaryCondition<FieldMHD<3>, GridLayoutMHD3D>>(P_fixed);
    auto rhoV_bc = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<3>, GridLayoutMHD3D>>();
    auto B_bc    = std::make_shared<FieldDivergenceFreeTransverseNeumannBoundaryCondition<VecFieldMHD<3>, GridLayoutMHD3D>>();
    auto thermo  = std::make_shared<IdealGasThermo>(gamma);

    FieldTotalEnergyFromPressureBoundaryCondition<FieldMHD<3>, GridLayoutMHD3D> bc{
        rho_bc, rhoV_bc, B_bc, P_bc, thermo};

    bc.apply(EtotField, BoundaryLocation::XLower, mhd3DXLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::XUpper, mhd3DXUpperGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::YLower, mhd3DYLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::YUpper, mhd3DYUpperGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::ZLower, mhd3DZLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::ZUpper, mhd3DZUpperGhostBox(), layout, 0.0, acc);

    auto etotQty       = MHDQuantity::Scalar::Etot;
    std::uint32_t ps_x = layout.physicalStartIndex(etotQty, Direction::X);
    std::uint32_t pe_x = layout.physicalEndIndex(etotQty, Direction::X);
    std::uint32_t ps_y = layout.physicalStartIndex(etotQty, Direction::Y);
    std::uint32_t pe_y = layout.physicalEndIndex(etotQty, Direction::Y);
    std::uint32_t ps_z = layout.physicalStartIndex(etotQty, Direction::Z);
    std::uint32_t pe_z = layout.physicalEndIndex(etotQty, Direction::Z);

    for (std::uint32_t g = 0; g < mhdGhostWidth; ++g)
    {
        for (std::uint32_t iy = ps_y; iy <= pe_y; ++iy)
            for (std::uint32_t iz = ps_z; iz <= pe_z; ++iz)
            {
                EXPECT_NEAR(EtotField(ps_x - 1u - g, iy, iz), etot_ghost, 1e-12)
                    << "XLower g=" << g;
                EXPECT_NEAR(EtotField(pe_x + 1u + g, iy, iz), etot_ghost, 1e-12)
                    << "XUpper g=" << g;
            }
        for (std::uint32_t ix = ps_x; ix <= pe_x; ++ix)
            for (std::uint32_t iz = ps_z; iz <= pe_z; ++iz)
            {
                EXPECT_NEAR(EtotField(ix, ps_y - 1u - g, iz), etot_ghost, 1e-12)
                    << "YLower g=" << g;
                EXPECT_NEAR(EtotField(ix, pe_y + 1u + g, iz), etot_ghost, 1e-12)
                    << "YUpper g=" << g;
            }
        for (std::uint32_t ix = ps_x; ix <= pe_x; ++ix)
            for (std::uint32_t iy = ps_y; iy <= pe_y; ++iy)
            {
                EXPECT_NEAR(EtotField(ix, iy, ps_z - 1u - g), etot_ghost, 1e-12)
                    << "ZLower g=" << g;
                EXPECT_NEAR(EtotField(ix, iy, pe_z + 1u + g), etot_ghost, 1e-12)
                    << "ZUpper g=" << g;
            }
    }
}

/**
 * @brief Interior Etot must remain unchanged after the 3D FixedPressureOutflow BC is applied.
 */
TEST_F(FixedPressureOutflowBC3D, InteriorEtotUnchangedAfterBC)
{
    auto rho_bc  = std::make_shared<FieldNeumannBoundaryCondition<FieldMHD<3>, GridLayoutMHD3D>>();
    auto P_bc    = std::make_shared<FieldDirichletBoundaryCondition<FieldMHD<3>, GridLayoutMHD3D>>(P_fixed);
    auto rhoV_bc = std::make_shared<FieldNeumannBoundaryCondition<VecFieldMHD<3>, GridLayoutMHD3D>>();
    auto B_bc    = std::make_shared<FieldDivergenceFreeTransverseNeumannBoundaryCondition<VecFieldMHD<3>, GridLayoutMHD3D>>();
    auto thermo  = std::make_shared<IdealGasThermo>(gamma);

    FieldTotalEnergyFromPressureBoundaryCondition<FieldMHD<3>, GridLayoutMHD3D> bc{
        rho_bc, rhoV_bc, B_bc, P_bc, thermo};

    bc.apply(EtotField, BoundaryLocation::XLower, mhd3DXLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::XUpper, mhd3DXUpperGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::YLower, mhd3DYLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::YUpper, mhd3DYUpperGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::ZLower, mhd3DZLowerGhostBox(), layout, 0.0, acc);
    bc.apply(EtotField, BoundaryLocation::ZUpper, mhd3DZUpperGhostBox(), layout, 0.0, acc);

    auto etotQty       = MHDQuantity::Scalar::Etot;
    std::uint32_t ps_x = layout.physicalStartIndex(etotQty, Direction::X);
    std::uint32_t pe_x = layout.physicalEndIndex(etotQty, Direction::X);
    std::uint32_t ps_y = layout.physicalStartIndex(etotQty, Direction::Y);
    std::uint32_t pe_y = layout.physicalEndIndex(etotQty, Direction::Y);
    std::uint32_t ps_z = layout.physicalStartIndex(etotQty, Direction::Z);
    std::uint32_t pe_z = layout.physicalEndIndex(etotQty, Direction::Z);

    for (std::uint32_t ix = ps_x; ix <= pe_x; ++ix)
        for (std::uint32_t iy = ps_y; iy <= pe_y; ++iy)
            for (std::uint32_t iz = ps_z; iz <= pe_z; ++iz)
                EXPECT_DOUBLE_EQ(EtotField(ix, iy, iz), etot_val)
                    << "interior index (" << ix << "," << iy << "," << iz << ")";
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
