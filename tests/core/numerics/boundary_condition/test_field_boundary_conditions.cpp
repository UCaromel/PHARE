#include "gtest/gtest.h"

#include "core/boundary/boundary_defs.hpp"
#include "core/data/grid/grid.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayoutimplyee.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/data/patch_field_accessor.hpp"
#include "core/numerics/boundary_condition/field_antisymmetric_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_dirichlet_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_neumann_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_none_boundary_condition.hpp"
#include "core/numerics/boundary_condition/field_symmetric_boundary_condition.hpp"
#include "core/utilities/box/box.hpp"
#include "tests/core/data/tensorfield/test_tensorfield_fixtures.hpp"

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


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
