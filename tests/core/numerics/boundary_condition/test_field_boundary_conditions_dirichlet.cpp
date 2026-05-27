#include "gtest/gtest.h"

#include "core/numerics/boundary_condition/field_dirichlet_boundary_condition.hpp"
#include "tests/core/numerics/boundary_condition/hybrid_bc_test_fixtures.hpp"

using namespace PHARE::core;


TEST_F(FieldBC1D, DirichletSetsLowerGhostByLinearExtrapolation)
{
    double const value = 3.0;
    FieldDirichletBoundaryCondition<Field1D, GridLayout1D> bc{value};
    bc.apply(field, BoundaryLocation::XLower, lowerGhostCellBox(), layout, makeCtx(acc, 0.0));

    // Interior is constant = interiorValue, so ghost = 2*value - interiorValue
    double expected = 2.0 * value - interiorValue;
    for (std::uint32_t g = 0; g < ghostWidth; ++g)
        EXPECT_DOUBLE_EQ(field(g), expected);
}

TEST_F(FieldBC1D, DirichletSetsUpperGhostByLinearExtrapolation)
{
    double const value = 3.0;
    FieldDirichletBoundaryCondition<Field1D, GridLayout1D> bc{value};
    bc.apply(field, BoundaryLocation::XUpper, upperGhostCellBox(), layout, makeCtx(acc, 0.0));

    double expected       = 2.0 * value - interiorValue;
    std::uint32_t allocSz = grid.shape()[0];
    for (std::uint32_t g = 0; g < ghostWidth; ++g)
        EXPECT_DOUBLE_EQ(field(allocSz - 1 - g), expected);
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
