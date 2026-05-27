#include "gtest/gtest.h"

// TODO: add direct tests for FieldDivergenceFreeTransverseNeumannBoundaryCondition.
// Currently exercised only indirectly via test_field_boundary_conditions_total_energy_from_pressure.

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
