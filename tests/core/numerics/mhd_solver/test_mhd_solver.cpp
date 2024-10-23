#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

#include "amr/types/amr_types.hpp"
#include "core/mhd/mhd_quantities.hpp"

#include "amr/wrappers/hierarchy.hpp"
#include "core/data/grid/gridlayoutimplyee_mhd.hpp"
#include "core/numerics/boundary_condition/boundary_condition.hpp"
#include "phare_core.hpp"
#include "tests/core/data/gridlayout/test_gridlayout.hpp"
#include "tests/core/data/mhd_state/test_mhd_state_fixtures.hpp"
#include "amr/solvers/solver_mhd.hpp"

constexpr std::uint32_t cells = 65;
constexpr std::size_t dim = 1, interp = 1;

using YeeLayout_t       = PHARE::core::GridLayoutImplYeeMHD<dim, interp>;
using GridLayoutMHD     = PHARE::core::GridLayout<YeeLayout_t>;
using GridLayout_t      = TestGridLayout<GridLayoutMHD>;
using BoundaryCondition = PHARE::core::BoundaryCondition<dim, interp>;

using DummyModelView = PHARE::core::UsableMHDState<dim>;


struct DummyMHDModel
{
    using field_type                       = typename PHARE::core::UsableMHDState<dim>::Grid_t;
    static constexpr std::size_t dimension = dim;
    static constexpr auto model_name       = "mhd_model";
};

struct DummyMessenger
{
};

struct DummyHierarchy
{
    auto getPatchLevel(std::size_t lvl) const
    {
        int* a = nullptr;
        *a     = 1;
        return a;
    };
};

struct DummyTypes
{
    using patch_t     = PHARE::amr::SAMRAI_Types::patch_t;
    using level_t     = int;
    using hierarchy_t = DummyHierarchy;
};


TEST(UsableMHDStateTest, ConstructionTest)
{
    GridLayout_t layout{cells};
    DummyModelView dummy_view(layout);
    PHARE::solver::SolverMHD<DummyMHDModel, DummyTypes, DummyMessenger, DummyModelView>
        TestMHDSolver;


    DummyMessenger dummy_messenger;

    DummyHierarchy dummy_hierachy;

    TestMHDSolver.advanceLevel(dummy_hierachy, 1, dummy_view, dummy_messenger, 0.0, 0.01);
    ASSERT_NO_THROW();
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
