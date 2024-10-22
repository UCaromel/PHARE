#include <amr/physical_models/physical_model.hpp>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

#include "amr/types/amr_types.hpp"
#include "core/mhd/mhd_quantities.hpp"
#include "amr/messengers/messenger.hpp"


#include "amr/wrappers/hierarchy.hpp"
#include "core/data/grid/gridlayoutimplyee_mhd.hpp"
#include "core/numerics/boundary_condition/boundary_condition.hpp"
#include "phare_core.hpp"
#include "tests/core/data/gridlayout/test_gridlayout.hpp"
#include "tests/core/data/mhd_state/test_mhd_state_fixtures.hpp"
#include "amr/solvers/solver_mhd.hpp"
#include "amr/solvers/solver.hpp"

constexpr std::uint32_t cells = 65;
constexpr std::size_t dim = 1, interp = 1;

using YeeLayout_t       = PHARE::core::GridLayoutImplYeeMHD<dim, interp>;
using GridLayoutMHD     = PHARE::core::GridLayout<YeeLayout_t>;
using GridLayout_t      = TestGridLayout<GridLayoutMHD>;
using BoundaryCondition = PHARE::core::BoundaryCondition<dim, interp>;

using DummyModelView = PHARE::core::UsableMHDState<dim>;


template<std::size_t dim>
class UsableMHDStateWrapper : public PHARE::solver::ISolverModelView
{
public:
    UsableMHDStateWrapper(PHARE::core::UsableMHDState<dim>& mhdState)
        : mhdState_(mhdState)
    {
    }

private:
    PHARE::core::UsableMHDState<dim>& mhdState_; // Reference to the original state
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


struct DummyMHDModel : public PHARE::solver::IPhysicalModel<DummyTypes>
{
    using field_type                       = typename PHARE::core::UsableMHDState<dim>::Grid_t;
    static constexpr std::size_t dimension = dim;
    static constexpr auto model_name       = "mhd_model";
};


class DummyMessenger : public PHARE::amr::IMessenger<PHARE::solver::IPhysicalModel<DummyTypes>>
{
    using IPhysicalModel = PHARE::solver::IPhysicalModel<DummyTypes>;

    std::string name() override { return "DummyMessenger"; }

    std::unique_ptr<PHARE::amr::IMessengerInfo> emptyInfoFromCoarser() override
    {
        return std::make_unique<PHARE::amr::IMessengerInfo>();
    }

    std::unique_ptr<PHARE::amr::IMessengerInfo> emptyInfoFromFiner() override
    {
        return std::make_unique<PHARE::amr::IMessengerInfo>();
    }

    void allocate(SAMRAI::hier::Patch& patch, double const allocateTime) const override {}

    void registerQuantities(std::unique_ptr<PHARE::amr::IMessengerInfo> fromCoarserInfo,
                            std::unique_ptr<PHARE::amr::IMessengerInfo> fromFinerInfo) override
    {
    }

    void registerLevel(const std::shared_ptr<SAMRAI::hier::PatchHierarchy>& hierarchy,
                       int level) override
    {
    }

    void regrid(std::shared_ptr<SAMRAI::hier::PatchHierarchy> const& hierarchy,
                int const levelNumber, std::shared_ptr<SAMRAI::hier::PatchLevel> const& oldLevel,
                IPhysicalModel& model, double const initDataTime) override
    {
    }

    void initLevel(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                   double const initDataTime) override
    {
    }

    void firstStep(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                   const std::shared_ptr<SAMRAI::hier::PatchHierarchy>& hierarchy,
                   double const currentTime, double const prevCoarserTime,
                   double const newCoarserTime) override
    {
    }

    void lastStep(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level) override {}

    void prepareStep(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                     double currentTime) override
    {
    }

    void fillRootGhosts(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                        double const initDataTime) override
    {
    }

    void synchronize(SAMRAI::hier::PatchLevel& level) override {}

    void postSynchronize(IPhysicalModel& model, SAMRAI::hier::PatchLevel& level,
                         double const time) override
    {
    }

    std::string fineModelName() const override { return "DummyMHDModel"; }
    std::string coarseModelName() const override { return "DummyMHDModel"; }
};


TEST(UsableMHDStateTest, ConstructionTest)
{
    GridLayout_t layout{cells};
    DummyModelView state(layout);
    PHARE::solver::SolverMHD<DummyMHDModel, DummyTypes, DummyMessenger, DummyModelView>
        TestMHDSolver;

    UsableMHDStateWrapper<dim> dummy_view(state);

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
