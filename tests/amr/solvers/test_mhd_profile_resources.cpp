#include "phare_core.hpp"

#include "amr/messenger_registration.hpp"
#include "amr/messengers/mhd_messenger.hpp"
#include "amr/resources_manager/amr_utils.hpp"
#include "amr/resources_manager/resources_manager.hpp"
#include "amr/solvers/mhd_resolver.hpp"
#include "amr/solvers/solver_mhd.hpp"
#include "amr/solvers/time_integrator/point_value_approximation.hpp"
#include "core/numerics/constrained_transport/upwind_constrained_transport_utils.hpp"
#include "tests/amr/solvers/convergence/hall_convergence_test_common.hpp"
#include "tests/amr/solvers/periodic_hierarchy.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace
{
constexpr PHARE::SimOpts O2Opts{3,
                                0,
                                0,
                                PHARE::MHDOpts::MHDOrder::O2,
                                PHARE::MHDOpts::ReconstructionType::WENOZ,
                                PHARE::MHDOpts::SlopeLimiterType::None,
                                PHARE::MHDOpts::RiemannSolverType::Rusanov};
constexpr PHARE::SimOpts O4Opts{3,
                                0,
                                0,
                                PHARE::MHDOpts::MHDOrder::O4,
                                PHARE::MHDOpts::ReconstructionType::WENOZ,
                                PHARE::MHDOpts::SlopeLimiterType::None,
                                PHARE::MHDOpts::RiemannSolverType::Rusanov};

template<auto opts>
struct MHDTestTypes
{
    using CoreTypes = PHARE::core::PHARE_Types<opts>;
    using Layout    = typename CoreTypes::MHD::GridLayout_t;
    using Grid      = typename CoreTypes::MHD::Grid_t;
    using VecField  = typename CoreTypes::MHD::VecField_t;
    using Model     = PHARE::solver::MHDModel<Layout, VecField, PHARE::amr::SAMRAI_Types, Grid>;
    using ResourcesManager = PHARE::amr::ResourcesManager<Layout, Grid>;
    // resolved as the solver does, so a wrong O4 selection shows up as the wrong resource set
    using Approx = typename PHARE::solver::MHDResolver<opts, Model>::PointValueApproximation;
};

TEST(MHDProfileResources, O2HasNoPointValueResources)
{
    using Types  = MHDTestTypes<O2Opts>;
    using Approx = Types::Approx;

    static_assert(
        std::tuple_size_v<decltype(std::declval<Approx const&>().getCompileTimeResourcesViewList())>
        == 0);

    auto rm = std::make_shared<typename Types::ResourcesManager>();
    Approx approximation;
    rm->registerResources(approximation);
    EXPECT_FALSE(rm->getID("point_value_rho").has_value());
}

TEST(MHDProfileResources, O4AllocatesPointValueResources)
{
    using Types  = MHDTestTypes<O4Opts>;
    using Approx = Types::Approx;

    auto hierarchy = makePeriodicHierarchy3D(8);
    auto level     = hierarchy->getPatchLevel(0);
    auto rm        = std::make_shared<typename Types::ResourcesManager>();
    Approx approximation;

    rm->registerResources(approximation);
    for (auto& patch : *level)
        rm->allocate(approximation, *patch, 0.0);

    // PointValueState publishes one key per member, TensorField members included: the keys are
    // the names given in point_value_state.hpp, not per-component names.
    auto const pointIDs = rm->getIDsList(
        std::string{"point_value_rho"}, std::string{"point_value_V"}, std::string{"point_value_B"},
        std::string{"point_value_P"}, std::string{"point_value_J"}, std::string{"point_value_rhoV"},
        std::string{"point_value_Etot"});
    EXPECT_EQ(std::set<int>(pointIDs.begin(), pointIDs.end()).size(), pointIDs.size());
    for (auto& patch : *level)
        for (auto const id : pointIDs)
            EXPECT_TRUE(patch->checkAllocated(id));

    EXPECT_FALSE(rm->getID("point_value_rhoV_x").has_value());
    EXPECT_FALSE(rm->getID("point_value_B_x").has_value());
}
// The transverse-current buffers (j_t_*, rho_t_*) are written by Godunov only for Hall schemes:
// resistive and hyper-resistive EMF terms live in the separate DissipativeElectricField at edges.
TEST(MHDCTStateResources, TransverseCurrentIsOwnedByHallStatesOnly)
{
    using Types    = MHDTestTypes<O2Opts>;
    using VecField = Types::VecField;
    using State    = PHARE::core::UpwindConstrainedTransportState<VecField>;

    State hallState{true};
    State idealState{false};
    EXPECT_EQ(hallState.getRunTimeResourcesViewList().size(), 1u);
    EXPECT_EQ(idealState.getRunTimeResourcesViewList().size(), 0u);

    auto rm = std::make_shared<typename Types::ResourcesManager>();
    rm->registerResources(idealState);
    EXPECT_FALSE(rm->getID("j_t_x").has_value());
    EXPECT_FALSE(rm->getID("rho_t_x").has_value());
    rm->registerResources(hallState);
    EXPECT_TRUE(rm->getID("j_t_x").has_value());
    EXPECT_TRUE(rm->getID("rho_t_x").has_value());
}

// ExactHall3D has period 1 and the fixture domain is the unit cube, so the profile value at a
// ghost node's coordinate is the value its periodic image carries. The fill NaNs the ghost layer
// first: a node no transaction wrote shows up as non-finite.
template<typename Layout, typename Field, typename Fn>
void expectPeriodicGhostValues(Layout const& layout, Field& field, Fn&& fn, std::string const& name,
                               double const tol)
{
    auto const cent = layout.centering(field.physicalQuantity());

    std::size_t nonFinite  = 0;
    std::size_t mismatched = 0;
    double maxErr          = 0.0;

    for (auto i = layout.ghostStartIndex(cent[0], Direction::X);
         i <= layout.ghostEndIndex(cent[0], Direction::X); ++i)
        for (auto j = layout.ghostStartIndex(cent[1], Direction::Y);
             j <= layout.ghostEndIndex(cent[1], Direction::Y); ++j)
            for (auto kk = layout.ghostStartIndex(cent[2], Direction::Z);
                 kk <= layout.ghostEndIndex(cent[2], Direction::Z); ++kk)
            {
                auto const v = field(i, j, kk);
                if (!std::isfinite(v))
                {
                    ++nonFinite;
                    continue;
                }
                auto const c = layout.fieldNodeCoordinates(
                    field, layout.localToAMR(Point{i, j, kk}.as_signed()));
                auto const err = std::abs(v - fn(c[0], c[1], c[2]));
                maxErr         = std::max(maxErr, err);
                if (err > tol)
                    ++mismatched;
            }

    EXPECT_EQ(nonFinite, std::size_t{0}) << name << ": unfilled ghost nodes";
    EXPECT_EQ(mismatched, std::size_t{0}) << name << ": max error " << maxErr;
}

// Root level of a periodic single-level hierarchy: the ghost schedules, whose time refiners read
// the solver's old state, must fill the whole ghost layer from the level's periodic images.
template<auto opts>
void expectRootConservativeGhostsFilledPeriodically(int const nCells)
{
    using Types     = MHDTestTypes<opts>;
    using Model     = typename Types::Model;
    using Layout    = typename Types::Layout;
    using Messenger = PHARE::amr::MHDMessenger<Model>;
    using Stepper   = typename PHARE::solver::MHDResolver<opts, Model>::MHDTimeStepper_t;
    using Solver    = PHARE::solver::SolverMHD<Model, PHARE::amr::SAMRAI_Types, Stepper>;

    SCOPED_TRACE("nCells=" + std::to_string(nCells));

    auto hierarchy = makePeriodicHierarchy3D(nCells);
    auto level     = hierarchy->getPatchLevel(0);
    auto rm        = std::make_shared<typename Types::ResourcesManager>();
    Model model{makeHall3DMHDModelDict(), rm};
    Solver solver{makeHall3DComputeFluxDict()};
    Messenger messenger{rm, 0};

    rm->registerResources(model.state);
    solver.registerResources(model);
    PHARE::solver::MessengerRegistration::registerQuantities<PHARE::amr::SAMRAI_Types>(
        messenger, model, model, solver);
    messenger.registerLevel(hierarchy, 0);
    for (auto& patch : *level)
    {
        model.allocate(*patch, 0.0);
        solver.allocate(model, *patch, 0.0);
        messenger.allocate(*patch, 0.0);
    }

    auto const rhoVx = [](double x, double y, double z) {
        return ExactHall3D::rho(x, y, z) * ExactHall3D::vx(x, y, z);
    };
    auto const rhoVy = [](double x, double y, double z) {
        return ExactHall3D::rho(x, y, z) * ExactHall3D::vy(x, y, z);
    };
    auto const rhoVz = [](double x, double y, double z) {
        return ExactHall3D::rho(x, y, z) * ExactHall3D::vz(x, y, z);
    };

    for (auto& patch : *level)
    {
        auto guard        = rm->setOnPatch(*patch, model.state);
        auto const layout = PHARE::amr::layoutFromPatch<Layout>(*patch);
        fillUsableField(layout, model.state.rho, ExactHall3D::rho);
        fillUsableField(layout, model.state.Etot, ExactHall3D::etot);
        fillUsableField(layout, model.state.rhoV(Component::X), rhoVx);
        fillUsableField(layout, model.state.rhoV(Component::Y), rhoVy);
        fillUsableField(layout, model.state.rhoV(Component::Z), rhoVz);
        fillUsableField(layout, model.state.B(Component::X), ExactHall3D::bx);
        fillUsableField(layout, model.state.B(Component::Y), ExactHall3D::by);
        fillUsableField(layout, model.state.B(Component::Z), ExactHall3D::bz);
    }

    messenger.fillMagneticGhosts(model.state.B, *level, 0.0);
    messenger.fillMomentsGhosts(model.state, *level, 0.0);

    constexpr double tol = 1.e-12;
    for (auto& patch : *level)
    {
        auto guard        = rm->setOnPatch(*patch, model.state);
        auto const layout = PHARE::amr::layoutFromPatch<Layout>(*patch);
        expectPeriodicGhostValues(layout, model.state.rho, ExactHall3D::rho, "rho", tol);
        expectPeriodicGhostValues(layout, model.state.Etot, ExactHall3D::etot, "Etot", tol);
        expectPeriodicGhostValues(layout, model.state.rhoV(Component::X), rhoVx, "rhoV_x", tol);
        expectPeriodicGhostValues(layout, model.state.rhoV(Component::Y), rhoVy, "rhoV_y", tol);
        expectPeriodicGhostValues(layout, model.state.rhoV(Component::Z), rhoVz, "rhoV_z", tol);
        expectPeriodicGhostValues(layout, model.state.B(Component::X), ExactHall3D::bx, "B_x", tol);
        expectPeriodicGhostValues(layout, model.state.B(Component::Y), ExactHall3D::by, "B_y", tol);
        expectPeriodicGhostValues(layout, model.state.B(Component::Z), ExactHall3D::bz, "B_z", tol);
    }
}

TEST(MHDRootGhostFill, O2FillsEveryConservativeGhostFromPeriodicImages)
{
    expectRootConservativeGhostsFilledPeriodically<O2Opts>(8);
    expectRootConservativeGhostsFilledPeriodically<O2Opts>(32);
}
} // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    SAMRAI::tbox::SAMRAI_MPI::init(&argc, &argv);
    SAMRAI::tbox::SAMRAIManager::initialize();
    SAMRAI::tbox::SAMRAIManager::startup();

    int const testResult = RUN_ALL_TESTS();

    SAMRAI::tbox::SAMRAIManager::shutdown();
    SAMRAI::tbox::SAMRAIManager::finalize();
    SAMRAI::tbox::SAMRAI_MPI::finalize();
    return testResult;
}
