#include "phare_core.hpp"

#include "amr/messenger_registration.hpp"
#include "amr/messengers/mc2011_temporal_transfer.hpp"
#include "amr/messengers/mhd_messenger.hpp"
#include "amr/messengers/mhd_temporal_transfer.hpp"
#include "amr/resources_manager/amr_utils.hpp"
#include "amr/resources_manager/resources_manager.hpp"
#include "amr/solvers/mhd_resolver.hpp"
#include "amr/solvers/solver_mhd.hpp"
#include "amr/solvers/time_integrator/point_value_approximation.hpp"
#include "core/models/mhd_conserved_state.hpp"
#include "core/numerics/constrained_transport/upwind_constrained_transport_utils.hpp"
#include "core/numerics/rk_stage_context.hpp"
#include "simulator/simulator.hpp"
#include "tests/amr/solvers/convergence/hall_convergence_test_common.hpp"
#include "tests/amr/solvers/periodic_hierarchy.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <set>
#include <stdexcept>
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

inline PHARE::initializer::PHAREDict
makeSolverDict(PHARE::MHDOpts::TimeIntegratorType const integrator)
{
    auto dict                    = makeHall3DComputeFluxDict();
    dict["time_integrator_type"] = static_cast<int>(integrator);
    return dict;
}

// A model, solver and messenger registered on the root level of a periodic single-level
// hierarchy, as the simulator wires them.
template<auto opts, typename Transfer>
struct SolverFixture
{
    using Types     = MHDTestTypes<opts>;
    using Model     = typename Types::Model;
    using Messenger = PHARE::amr::MHDMessenger<Model, Transfer>;
    using Stepper   = typename PHARE::solver::MHDResolver<opts, Model, Messenger>::MHDTimeStepper_t;
    using Solver    = PHARE::solver::SolverMHD<Model, PHARE::amr::SAMRAI_Types, Stepper>;

    SolverFixture(int const nCells, PHARE::MHDOpts::TimeIntegratorType const integrator)
        : hierarchy{makePeriodicHierarchy3D(nCells)}
        , level{hierarchy->getPatchLevel(0)}
        , rm{std::make_shared<typename Types::ResourcesManager>()}
        , model{makeHall3DMHDModelDict(), rm}
        , solver{makeSolverDict(integrator)}
        , messenger{rm, 0}
    {
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
    }

    std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy;
    std::shared_ptr<SAMRAI::hier::PatchLevel> level;
    std::shared_ptr<typename Types::ResourcesManager> rm;
    Model model;
    Solver solver;
    Messenger messenger;
};

TEST(MHDProfileResources, O2HasNoPointValueOrTemporalTransferResources)
{
    using Types    = MHDTestTypes<O2Opts>;
    using Approx   = Types::Approx;
    using Transfer = PHARE::amr::Linear2TemporalTransfer<Types::Model>;

    static_assert(
        std::tuple_size_v<decltype(std::declval<Approx const&>().getCompileTimeResourcesViewList())>
        == 0);
    static_assert(std::tuple_size_v<
                      decltype(std::declval<Transfer const&>().getCompileTimeResourcesViewList())>
                  == 0);

    SolverFixture<O2Opts, Transfer> f{8, PHARE::MHDOpts::TimeIntegratorType::TVDRK3};
    Approx approximation;
    f.rm->registerResources(approximation);
    for (auto& patch : *f.level)
        f.rm->allocate(approximation, *patch, 0.0);

    EXPECT_FALSE(f.rm->getID("point_value_rho").has_value());
    EXPECT_FALSE(f.rm->getID("mc2011_assembled_rho").has_value());

    // the solver's old state publishes one key per conserved quantity: rhoV and B are
    // TensorFields, registered under their own name, not per component
    auto const names = PHARE::core::MHDStateNames{"MHDSolver_stateOld"};
    auto const ids   = f.rm->getIDsList(names.rho(), names.rhoV(), names.Etot(), names.B());
    auto const uniqueIDs
        = std::set<int>{std::get<0>(ids), std::get<1>(ids), std::get<2>(ids), std::get<3>(ids)};
    EXPECT_EQ(uniqueIDs.size(), 4u);
    for (auto& patch : *f.level)
        for (auto const id : uniqueIDs)
            EXPECT_TRUE(patch->checkAllocated(id));
    EXPECT_FALSE(f.rm->getID(names.rhoV() + "_x").has_value());
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

// Root level of a periodic single-level hierarchy: the ghost schedules must fill the whole ghost
// layer from the level's periodic images, for either temporal transfer.
template<auto opts, typename Transfer>
void expectRootConservativeGhostsFilledPeriodically(
    int const nCells, PHARE::MHDOpts::TimeIntegratorType const integrator)
{
    using Layout = typename MHDTestTypes<opts>::Layout;

    SCOPED_TRACE("nCells=" + std::to_string(nCells));

    SolverFixture<opts, Transfer> f{nCells, integrator};
    auto& rm    = f.rm;
    auto& level = f.level;
    auto& model = f.model;

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

    f.messenger.fillConservativeGhosts(model.state, *level, 0.0,
                                       PHARE::core::RKStageContext::endOfStep(0.0));

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
    using Transfer = PHARE::amr::Linear2TemporalTransfer<MHDTestTypes<O2Opts>::Model>;
    for (auto const n : {8, 32})
        expectRootConservativeGhostsFilledPeriodically<O2Opts, Transfer>(
            n, PHARE::MHDOpts::TimeIntegratorType::TVDRK3);
}

TEST(MHDRootGhostFill, O4MC2011FillsEveryConservativeGhostFromPeriodicImages)
{
    using Transfer = PHARE::amr::MC2011TemporalTransfer<MHDTestTypes<O4Opts>::Model>;
    for (auto const n : {8, 32})
        expectRootConservativeGhostsFilledPeriodically<O4Opts, Transfer>(
            n, PHARE::MHDOpts::TimeIntegratorType::SSPRK4_5);
}

TEST(MC2011TemporalTransfer, RequiresPublishedOwnersAndOnlyAllocatesAssembledScratch)
{
    using Types    = MHDTestTypes<O4Opts>;
    using State    = PHARE::core::MHDConservedState<typename Types::VecField>;
    using Transfer = PHARE::amr::MC2011TemporalTransfer<Types::Model>;

    auto rm = std::make_shared<typename Types::ResourcesManager>();
    State old{"owned_old"}, s1{"owned_s1"}, s2{"owned_s2"}, s3{"owned_s3"}, s4{"owned_s4"},
        unp1{"owned_final"};
    for (auto* state : {&old, &s1, &s2, &s3, &s4, &unp1})
        rm->registerResources(*state);

    PHARE::amr::MHDMessengerInfo info;
    info.oldState = old.names;
    info.ssprk54History
        = PHARE::amr::SSPRK54HistoryNames{{s1.names, s2.names, s3.names, s4.names}, unp1.names};

    Transfer transfer{rm};
    rm->registerResources(transfer);
    transfer.registerQuantities(info);
    EXPECT_TRUE(transfer.hasCoarseFine());
    EXPECT_EQ(rm->getID(info.oldState.rho()), rm->getID(old.rho.name()));
    auto const stage3B = rm->getID(s3.B.name());
    ASSERT_TRUE(stage3B.has_value());
    EXPECT_EQ(rm->getID(info.ssprk54History->stages[2].B()), stage3B);
    EXPECT_NE(rm->getID(transfer.assembledStateNames().rho()), rm->getID(info.oldState.rho()));

    // without the SSPRK(5,4) stage history (any other integrator) the transfer is inactive, not
    // an error: MHDMessenger refuses finer levels in that case
    auto noHistory           = info;
    noHistory.ssprk54History = std::nullopt;
    Transfer inactive{rm};
    inactive.registerQuantities(noHistory);
    EXPECT_FALSE(inactive.hasCoarseFine());

    auto unowned                      = info;
    unowned.ssprk54History->stages[0] = PHARE::core::MHDStateNames{"unowned"};
    Transfer missingOwner{rm};
    EXPECT_THROW(missingOwner.registerQuantities(unowned), std::invalid_argument);
}

// Counts the messenger's calls into the transfer, per SSPRK(5,4) stage.
template<typename Model>
class CountingTransfer : public PHARE::amr::MC2011TemporalTransfer<Model>
{
    using Base = PHARE::amr::MC2011TemporalTransfer<Model>;

public:
    using Base::Base;
    static inline std::size_t outsideStages = 0;
    static inline std::array<std::size_t, 5> perStage{};

    static void reset()
    {
        outsideStages = 0;
        perStage.fill(0);
    }

    template<typename State, typename Level>
    void prepareTarget(State& state, Level const& level, PHARE::core::RKStageContext const& context)
    {
        if (context.stepDuration == 0.0)
            ++outsideStages;
        else
            ++perStage.at(context.stageIndex);
        Base::prepareTarget(state, level, context);
    }
};

TEST(SSPRK54ConservativeGhosts, RealMessengerPreparesExactlyOncePerStage)
{
    using Transfer = CountingTransfer<MHDTestTypes<O4Opts>::Model>;

    SolverFixture<O4Opts, Transfer> f{8, PHARE::MHDOpts::TimeIntegratorType::SSPRK4_5};
    f.model.initialize(*f.level);
    f.solver.prepareStep(f.model, *f.level, 0.0);

    Transfer::reset();
    f.solver.advanceLevel(*f.hierarchy, 0, f.model, f.messenger, 0.0, 1.e-4);

    EXPECT_EQ(Transfer::outsideStages, 0u);
    EXPECT_EQ(Transfer::perStage, (std::array<std::size_t, 5>{1, 1, 1, 1, 1}));
}

// the field-refinement order follows the compiled MHD order, it is not an input
TEST(RefinementConfigFor, followsTheCompiledProfile)
{
    EXPECT_EQ(PHARE::refinementConfigFor<O2Opts>().order, PHARE::amr::FieldRefinementOrder::Linear);
    EXPECT_EQ(PHARE::refinementConfigFor<O4Opts>().order, PHARE::amr::FieldRefinementOrder::Cubic);

    constexpr PHARE::SimOpts hybridOnly{1, 1};
    EXPECT_EQ(PHARE::refinementConfigFor<hybridOnly>().order,
              PHARE::amr::FieldRefinementOrder::Linear);
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
