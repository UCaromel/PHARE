#include "phare_core.hpp"

#include "amr/messengers/mhd_messenger.hpp"
#include "amr/messengers/mhd_temporal_transfer.hpp"
#include "amr/messengers/mc2011_temporal_transfer.hpp"
#include "amr/messenger_registration.hpp"
#include "amr/resources_manager/amr_utils.hpp"
#include "amr/resources_manager/resources_manager.hpp"
#include "amr/solvers/mhd_resolver.hpp"
#include "amr/solvers/solver_mhd.hpp"
#include "amr/solvers/time_integrator/point_value_approximation.hpp"
#include "core/models/mhd_state_increment.hpp"
#include "core/numerics/constrained_transport/upwind_constrained_transport_utils.hpp"
#include "tests/core/numerics/convergence/hall_convergence_test_common.hpp"

#include <gtest/gtest.h>

#include <set>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace
{
constexpr PHARE::SimOpts O2Opts{
    3, 0, 0, PHARE::MHDOpts::MHDOrder::O2, PHARE::MHDOpts::TimeIntegratorType::TVDRK3,
    PHARE::MHDOpts::ReconstructionType::WENOZ, PHARE::MHDOpts::SlopeLimiterType::None,
    PHARE::MHDOpts::RiemannSolverType::Rusanov};
constexpr PHARE::SimOpts O4Opts{
    3, 0, 0, PHARE::MHDOpts::MHDOrder::O4, PHARE::MHDOpts::TimeIntegratorType::TVDRK3,
    PHARE::MHDOpts::ReconstructionType::WENOZ, PHARE::MHDOpts::SlopeLimiterType::None,
    PHARE::MHDOpts::RiemannSolverType::Rusanov};
constexpr PHARE::SimOpts O4SSPRKOpts{
    3, 0, 0, PHARE::MHDOpts::MHDOrder::O4, PHARE::MHDOpts::TimeIntegratorType::SSPRK4_5,
    PHARE::MHDOpts::ReconstructionType::WENOZ, PHARE::MHDOpts::SlopeLimiterType::None,
    PHARE::MHDOpts::RiemannSolverType::Rusanov};

template<auto opts>
struct MHDTestTypes
{
    using CoreTypes = PHARE::core::PHARE_Types<opts>;
    using Layout    = typename CoreTypes::MHD::GridLayout_t;
    using Grid      = typename CoreTypes::MHD::Grid_t;
    using VecField  = typename CoreTypes::MHD::VecField_t;
    using Model = PHARE::solver::MHDModel<Layout, VecField, PHARE::amr::SAMRAI_Types, Grid>;
    using ResourcesManager = PHARE::amr::ResourcesManager<Layout, Grid>;
};

// Test-only composition seam: the real MHDMessenger calls this policy once from
// each conservative fill.  No production counter is introduced.
template<typename Model>
class CountingNoCoarseFineTransfer : public PHARE::amr::NoCoarseFineTemporalTransfer<Model>
{
    using Base = PHARE::amr::NoCoarseFineTemporalTransfer<Model>;

public:
    using Base::Base;
    static inline std::size_t ordinaryPreparations = 0;
    static inline std::array<std::size_t, 5> stagePreparations{};

    static void reset()
    {
        ordinaryPreparations = 0;
        stagePreparations.fill(0);
    }

    template<typename State, typename Level>
    void prepareTarget(State& state, Level const& level, double const time)
    {
        ++ordinaryPreparations;
        Base::prepareTarget(state, level, time);
    }

    template<typename State, typename Level>
    void prepareTarget(State& state, Level const& level, double const time,
                       PHARE::solver::RKStageContext const& context)
    {
        if (context.stageIndex >= stagePreparations.size())
            throw std::out_of_range{"invalid SSPRK stage"};
        ++stagePreparations[context.stageIndex];
        Base::prepareTarget(state, level, time, context);
    }
};

TEST(MHDStateIncrementNames, RejectsPartialFamilies)
{
    auto names = PHARE::core::MHDStateIncrementNames{"old"};
    names.B_z  = "other_B_z";
    EXPECT_THROW(static_cast<void>(names.validatedBaseName()), std::invalid_argument);
}

template<bool Hall, bool Resistivity, bool HyperResistivity, typename VecField>
constexpr std::size_t ctStateResourceCount()
{
    using State
        = PHARE::core::UpwindConstrainedTransportState<VecField, Hall, Resistivity,
                                                       HyperResistivity>;
    return std::tuple_size_v<
        decltype(std::declval<State&>().getCompileTimeResourcesViewList())>;
}

// GodunovFluxes writes the transverse-current buffers (j_t_*, rho_t_*) through the save()
// overload it selects for `Hall || Resistivity || HyperResistivity`, and they are read back by
// the Hall EMF terms, the resistive flux contribution and the spatial hyper-resistive term.
// Registering them on the narrower `Hall || Resistivity` left a hyper-resistivity-only scheme
// (Hall off, eta off, nu on) writing through a VecField whose buffer was never set -- invisible
// at initialize, and surfacing on the first advance as "Error - TensorField not usable".
TEST(MHDCTStateResources, TransverseCurrentIsOwnedByEverySchemeThatCarriesACurrent)
{
    using Types    = MHDTestTypes<O2Opts>;
    using VecField = Types::VecField;

    constexpr auto ideal     = ctStateResourceCount<false, false, false, VecField>();
    constexpr auto hallOnly  = ctStateResourceCount<true, false, false, VecField>();
    constexpr auto resOnly   = ctStateResourceCount<false, true, false, VecField>();
    constexpr auto hyperOnly = ctStateResourceCount<false, false, true, VecField>();

    static_assert(hallOnly > ideal, "a Hall scheme owns the transverse-current buffers");
    static_assert(resOnly == hallOnly, "so does a resistive scheme");
    static_assert(hyperOnly == hallOnly,
                  "and so must a hyper-resistivity-only scheme -- GodunovFluxes writes j_t/rho_t "
                  "for Hall || Resistivity || HyperResistivity, so any narrower predicate writes "
                  "through an unset buffer");

    // and they are reachable as real resources under that configuration
    auto rm = std::make_shared<typename Types::ResourcesManager>();
    PHARE::core::UpwindConstrainedTransportState<VecField, false, false, true> hyperOnlyState{};
    rm->registerResources(hyperOnlyState);
    EXPECT_TRUE(rm->getID("j_t_x").has_value());
    EXPECT_TRUE(rm->getID("rho_t_x").has_value());
}

TEST(MHDProfileResources, O2HasNoPointOrTemporalPolicyResources)
{
    using Types      = MHDTestTypes<O2Opts>;
    using Model      = Types::Model;
    using Approx     = PHARE::solver::SecondOrderPointValueApproximation<Model>;
    using Transfer   = PHARE::amr::Linear2TemporalTransfer<Model>;
    using Messenger  = PHARE::amr::MHDMessenger<Model, Transfer>;
    using Stepper    = typename PHARE::solver::MHDResolver<O2Opts, Model>::MHDTimeStepper_t;
    using Solver     = PHARE::solver::SolverMHD<Model, PHARE::amr::SAMRAI_Types, Stepper, Messenger>;

    static_assert(std::tuple_size_v<decltype(std::declval<Approx const&>()
                                                 .getCompileTimeResourcesViewList())>
                  == 0);
    static_assert(std::tuple_size_v<decltype(std::declval<Transfer const&>()
                                                 .getCompileTimeResourcesViewList())>
                  == 0);

    auto hierarchy = makePeriodicHierarchy3D(8);
    auto level     = hierarchy->getPatchLevel(0);
    auto rm        = std::make_shared<typename Types::ResourcesManager>();
    Model model{makeHall3DMHDModelDict(), rm};
    Solver solver{makeHall3DComputeFluxDict()};
    Approx approximation;
    Messenger messenger{rm, 0};

    rm->registerResources(model.state);
    solver.registerResources(model);
    rm->registerResources(approximation);
    PHARE::solver::MessengerRegistration::registerQuantities<PHARE::amr::SAMRAI_Types>(
        messenger, model, model, solver);
    messenger.registerLevel(hierarchy, 0);
    for (auto& patch : *level)
    {
        model.allocate(*patch, 0.0);
        solver.allocate(model, *patch, 0.0);
        rm->allocate(approximation, *patch, 0.0);
        messenger.allocate(*patch, 0.0);
    }

    EXPECT_FALSE(rm->getID("point_value_rho").has_value());
    EXPECT_FALSE(rm->getID("mc2011_assembled_rho").has_value());

    // The eight-name family is the naming contract; the registration granularity is coarser.
    // rhoV and B are TensorFields, and a TensorField is itself a resource (is_tensor_field_v
    // feeds is_resource), so the ResourcesManager holds one patch data per TensorField, keyed by
    // the TensorField's own name. The solver-owned old state therefore publishes four keys, and
    // the per-component names in MHDStateIncrementNames name the Fields *inside* those
    // TensorFields (core::detail::tensor_field_names) -- never keys. This is exactly the
    // convention registerGhostComms_ relies on when it asks SAMRAI for `oldBase + "_rhoV"`.
    auto const names = PHARE::core::MHDStateIncrementNames{"MHDSolver_stateOld"};
    auto const base  = names.validatedBaseName();
    auto const ids   = rm->getIDsList(names.rho, base + "_rhoV", names.Etot, base + "_B");
    auto const uniqueIDs
        = std::set<int>{std::get<0>(ids), std::get<1>(ids), std::get<2>(ids), std::get<3>(ids)};
    std::apply(
        [&](auto const... id) {
            (([&] {
                 for (auto& patch : *level)
                     EXPECT_TRUE(patch->checkAllocated(id));
             }()),
             ...);
        },
        ids);
    EXPECT_EQ(uniqueIDs.size(), 4u);

    // Pin the granularity itself: a component name resolving to an ID would mean TensorField
    // registration had been split per component, which every `base + "_rhoV"` call site assumes
    // it is not.
    EXPECT_FALSE(rm->getID(names.rhoV_x).has_value());
    EXPECT_FALSE(rm->getID(names.B_z).has_value());
}

// The ExactHall3D profile has period 1 in every direction and the fixture domain is the unit
// cube, so the analytic value at a ghost node's own coordinate is exactly the value the periodic
// source node carries.  Seeding every allocated node from the profile therefore turns "was this
// ghost filled from its periodic image" into a pure value check -- and, because
// fillConservativeGhosts() NaNs the whole ghost layer first, a node that no transaction wrote
// shows up as non-finite rather than as a stale value that happens to look plausible.
template<typename Layout, typename Field, typename Fn>
void expectPeriodicGhostValues(Layout const& layout, Field& field, Fn&& fn,
                               std::string const& name, double const tol)
{
    auto const cent = layout.centering(field.physicalQuantity());

    std::size_t nonFinite = 0;
    std::size_t mismatched = 0;
    double maxErr = 0.0;
    std::string firstBad;

    auto const note = [&](auto i, auto j, auto kk) {
        if (firstBad.empty())
            firstBad = "(" + std::to_string(static_cast<int>(i)) + ","
                       + std::to_string(static_cast<int>(j)) + ","
                       + std::to_string(static_cast<int>(kk)) + ")";
    };

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
                    note(i, j, kk);
                    continue;
                }
                auto const c = layout.fieldNodeCoordinates(
                    field, layout.localToAMR(Point{i, j, kk}.as_signed()));
                auto const err = std::abs(v - fn(c[0], c[1], c[2]));
                maxErr = std::max(maxErr, err);
                if (err > tol)
                {
                    ++mismatched;
                    note(i, j, kk);
                }
            }

    EXPECT_EQ(nonFinite, std::size_t{0})
        << name << ": " << nonFinite << " node(s) still hold the pre-fill NaN sentinel, first at "
        << firstBad << " -- the schedule did not fill them";
    EXPECT_EQ(mismatched, std::size_t{0})
        << name << ": " << mismatched << " node(s) off the periodic profile, first at " << firstBad
        << ", max error " << maxErr;
}

// Root level (next coarser level number == -1) with a genuinely periodic single-level hierarchy:
// the GhostField schedules must transport the whole ghost layer from the level's own periodic
// images.  Constructing the schedules is not the property under test -- filling is.
template<auto opts, typename Transfer>
void expectRootConservativeGhostsFilledPeriodically(int const nCells)
{
    using Types     = MHDTestTypes<opts>;
    using Model     = typename Types::Model;
    using Layout    = typename Types::Layout;
    using Messenger = PHARE::amr::MHDMessenger<Model, Transfer>;
    using Stepper   = typename PHARE::solver::MHDResolver<opts, Model>::MHDTimeStepper_t;
    using Solver    = PHARE::solver::SolverMHD<Model, PHARE::amr::SAMRAI_Types, Stepper, Messenger>;

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

    messenger.fillConservativeGhosts(model.state, *level, 0.0);

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

TEST(MHDRootGhostFill, O2Linear2FillsEveryConservativeGhostFromPeriodicImages)
{
    using Types = MHDTestTypes<O2Opts>;
    expectRootConservativeGhostsFilledPeriodically<
        O2Opts, PHARE::amr::Linear2TemporalTransfer<Types::Model>>(8);
    expectRootConservativeGhostsFilledPeriodically<
        O2Opts, PHARE::amr::Linear2TemporalTransfer<Types::Model>>(32);
}

TEST(MHDRootGhostFill, O4NoCoarseFineFillsEveryConservativeGhostFromPeriodicImages)
{
    using Types = MHDTestTypes<O4SSPRKOpts>;
    expectRootConservativeGhostsFilledPeriodically<
        O4SSPRKOpts, PHARE::amr::NoCoarseFineTemporalTransfer<Types::Model>>(8);
    expectRootConservativeGhostsFilledPeriodically<
        O4SSPRKOpts, PHARE::amr::NoCoarseFineTemporalTransfer<Types::Model>>(32);
}

TEST(MHDProfileResources, O4AllocatesPointValueResources)
{
    using Types    = MHDTestTypes<O4Opts>;
    using Model    = Types::Model;
    using Approx   = PHARE::solver::FourthOrderPointValueApproximation<Model>;
    using Transfer = PHARE::amr::NoCoarseFineTemporalTransfer<Model>;

    static_assert(std::tuple_size_v<decltype(std::declval<Transfer const&>()
                                                 .getCompileTimeResourcesViewList())>
                  == 0);

    auto hierarchy = makePeriodicHierarchy3D(8);
    auto level     = hierarchy->getPatchLevel(0);
    auto rm        = std::make_shared<typename Types::ResourcesManager>();
    Approx approximation;
    Transfer transfer{rm};

    rm->registerResources(approximation);
    rm->registerResources(transfer);
    for (auto& patch : *level)
    {
        rm->allocate(approximation, *patch, 0.0);
        rm->allocate(transfer, *patch, 0.0);
    }

    // PointValueState publishes one key per member, TensorField members included: the keys are
    // the names given in point_value_handler_utils.hpp, not per-component names.
    auto const pointIDs = rm->getIDsList(std::string{"point_value_rho"},
                                         std::string{"point_value_V"}, std::string{"point_value_B"},
                                         std::string{"point_value_P"}, std::string{"point_value_J"},
                                         std::string{"point_value_rhoV"},
                                         std::string{"point_value_Etot"});
    EXPECT_EQ(std::set<int>(pointIDs.begin(), pointIDs.end()).size(), pointIDs.size());
    for (auto& patch : *level)
        for (auto const id : pointIDs)
            EXPECT_TRUE(patch->checkAllocated(id));

    // Same granularity pin as the O2 case: components of point_value_rhoV / point_value_B live
    // inside their TensorField and are not resources of their own.
    EXPECT_FALSE(rm->getID("point_value_rhoV_x").has_value());
    EXPECT_FALSE(rm->getID("point_value_B_x").has_value());
    EXPECT_FALSE(rm->getID("mc2011_assembled_rho").has_value());
}

TEST(MC2011TemporalTransfer, RequiresPublishedOwnersAndOnlyAllocatesAssembledScratch)
{
    using Types = MHDTestTypes<O4Opts>;
    using Model = Types::Model;
    using Increment = PHARE::core::MHDStateIncrement<typename Types::VecField>;
    using Transfer = PHARE::amr::MC2011TemporalTransfer<Model>;

    auto rm = std::make_shared<typename Types::ResourcesManager>();
    Increment old{"owned_old"}, s1{"owned_s1"}, s2{"owned_s2"}, s3{"owned_s3"}, s4{"owned_s4"},
        unp1{"owned_final"};
    rm->registerResources(old); rm->registerResources(s1); rm->registerResources(s2);
    rm->registerResources(s3); rm->registerResources(s4); rm->registerResources(unp1);

    PHARE::amr::MHDMessengerInfo info;
    info.oldState = PHARE::core::MHDStateIncrementNames{old};
    info.ssprk54History = PHARE::amr::SSPRK54HistoryNames{
        {PHARE::core::MHDStateIncrementNames{s1}, PHARE::core::MHDStateIncrementNames{s2},
         PHARE::core::MHDStateIncrementNames{s3}, PHARE::core::MHDStateIncrementNames{s4}},
        PHARE::core::MHDStateIncrementNames{unp1}};

    Transfer transfer{rm};
    rm->registerResources(transfer);
    transfer.registerQuantities(info);
    EXPECT_EQ(rm->getID(info.oldState.rho), rm->getID(old.rho.name()));
    // Compared at the granularity the ResourcesManager actually keys on: s3.B is one registered
    // TensorField, so its key is s3.B.name() == <stage base> + "_B". Asserting the ID exists
    // first keeps this from passing vacuously on two nullopts, which is how the old
    // component-name form (stages[2].B_y vs getComponentName(Y)) hid the owner-check defect.
    auto const stage3B = rm->getID(s3.B.name());
    ASSERT_TRUE(stage3B.has_value());
    EXPECT_EQ(rm->getID(info.ssprk54History->stages[2].validatedBaseName() + "_B"), stage3B);
    EXPECT_NE(rm->getID(transfer.assembledStateNames().rho), rm->getID(info.oldState.rho));

    auto malformed = info;
    malformed.ssprk54History->finalState.B_z = "not_the_final_B_z";
    Transfer invalidFamily{rm};
    EXPECT_THROW(invalidFamily.registerQuantities(malformed), std::invalid_argument);

    auto unowned = info;
    unowned.ssprk54History->stages[0] = PHARE::core::MHDStateIncrementNames{"unowned"};
    Transfer missingOwner{rm};
    EXPECT_THROW(missingOwner.registerQuantities(unowned), std::invalid_argument);
}

TEST(SSPRK54ConservativeGhosts, RealMessengerPreparesExactlyOncePerStage)
{
    using Types = MHDTestTypes<O4SSPRKOpts>;
    using Model = Types::Model;
    using Transfer = CountingNoCoarseFineTransfer<Model>;
    using Messenger = PHARE::amr::MHDMessenger<Model, Transfer>;
    using Stepper = typename PHARE::solver::MHDResolver<O4SSPRKOpts, Model>::MHDTimeStepper_t;
    using Solver = PHARE::solver::SolverMHD<Model, PHARE::amr::SAMRAI_Types, Stepper, Messenger>;

    auto hierarchy = makePeriodicHierarchy3D(8);
    auto level = hierarchy->getPatchLevel(0);
    auto rm = std::make_shared<typename Types::ResourcesManager>();
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
    model.initialize(*level);
    solver.prepareStep(model, *level, 0.0);

    Transfer::reset();
    solver.advanceLevel(*hierarchy, 0, model, messenger, 0.0, 1.e-4);

    EXPECT_EQ(Transfer::ordinaryPreparations, 0u);
    EXPECT_EQ(Transfer::stagePreparations,
              (std::array<std::size_t, 5>{1, 1, 1, 1, 1}));
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
