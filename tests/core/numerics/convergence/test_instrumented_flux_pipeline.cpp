#include "phare_core.hpp"
#include "phare_solver.hpp"
#include "gtest/gtest.h"

#include <memory>
#include <type_traits>
#include <vector>

#include "amr/resources_manager/amr_utils.hpp"
#include "amr/solvers/time_integrator/compute_fluxes.hpp"
#include "amr/solvers/time_integrator/point_value_approximation.hpp"
#include "tests/core/numerics/convergence/convergence_test_framework.hpp"
#include "core/numerics/godunov_fluxes/godunov_fluxes.hpp"
#include "tests/core/numerics/convergence/exact_solutions.hpp"
#include "tests/core/numerics/convergence/hall_convergence_test_common.hpp"

#include <SAMRAI/tbox/SAMRAIManager.h>
#include <SAMRAI/tbox/SAMRAI_MPI.h>

using namespace PHARE::core;
using PHARE::test::MultiQuantityConvergenceStudy;
using PHARE::test::fillCellAveragedField;
using PHARE::test::fillFaceAveragedField;
using PHARE::test::l2EdgeAveragedError;
using PHARE::test::l2FaceAveragedFluxError;

namespace
{
template<bool EnableHall>
struct FVMethod3D
{
    // Godunov takes a Reconstruction template of exactly one parameter.
    // PointValueWENOZReconstruction carries a defaulted SlopeLimiter, and binding a
    // two-parameter template to a one-parameter template template parameter is P0522 relaxed
    // matching -- applied by default by GCC, not by Clang. Wrap it down to one parameter, the
    // same way the solver does in MHDResolver::Reconstruction_t.
    template<typename GridLayoutT>
    using Reconstruction = PointValueWENOZReconstruction<GridLayoutT>;

    template<typename GridLayoutT>
    using type = Godunov<GridLayoutT, Reconstruction, Rusanov<EnableHall>,
                         MHDEquations<EnableHall, false, false>>;
};

template<bool EnableHall>
void runInstrumentedPipelineTest()
{
    constexpr PHARE::SimOpts opts{
        3, 0, 0, PHARE::MHDOpts::MHDOrder::O4,
        PHARE::MHDOpts::TimeIntegratorType::SSPRK4_5,
        PHARE::MHDOpts::ReconstructionType::WENOZ, PHARE::MHDOpts::SlopeLimiterType::None,
        PHARE::MHDOpts::RiemannSolverType::Rusanov, EnableHall};
    using Layout = typename PHARE::core::PHARE_Types<opts>::MHD::GridLayout_t;
    using Array3D = NdArrayVector<3>;
    using Grid3D = Grid<Array3D, MHDQuantity::Scalar>;
    using Field3D = Field<3, MHDQuantity::Scalar>;
    using VecField3D = VecField<Field3D, MHDQuantity>;
    using ResourcesManagerT = PHARE::amr::ResourcesManager<Layout, Grid3D>;
    using MHDModelT
        = PHARE::solver::MHDModel<Layout, VecField3D, PHARE::amr::SAMRAI_Types, Grid3D,
                                 PHARE::solver::initRepresentationFor<opts.mhd_order>>;
    using FluxesT = AllFluxes<Field3D, VecField3D>;
    using Exact = std::conditional_t<EnableHall, PHARE::test::ExactHall3D, PHARE::test::ExactIdealMHD3D>;
    // Resolved through the profile trait, not hand-picked: a bug making MHDProfileTraits<O4>
    // hand back the second-order approximation must break this measurement, not slip past it.
    using Approximation =
        typename PHARE::solver::MHDResolver<opts, MHDModelT>::PointValueApproximation;
    using ComputeFluxesT = PHARE::solver::ComputeFluxes<
        typename FVMethod3D<EnableHall>::template type<Layout>, MHDModelT, Approximation>;

    MultiQuantityConvergenceStudy study;
    for (auto const name : {"flux_rho_x", "flux_rho_y", "flux_rho_z", "flux_etot_x",
                            "flux_etot_y", "flux_etot_z", "electric_x", "electric_y",
                            "electric_z"})
        study.addQuantity(name);

    std::vector<int> const gridSizes{16, 32, 64};
    for (auto const n : gridSizes)
    {
        auto hierarchy = makePeriodicHierarchy3D(n);
        auto level     = hierarchy->getPatchLevel(0);
        auto resman    = std::make_shared<ResourcesManagerT>();
        auto modelDict = makeHall3DMHDModelDict();
        auto fluxDict  = makeHall3DComputeFluxDict();
        MHDModelT model{modelDict, resman};

        FluxesT fluxes{{"test_rho_fx", MHDQuantity::Scalar::ScalarFlux_x},
                       {"test_rhoV_fx", MHDQuantity::Vector::VecFlux_x},
                       {"test_B_fx", MHDQuantity::Vector::VecFlux_x},
                       {"test_Etot_fx", MHDQuantity::Scalar::ScalarFlux_x},
                       {"test_rho_fy", MHDQuantity::Scalar::ScalarFlux_y},
                       {"test_rhoV_fy", MHDQuantity::Vector::VecFlux_y},
                       {"test_B_fy", MHDQuantity::Vector::VecFlux_y},
                       {"test_Etot_fy", MHDQuantity::Scalar::ScalarFlux_y},
                       {"test_rho_fz", MHDQuantity::Scalar::ScalarFlux_z},
                       {"test_rhoV_fz", MHDQuantity::Vector::VecFlux_z},
                       {"test_B_fz", MHDQuantity::Vector::VecFlux_z},
                       {"test_Etot_fz", MHDQuantity::Scalar::ScalarFlux_z}};

        model.resourcesManager->registerResources(model.state);
        model.resourcesManager->registerResources(fluxes);

        ComputeFluxesT computeFluxes{fluxDict};
        computeFluxes.registerResources(model);
        for (auto& patch : *level)
        {
            model.allocate(*patch, 0.0);
            computeFluxes.allocate(model, *patch, 0.0);
            model.resourcesManager->allocate(fluxes, *patch, 0.0);
        }

        for (auto& patch : *level)
        {
            auto guard = model.resourcesManager->setOnPatch(*patch, model.state, fluxes);
            auto patchLayout = PHARE::amr::layoutFromPatch<Layout>(*patch);

            fillCellAveragedField(patchLayout, model.state.rho, Exact::rho);
            fillCellAveragedField(patchLayout, model.state.Etot, Exact::etot);
            fillCellAveragedField(patchLayout, model.state.rhoV(Component::X),
                                  [](double x, double y, double z) {
                                      return Exact::rho(x, y, z) * Exact::vx(x, y, z);
                                  });
            fillCellAveragedField(patchLayout, model.state.rhoV(Component::Y),
                                  [](double x, double y, double z) {
                                      return Exact::rho(x, y, z) * Exact::vy(x, y, z);
                                  });
            fillCellAveragedField(patchLayout, model.state.rhoV(Component::Z),
                                  [](double x, double y, double z) {
                                      return Exact::rho(x, y, z) * Exact::vz(x, y, z);
                                  });

            fillFaceAveragedField<Layout, decltype(model.state.B(Component::X)), decltype(Exact::bx),
                                  Direction::X>(patchLayout, model.state.B(Component::X), Exact::bx);
            fillFaceAveragedField<Layout, decltype(model.state.B(Component::Y)), decltype(Exact::by),
                                  Direction::Y>(patchLayout, model.state.B(Component::Y), Exact::by);
            fillFaceAveragedField<Layout, decltype(model.state.B(Component::Z)), decltype(Exact::bz),
                                  Direction::Z>(patchLayout, model.state.B(Component::Z), Exact::bz);
        }

        computeFluxes(model, model.state, fluxes, *level, 0.0);

        for (auto& patch : *level)
        {
            auto guard = model.resourcesManager->setOnPatch(*patch, model.state, fluxes);
            auto patchLayout = PHARE::amr::layoutFromPatch<Layout>(*patch);
            fillFluxGhosts(patchLayout, fluxes);
            periodicFillGhostsVec(patchLayout, model.state.E);

            study.recordError("flux_rho_x", n,
                              l2FaceAveragedFluxError<Layout, Direction::X>(
                                  patchLayout, fluxes.rho_fx, [](double x, double y, double z) {
                                      return Exact::flux(Direction::X, x, y, z)[0];
                                  }));
            study.recordError("flux_rho_y", n,
                              l2FaceAveragedFluxError<Layout, Direction::Y>(
                                  patchLayout, fluxes.rho_fy, [](double x, double y, double z) {
                                      return Exact::flux(Direction::Y, x, y, z)[0];
                                  }));
            study.recordError("flux_rho_z", n,
                              l2FaceAveragedFluxError<Layout, Direction::Z>(
                                  patchLayout, fluxes.rho_fz, [](double x, double y, double z) {
                                      return Exact::flux(Direction::Z, x, y, z)[0];
                                  }));
            study.recordError("flux_etot_x", n,
                              l2FaceAveragedFluxError<Layout, Direction::X>(
                                  patchLayout, fluxes.Etot_fx, [](double x, double y, double z) {
                                      return Exact::flux(Direction::X, x, y, z)[4];
                                  }));
            study.recordError("flux_etot_y", n,
                              l2FaceAveragedFluxError<Layout, Direction::Y>(
                                  patchLayout, fluxes.Etot_fy, [](double x, double y, double z) {
                                      return Exact::flux(Direction::Y, x, y, z)[4];
                                  }));
            study.recordError("flux_etot_z", n,
                              l2FaceAveragedFluxError<Layout, Direction::Z>(
                                  patchLayout, fluxes.Etot_fz, [](double x, double y, double z) {
                                      return Exact::flux(Direction::Z, x, y, z)[4];
                                  }));
            study.recordError("electric_x", n,
                              l2EdgeAveragedError<Layout, Direction::X>(
                                  patchLayout, model.state.E(Component::X), [](double x, double y,
                                                                              double z) {
                                      return Exact::electric(x, y, z)[0];
                                  }));
            study.recordError("electric_y", n,
                              l2EdgeAveragedError<Layout, Direction::Y>(
                                  patchLayout, model.state.E(Component::Y), [](double x, double y,
                                                                              double z) {
                                      return Exact::electric(x, y, z)[1];
                                  }));
            study.recordError("electric_z", n,
                              l2EdgeAveragedError<Layout, Direction::Z>(
                                  patchLayout, model.state.E(Component::Z), [](double x, double y,
                                                                              double z) {
                                      return Exact::electric(x, y, z)[2];
                                  }));
        }
    }

    study.computeOrders();
    study.printSummary();

    for (auto const name : {"flux_rho_x", "flux_rho_y", "flux_rho_z", "flux_etot_x",
                            "flux_etot_y", "flux_etot_z", "electric_x", "electric_y",
                            "electric_z"})
    {
        auto const& result = study.getResult(name);
        ASSERT_EQ(result.errors.size(), gridSizes.size()) << name;
        ASSERT_EQ(result.orders.size(), gridSizes.size() - 1) << name;
        for (auto const order : result.orders)
        {
            if constexpr (EnableHall)
                EXPECT_GT(order, 1.75) << name;
            else
                EXPECT_GE(order, 3.5) << name;
        }
    }
}
} // namespace

TEST(InstrumentedPipeline, IdealMHD3D)
{
    runInstrumentedPipelineTest<false>();
}

TEST(InstrumentedPipeline, HallMHD3D)
{
    runInstrumentedPipelineTest<true>();
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    SAMRAI::tbox::SAMRAI_MPI::init(&argc, &argv);
    SAMRAI::tbox::SAMRAIManager::initialize();
    SAMRAI::tbox::SAMRAIManager::startup();

    int testResult = RUN_ALL_TESTS();

    SAMRAI::tbox::SAMRAIManager::shutdown();
    SAMRAI::tbox::SAMRAIManager::finalize();
    SAMRAI::tbox::SAMRAI_MPI::finalize();
    return testResult;
}
