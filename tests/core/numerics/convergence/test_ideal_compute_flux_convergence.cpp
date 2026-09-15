/**
 * @file test_ideal_compute_flux_convergence.cpp
 *
 * Full Ideal MHD 3D ComputeFluxes convergence test with periodic ghost filling.
 * 
 * This test is a CRITICAL DIAGNOSTIC:
 * - If this passes with order 4: Hall term is the bottleneck
 * - If this fails: Problem is more fundamental (3D indexing, cross-derivatives, etc.)
 * 
 * Tests E-field (E = -V×B, no Hall term) which is the "flux" for B evolution.
 */

#include "phare_core.hpp"
#include "phare_solver.hpp"
#include "gtest/gtest.h"

#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "amr/resources_manager/amr_utils.hpp"
#include "amr/solvers/time_integrator/point_value_approximation.hpp"
#include "amr/solvers/time_integrator/compute_fluxes.hpp"
#include "core/numerics/godunov_fluxes/godunov_fluxes.hpp"
#include "tests/core/numerics/convergence/hall_convergence_test_common.hpp"
#include "tests/core/numerics/convergence/exact_solutions.hpp"

#include <SAMRAI/tbox/SAMRAIManager.h>
#include <SAMRAI/tbox/SAMRAI_MPI.h>

using namespace PHARE::core;
using PHARE::test::fillCellAveragedField;
using PHARE::test::fillFaceAveragedField;
using PHARE::test::l2FaceAveragedFluxError;
using PHARE::test::l2EdgeAveragedError;

namespace
{

// Ideal MHD equations: Hall = false, Resistivity = false, HyperResistivity = false
template<typename MHDModelT, bool PointValues>
struct IdealMHDFVMethod3D
{
    template<typename GridLayoutT>
    using Reconstruction = WENOZReconstruction<GridLayoutT, void, PointValues>;

    template<typename GridLayoutT>
    using type
        = Godunov<GridLayoutT, Reconstruction, Rusanov<true>, MHDEquations<false, false, false>>;
};

using ExactIdealMHD3D = PHARE::test::ExactIdealMHD3D;

template<PHARE::MHDOpts::MHDOrder Order>
auto runIdealMHDFluxConvergence()
{
    constexpr PHARE::SimOpts opts{
        3, 0, 0, Order, PHARE::MHDOpts::TimeIntegratorType::SSPRK4_5,
        PHARE::MHDOpts::ReconstructionType::WENOZ, PHARE::MHDOpts::SlopeLimiterType::None,
        PHARE::MHDOpts::RiemannSolverType::Rusanov};
    using Layout = typename PHARE::core::PHARE_Types<opts>::MHD::GridLayout_t;
    using Array3D = NdArrayVector<3>;
    using Grid3D = Grid<Array3D, MHDQuantity::Scalar>;
    using Field3D = Field<3, MHDQuantity::Scalar>;
    using VecField3D = VecField<Field3D, MHDQuantity>;
    using ResourcesManagerT = PHARE::amr::ResourcesManager<Layout, Grid3D>;
    using MHDModelT
        = PHARE::solver::MHDModel<Layout, VecField3D, PHARE::amr::SAMRAI_Types, Grid3D,
                                 PHARE::solver::initRepresentationFor<Order>>;
    using FluxesT = AllFluxes<Field3D, VecField3D>;
    using Approximation = std::conditional_t<
        Order == PHARE::MHDOpts::MHDOrder::O2,
        PHARE::solver::SecondOrderPointValueApproximation<MHDModelT>,
        PHARE::solver::FourthOrderPointValueApproximation<MHDModelT>>;
    using ComputeFluxesT = PHARE::solver::ComputeFluxes<
        typename IdealMHDFVMethod3D<MHDModelT, Order == PHARE::MHDOpts::MHDOrder::O4>::template type<
            Layout>,
        MHDModelT, Approximation>;

    std::vector<int> nCells = {16, 32, 64};
    std::map<std::string, std::vector<double>> errors;
    auto push_error = [&](std::string const& key, double err) { errors[key].push_back(err); };

    for (auto n : nCells)
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

            // Conserved quantities are cell averages.
            fillCellAveragedField(patchLayout, model.state.rho, ExactIdealMHD3D::rho);
            fillCellAveragedField(patchLayout, model.state.Etot, ExactIdealMHD3D::etot);
            fillCellAveragedField(patchLayout, model.state.rhoV(Component::X), [](double x, double y, double z) {
                return ExactIdealMHD3D::rho(x, y, z) * ExactIdealMHD3D::vx(x, y, z);
            });
            fillCellAveragedField(patchLayout, model.state.rhoV(Component::Y), [](double x, double y, double z) {
                return ExactIdealMHD3D::rho(x, y, z) * ExactIdealMHD3D::vy(x, y, z);
            });
            fillCellAveragedField(patchLayout, model.state.rhoV(Component::Z), [](double x, double y, double z) {
                return ExactIdealMHD3D::rho(x, y, z) * ExactIdealMHD3D::vz(x, y, z);
            });
            
            // Magnetic field components are face averages.
            fillFaceAveragedField<Layout, decltype(model.state.B(Component::X)), decltype(ExactIdealMHD3D::bx), Direction::X>(
                patchLayout, model.state.B(Component::X), ExactIdealMHD3D::bx);
            fillFaceAveragedField<Layout, decltype(model.state.B(Component::Y)), decltype(ExactIdealMHD3D::by), Direction::Y>(
                patchLayout, model.state.B(Component::Y), ExactIdealMHD3D::by);
            fillFaceAveragedField<Layout, decltype(model.state.B(Component::Z)), decltype(ExactIdealMHD3D::bz), Direction::Z>(
                patchLayout, model.state.B(Component::Z), ExactIdealMHD3D::bz);
            
        }

        computeFluxes(model, model.state, fluxes, *level, 0.0);

        for (auto& patch : *level)
        {
            auto guard = model.resourcesManager->setOnPatch(*patch, model.state, fluxes);
            auto patchLayout = PHARE::amr::layoutFromPatch<Layout>(*patch);
            fillFluxGhosts(patchLayout, fluxes);
            periodicFillGhostsVec(patchLayout, model.state.E);

            // Mass fluxes - use FACE-AVERAGED exact fluxes for proper comparison
            push_error("rho_fx", l2FaceAveragedFluxError<Layout, Direction::X>(patchLayout, fluxes.rho_fx, [](double x, double y, double z) {
                return ExactIdealMHD3D::flux(Direction::X, x, y, z)[0];
            }));
            push_error("rho_fy", l2FaceAveragedFluxError<Layout, Direction::Y>(patchLayout, fluxes.rho_fy, [](double x, double y, double z) {
                return ExactIdealMHD3D::flux(Direction::Y, x, y, z)[0];
            }));
            push_error("rho_fz", l2FaceAveragedFluxError<Layout, Direction::Z>(patchLayout, fluxes.rho_fz, [](double x, double y, double z) {
                return ExactIdealMHD3D::flux(Direction::Z, x, y, z)[0];
            }));

            // Momentum fluxes (all components)
            for (int comp = 0; comp < 3; ++comp)
            {
                push_error("rhoV_fx_" + std::to_string(comp), 
                    l2FaceAveragedFluxError<Layout, Direction::X>(patchLayout, fluxes.rhoV_fx(static_cast<Component>(comp)), 
                        [comp](double x, double y, double z) {
                    return ExactIdealMHD3D::flux(Direction::X, x, y, z)[1 + comp];
                }));
                push_error("rhoV_fy_" + std::to_string(comp), 
                    l2FaceAveragedFluxError<Layout, Direction::Y>(patchLayout, fluxes.rhoV_fy(static_cast<Component>(comp)), 
                        [comp](double x, double y, double z) {
                    return ExactIdealMHD3D::flux(Direction::Y, x, y, z)[1 + comp];
                }));
                push_error("rhoV_fz_" + std::to_string(comp), 
                    l2FaceAveragedFluxError<Layout, Direction::Z>(patchLayout, fluxes.rhoV_fz(static_cast<Component>(comp)), 
                        [comp](double x, double y, double z) {
                    return ExactIdealMHD3D::flux(Direction::Z, x, y, z)[1 + comp];
                }));
            }

            // Energy fluxes (ideal MHD only, no Hall correction)
            push_error("Etot_fx", l2FaceAveragedFluxError<Layout, Direction::X>(patchLayout, fluxes.Etot_fx, [](double x, double y, double z) {
                return ExactIdealMHD3D::flux(Direction::X, x, y, z)[4];
            }));
            push_error("Etot_fy", l2FaceAveragedFluxError<Layout, Direction::Y>(patchLayout, fluxes.Etot_fy, [](double x, double y, double z) {
                return ExactIdealMHD3D::flux(Direction::Y, x, y, z)[4];
            }));
            push_error("Etot_fz", l2FaceAveragedFluxError<Layout, Direction::Z>(patchLayout, fluxes.Etot_fz, [](double x, double y, double z) {
                return ExactIdealMHD3D::flux(Direction::Z, x, y, z)[4];
            }));

            // E-field (ideal: E = -V×B, no Hall term)
            // This is the "flux" for B evolution: ∂B/∂t = -∇×E
            // E is edge-centered, so use EDGE-AVERAGED comparison
            push_error("Ex", l2EdgeAveragedError<Layout, Direction::X>(patchLayout, model.state.E(Component::X), [](double x, double y, double z) {
                return ExactIdealMHD3D::electric(x, y, z)[0];
            }));
            push_error("Ey", l2EdgeAveragedError<Layout, Direction::Y>(patchLayout, model.state.E(Component::Y), [](double x, double y, double z) {
                return ExactIdealMHD3D::electric(x, y, z)[1];
            }));
            push_error("Ez", l2EdgeAveragedError<Layout, Direction::Z>(patchLayout, model.state.E(Component::Z), [](double x, double y, double z) {
                return ExactIdealMHD3D::electric(x, y, z)[2];
            }));
        }
    }

    return std::make_tuple(nCells, errors);
}

} // namespace

template<PHARE::MHDOpts::MHDOrder Order>
void checkIdealMHDFluxConvergence(double const minimum, double const maximum)
{
    auto [nCells, errors] = runIdealMHDFluxConvergence<Order>();
    for (auto const& [name, error] : errors)
    {
        ASSERT_EQ(error.size(), nCells.size()) << name;
        for (std::size_t i = 1; i < error.size(); ++i)
        {
            auto const observed = convergenceOrder(error[i - 1], error[i]);
            EXPECT_GE(observed, minimum) << name << " refinement pair " << i;
            EXPECT_LE(observed, maximum) << name << " refinement pair " << i;
        }
    }
}

TEST(IdealMHDConvergence, SecondOrderPointValuePolicy)
{
    checkIdealMHDFluxConvergence<PHARE::MHDOpts::MHDOrder::O2>(1.75, 2.25);
}

TEST(IdealMHDConvergence, FourthOrderPointValuePolicy)
{
    checkIdealMHDFluxConvergence<PHARE::MHDOpts::MHDOrder::O4>(
        3.5, std::numeric_limits<double>::infinity());
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
