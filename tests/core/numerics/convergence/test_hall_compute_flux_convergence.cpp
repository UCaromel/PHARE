/**
 * @file test_hall_compute_flux_convergence.cpp
 *
 * Full Hall 3D ComputeFluxes convergence test with periodic ghost filling.
 */

#include "gtest/gtest.h"

#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "amr/resources_manager/amr_utils.hpp"
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
auto runFullFluxConvergence()
{
    using Layout = GridLayout<GridLayoutImplYeeMHD<3, 2>>;
    using Array3D = NdArrayVector<3>;
    using Grid3D = Grid<Array3D, MHDQuantity::Scalar>;
    using Field3D = Field<3, MHDQuantity::Scalar>;
    using VecField3D = VecField<Field3D, MHDQuantity>;
    using ResourcesManagerT = PHARE::amr::ResourcesManager<Layout, Grid3D>;
    using MHDModelT = PHARE::solver::MHDModel<Layout, VecField3D, PHARE::amr::SAMRAI_Types, Grid3D>;
    using FluxesT = AllFluxes<Field3D, VecField3D>;
    using ComputeFluxesT
        = PHARE::solver::ComputeFluxes<HallFVMethod3D<MHDModelT>::template type<Layout>, MHDModelT>;

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

        Layout layout = PHARE::amr::layoutFromPatch<Layout>(*(*level->begin()));
        Hall3DPeriodicGhostFiller<Layout, ResourcesManagerT> bc{layout, *model.resourcesManager};

        {
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

                // Use proper averaging for initialization
                fillCellAveragedField(patchLayout, model.state.rho, ExactHall3D::rho);
                fillCellAveragedField(patchLayout, model.state.P, ExactHall3D::pressure);
                fillCellAveragedField(patchLayout, model.state.Etot, ExactHall3D::etot);
                
                fillCellAveragedField(patchLayout, model.state.rhoV(Component::X), [](double x, double y, double z) {
                    return ExactHall3D::rho(x, y, z) * ExactHall3D::vx(x, y, z);
                });
                fillCellAveragedField(patchLayout, model.state.rhoV(Component::Y), [](double x, double y, double z) {
                    return ExactHall3D::rho(x, y, z) * ExactHall3D::vy(x, y, z);
                });
                fillCellAveragedField(patchLayout, model.state.rhoV(Component::Z), [](double x, double y, double z) {
                    return ExactHall3D::rho(x, y, z) * ExactHall3D::vz(x, y, z);
                });
                
                // V is also cell-centered (corrected!)
                fillCellAveragedField(patchLayout, model.state.V(Component::X), ExactHall3D::vx);
                fillCellAveragedField(patchLayout, model.state.V(Component::Y), ExactHall3D::vy);
                fillCellAveragedField(patchLayout, model.state.V(Component::Z), ExactHall3D::vz);
                
                // Only B is face-centered (electromagnetic quantities on Yee grid)
                fillFaceAveragedField<Layout, decltype(model.state.B(Component::X)), decltype(ExactHall3D::bx), Direction::X>(
                    patchLayout, model.state.B(Component::X), ExactHall3D::bx);
                fillFaceAveragedField<Layout, decltype(model.state.B(Component::Y)), decltype(ExactHall3D::by), Direction::Y>(
                    patchLayout, model.state.B(Component::Y), ExactHall3D::by);
                fillFaceAveragedField<Layout, decltype(model.state.B(Component::Z)), decltype(ExactHall3D::bz), Direction::Z>(
                    patchLayout, model.state.B(Component::Z), ExactHall3D::bz);
                
                // J is filled by Ampere now, not initialized
                fillUsableField(patchLayout, model.state.J(Component::X), [](double x, double y, double z) {
                    return ExactHall3D::current(x, y, z)[0];
                });
                fillUsableField(patchLayout, model.state.J(Component::Y), [](double x, double y, double z) {
                    return ExactHall3D::current(x, y, z)[1];
                });
                fillUsableField(patchLayout, model.state.J(Component::Z), [](double x, double y, double z) {
                    return ExactHall3D::current(x, y, z)[2];
                });
            }

            computeFluxes(model, model.state, fluxes, bc, *level, 0.0);
        }

        for (auto& patch : *level)
        {
            auto guard = model.resourcesManager->setOnPatch(*patch, model.state, fluxes);
            auto patchLayout = PHARE::amr::layoutFromPatch<Layout>(*patch);
            fillFluxGhosts(patchLayout, fluxes);
            periodicFillGhostsVec(patchLayout, model.state.E);

            // Use face-averaged comparisons for integral fluxes (after point_value_fluxes_to_integral)
            push_error("rho_fx", l2FaceAveragedFluxError<Layout, Direction::X>(patchLayout, fluxes.rho_fx, [](double x, double y, double z) {
                return ExactHall3D::flux(Direction::X, x, y, z)[0];
            }));
            push_error("rho_fy", l2FaceAveragedFluxError<Layout, Direction::Y>(patchLayout, fluxes.rho_fy, [](double x, double y, double z) {
                return ExactHall3D::flux(Direction::Y, x, y, z)[0];
            }));
            push_error("rho_fz", l2FaceAveragedFluxError<Layout, Direction::Z>(patchLayout, fluxes.rho_fz, [](double x, double y, double z) {
                return ExactHall3D::flux(Direction::Z, x, y, z)[0];
            }));
            for (int c = 0; c < 3; ++c)
            {
                auto comp = static_cast<Component>(c);
                push_error("rhoV_fx_" + std::to_string(c),
                           l2FaceAveragedFluxError<Layout, Direction::X>(patchLayout, fluxes.rhoV_fx(comp), [c](double x, double y, double z) {
                               return ExactHall3D::flux(Direction::X, x, y, z)[1 + c];
                           }));
                push_error("rhoV_fy_" + std::to_string(c),
                           l2FaceAveragedFluxError<Layout, Direction::Y>(patchLayout, fluxes.rhoV_fy(comp), [c](double x, double y, double z) {
                               return ExactHall3D::flux(Direction::Y, x, y, z)[1 + c];
                           }));
                push_error("rhoV_fz_" + std::to_string(c),
                           l2FaceAveragedFluxError<Layout, Direction::Z>(patchLayout, fluxes.rhoV_fz(comp), [c](double x, double y, double z) {
                               return ExactHall3D::flux(Direction::Z, x, y, z)[1 + c];
                           }));
            }
            push_error("Etot_fx", l2FaceAveragedFluxError<Layout, Direction::X>(patchLayout, fluxes.Etot_fx, [](double x, double y, double z) {
                return ExactHall3D::flux(Direction::X, x, y, z)[4];
            }));
            push_error("Etot_fy", l2FaceAveragedFluxError<Layout, Direction::Y>(patchLayout, fluxes.Etot_fy, [](double x, double y, double z) {
                return ExactHall3D::flux(Direction::Y, x, y, z)[4];
            }));
            push_error("Etot_fz", l2FaceAveragedFluxError<Layout, Direction::Z>(patchLayout, fluxes.Etot_fz, [](double x, double y, double z) {
                return ExactHall3D::flux(Direction::Z, x, y, z)[4];
            }));
            // E-field is edge-centered, use edge-averaged comparison
            push_error("Ex", l2EdgeAveragedError<Layout, Direction::X>(patchLayout, model.state.E(Component::X), [](double x, double y, double z) {
                return ExactHall3D::electric(x, y, z)[0];
            }));
            push_error("Ey", l2EdgeAveragedError<Layout, Direction::Y>(patchLayout, model.state.E(Component::Y), [](double x, double y, double z) {
                return ExactHall3D::electric(x, y, z)[1];
            }));
            push_error("Ez", l2EdgeAveragedError<Layout, Direction::Z>(patchLayout, model.state.E(Component::Z), [](double x, double y, double z) {
                return ExactHall3D::electric(x, y, z)[2];
            }));
        }
    }

    return std::make_tuple(nCells, errors);
}
} // namespace

TEST(HallConvergence, FullComputeFluxHall3DPeriodic)
{
    std::cout << "\n=== TEST 10: full ComputeFluxes Hall 3D periodic ===" << std::endl;
    auto [nCells, errors] = runFullFluxConvergence();

    std::cout << "  Observed orders (full flux machinery):" << std::endl;
    for (auto const& [name, err] : errors)
    {
        ASSERT_EQ(err.size(), nCells.size());
        std::cout << "    " << name << ": ";
        for (std::size_t i = 1; i < err.size(); ++i)
        {
            auto ord = convergenceOrder(err[i - 1], err[i]);
            std::cout << std::fixed << std::setprecision(2) << ord
                      << (i + 1 < err.size() ? ", " : "");
            EXPECT_GT(ord, 1.75) << "Insufficient convergence for " << name;
        }
        std::cout << std::endl;
    }
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
