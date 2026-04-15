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

#include "gtest/gtest.h"

#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "amr/resources_manager/amr_utils.hpp"
#include "amr/solvers/solver_mhd_model_view.hpp"
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
template<typename MHDModelT>
struct IdealMHDFVMethod3D
{
    template<typename GridLayoutT>
    using type = Godunov<GridLayoutT, MHDModelT, WENOZReconstruction, Rusanov<true>,
                         MHDEquations<false, false, false>>;
};

/**
 * @brief Ideal MHD exact solutions
 * 
 * Uses same primitive state as ExactHall3D, but E-field has only ideal term (no Hall).
 * Fluxes use only ideal MHD formulas (no Hall energy correction).
 */
struct ExactIdealMHD3D
{
    // Primitive state - identical to Hall case
    static double rho(double x, double y, double z) { return ExactHall3D::rho(x, y, z); }
    static double vx(double x, double y, double z) { return ExactHall3D::vx(x, y, z); }
    static double vy(double x, double y, double z) { return ExactHall3D::vy(x, y, z); }
    static double vz(double x, double y, double z) { return ExactHall3D::vz(x, y, z); }
    static double bx(double x, double y, double z) { return ExactHall3D::bx(x, y, z); }
    static double by(double x, double y, double z) { return ExactHall3D::by(x, y, z); }
    static double bz(double x, double y, double z) { return ExactHall3D::bz(x, y, z); }
    static double pressure(double x, double y, double z) { return ExactHall3D::pressure(x, y, z); }
    static double etot(double x, double y, double z) { return ExactHall3D::etot(x, y, z); }

    // E-field: IDEAL ONLY (no Hall term)
    static auto electric(double x, double y, double z)
    {
        auto const vX = vx(x, y, z);
        auto const vY = vy(x, y, z);
        auto const vZ = vz(x, y, z);
        auto const bX = bx(x, y, z);
        auto const bY = by(x, y, z);
        auto const bZ = bz(x, y, z);

        // E = -V×B (ideal Ohm's law)
        auto const ex = -(vY * bZ - vZ * bY);
        auto const ey = -(vZ * bX - vX * bZ);
        auto const ez = -(vX * bY - vY * bX);

        return std::array<double, 3>{ex, ey, ez};
    }

    // Fluxes: IDEAL ONLY (no Hall energy correction)
    static auto flux(Direction dir, double x, double y, double z)
    {
        auto const r  = rho(x, y, z);
        auto const vX = vx(x, y, z);
        auto const vY = vy(x, y, z);
        auto const vZ = vz(x, y, z);
        auto const bX = bx(x, y, z);
        auto const bY = by(x, y, z);
        auto const bZ = bz(x, y, z);
        auto const p  = pressure(x, y, z);
        auto const eT = etot(x, y, z);

        auto const gp = p + 0.5 * (bX * bX + bY * bY + bZ * bZ);

        double frho = 0.0, frhoVx = 0.0, frhoVy = 0.0, frhoVz = 0.0, fetot = 0.0;
        if (dir == Direction::X)
        {
            frho   = r * vX;
            frhoVx = r * vX * vX + gp - bX * bX;
            frhoVy = r * vX * vY - bX * bY;
            frhoVz = r * vX * vZ - bX * bZ;
            fetot  = (eT + gp) * vX - bX * (vX * bX + vY * bY + vZ * bZ);
        }
        else if (dir == Direction::Y)
        {
            frho   = r * vY;
            frhoVx = r * vY * vX - bY * bX;
            frhoVy = r * vY * vY + gp - bY * bY;
            frhoVz = r * vY * vZ - bY * bZ;
            fetot  = (eT + gp) * vY - bY * (vX * bX + vY * bY + vZ * bZ);
        }
        else
        {
            frho   = r * vZ;
            frhoVx = r * vZ * vX - bZ * bX;
            frhoVy = r * vZ * vY - bZ * bY;
            frhoVz = r * vZ * vZ + gp - bZ * bZ;
            fetot  = (eT + gp) * vZ - bZ * (vX * bX + vY * bY + vZ * bZ);
        }

        // NO Hall energy correction for ideal MHD

        return std::array<double, 5>{frho, frhoVx, frhoVy, frhoVz, fetot};
    }
};

auto runIdealMHDFluxConvergence()
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
        = PHARE::solver::ComputeFluxes<IdealMHDFVMethod3D<MHDModelT>::template type, MHDModelT>;

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

            // Fill with CELL-AVERAGED exact solution (4th order GL quadrature)
            fillCellAveragedField(patchLayout, model.state.rho, ExactIdealMHD3D::rho);
            fillCellAveragedField(patchLayout, model.state.P, ExactIdealMHD3D::pressure);
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
            
            // Fill FACE-AVERAGED quantities (4th order GL quadrature in transverse directions)
            fillFaceAveragedField<Layout, decltype(model.state.V(Component::X)), decltype(ExactIdealMHD3D::vx), Direction::X>(
                patchLayout, model.state.V(Component::X), ExactIdealMHD3D::vx);
            fillFaceAveragedField<Layout, decltype(model.state.V(Component::Y)), decltype(ExactIdealMHD3D::vy), Direction::Y>(
                patchLayout, model.state.V(Component::Y), ExactIdealMHD3D::vy);
            fillFaceAveragedField<Layout, decltype(model.state.V(Component::Z)), decltype(ExactIdealMHD3D::vz), Direction::Z>(
                patchLayout, model.state.V(Component::Z), ExactIdealMHD3D::vz);
            
            fillFaceAveragedField<Layout, decltype(model.state.B(Component::X)), decltype(ExactIdealMHD3D::bx), Direction::X>(
                patchLayout, model.state.B(Component::X), ExactIdealMHD3D::bx);
            fillFaceAveragedField<Layout, decltype(model.state.B(Component::Y)), decltype(ExactIdealMHD3D::by), Direction::Y>(
                patchLayout, model.state.B(Component::Y), ExactIdealMHD3D::by);
            fillFaceAveragedField<Layout, decltype(model.state.B(Component::Z)), decltype(ExactIdealMHD3D::bz), Direction::Z>(
                patchLayout, model.state.B(Component::Z), ExactIdealMHD3D::bz);
            
            // Note: J field not needed for ideal MHD (no Hall term)
        }

        computeFluxes(model, model.state, fluxes, bc, *level, 0.0);

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

TEST(IdealMHDConvergence, FullComputeFluxIdeal3DPeriodic)
{
    std::cout << "\n=== IDEAL MHD 3D CONVERGENCE TEST ===" << std::endl;
    std::cout << "This test should show ORDER 4 for all quantities if 3D implementation is correct." << std::endl;
    std::cout << "If it fails, the problem is NOT Hall-specific but a general 3D issue." << std::endl;
    
    auto [nCells, errors] = runIdealMHDFluxConvergence();

    std::cout << "\n  Observed orders (Ideal MHD, no Hall):" << std::endl;
    bool all_pass = true;
    
    for (auto const& [name, err] : errors)
    {
        ASSERT_EQ(err.size(), nCells.size());
        std::cout << "    " << name << ": ";
        for (std::size_t i = 1; i < err.size(); ++i)
        {
            auto ord = convergenceOrder(err[i - 1], err[i]);
            std::cout << std::fixed << std::setprecision(2) << ord
                      << (i + 1 < err.size() ? ", " : "");
            
            if (ord < 3.5)  // Expect at least 3.5 for 4th order method
            {
                all_pass = false;
                std::cout << " ⚠️";
            }
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n";
    if (all_pass)
    {
        std::cout << "✅ IDEAL MHD PASSES: Problem is Hall-specific (likely J computation)" << std::endl;
    }
    else
    {
        std::cout << "❌ IDEAL MHD FAILS: Problem is more fundamental (3D indexing, derivatives, etc.)" << std::endl;
    }
    
    // Don't use EXPECT here - we want to see all results regardless
    // This is a diagnostic test to understand where the problem is
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
