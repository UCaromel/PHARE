/**
 * @file test_instrumented_flux_pipeline.cpp
 * 
 * INSTRUMENTED PIPELINE CONVERGENCE TEST
 * 
 * This test instruments the ComputeFluxes pipeline to track convergence order
 * at each stage. Goal: Find exactly where spatial convergence degrades.
 * 
 * Pipeline stages tested:
 * 1. Initial state (rho, V, B, P) - should be exact (machine precision)
 * 2. Primitive to conservative conversion
 * 3. After point-value conversion (ToPointValue)
 * 4. After flux computation (Godunov)
 * 5. After CT (electric field)
 * 6. Final fluxes
 * 
 * Each stage is tested for convergence order to isolate the bottleneck.
 */

#include "gtest/gtest.h"

#include <iomanip>
#include <iostream>

#include "convergence_test_framework.hpp"
#include "exact_solutions.hpp"
#include "hall_convergence_test_common.hpp"

#include "amr/resources_manager/amr_utils.hpp"
#include "amr/solvers/solver_mhd_model_view.hpp"
#include "amr/solvers/time_integrator/compute_fluxes.hpp"

#include <SAMRAI/tbox/SAMRAIManager.h>
#include <SAMRAI/tbox/SAMRAI_MPI.h>

using namespace PHARE::core;
// Do not use 'using namespace PHARE::test' to avoid ExactHall3D collision
using PHARE::test::MultiQuantityConvergenceStudy;
using PHARE::test::computeFieldError;
using PHARE::test::ExactIdealMHD3D;

namespace
{

// Test both ideal and Hall MHD
template<bool EnableHall, typename MHDModelT>
struct FVMethod3D
{
    template<typename GridLayoutT>
    using type = Godunov<GridLayoutT, MHDModelT, WENOZReconstruction, Rusanov<true>,
                         MHDEquations<EnableHall, false, false>>;
};

/**
 * @brief Run instrumented pipeline test
 * 
 * @tparam EnableHall If true, test Hall MHD; if false, test ideal MHD
 */
template<bool EnableHall>
void runInstrumentedPipelineTest()
{
    using Layout = GridLayout<GridLayoutImplYeeMHD<3, 2>>;
    using Array3D = NdArrayVector<3>;
    using Grid3D = Grid<Array3D, MHDQuantity::Scalar>;
    using Field3D = Field<3, MHDQuantity::Scalar>;
    using VecField3D = VecField<Field3D, MHDQuantity>;
    using ResourcesManagerT = PHARE::amr::ResourcesManager<Layout, Grid3D>;
    using MHDModelT = PHARE::solver::MHDModel<Layout, VecField3D, PHARE::amr::SAMRAI_Types, Grid3D>;
    using FluxesT = AllFluxes<Field3D, VecField3D>;
    using ComputeFluxesT = PHARE::solver::ComputeFluxes<
        FVMethod3D<EnableHall, MHDModelT>::template type, MHDModelT>;

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "INSTRUMENTED PIPELINE: " << (EnableHall ? "HALL MHD" : "IDEAL MHD") << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // Use our new framework
    MultiQuantityConvergenceStudy study;
    
    // Stage 1: Initial state
    study.addQuantity("Stage1_rho_initial");
    study.addQuantity("Stage1_V_initial");
    study.addQuantity("Stage1_B_initial");
    study.addQuantity("Stage1_P_initial");
    
    // Stage 2: Fluxes by direction (to check directional bias)
    study.addQuantity("Stage2_flux_rho_X");
    study.addQuantity("Stage2_flux_rho_Y");
    study.addQuantity("Stage2_flux_rho_Z");
    study.addQuantity("Stage2_flux_Etot_X");
    study.addQuantity("Stage2_flux_Etot_Y");
    study.addQuantity("Stage2_flux_Etot_Z");
    
    // Stage 3: E-field (computed by CT)
    study.addQuantity("Stage3_E_X");
    study.addQuantity("Stage3_E_Y");
    study.addQuantity("Stage3_E_Z");

    std::vector<int> grid_sizes = {16, 32, 64};

    for (auto n : grid_sizes)
    {
        std::cout << "\n--- Testing resolution: " << n << "³ ---" << std::endl;
        
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

        // === STAGE 1: Initialize state with exact solution ===
        std::cout << "  Stage 1: Initializing state..." << std::endl;
        
        for (auto& patch : *level)
        {
            auto guard = model.resourcesManager->setOnPatch(*patch, model.state, fluxes);
            auto patchLayout = PHARE::amr::layoutFromPatch<Layout>(*patch);

            // Fill with exact solution
            fillUsableField(patchLayout, model.state.rho, ExactHall3D::rho);
            fillUsableField(patchLayout, model.state.V(Component::X), ExactHall3D::vx);
            fillUsableField(patchLayout, model.state.V(Component::Y), ExactHall3D::vy);
            fillUsableField(patchLayout, model.state.V(Component::Z), ExactHall3D::vz);
            fillUsableField(patchLayout, model.state.B(Component::X), ExactHall3D::bx);
            fillUsableField(patchLayout, model.state.B(Component::Y), ExactHall3D::by);
            fillUsableField(patchLayout, model.state.B(Component::Z), ExactHall3D::bz);
            fillUsableField(patchLayout, model.state.P, ExactHall3D::pressure);
            fillUsableField(patchLayout, model.state.rhoV(Component::X), [](double x, double y, double z) {
                return ExactHall3D::rho(x, y, z) * ExactHall3D::vx(x, y, z);
            });
            fillUsableField(patchLayout, model.state.rhoV(Component::Y), [](double x, double y, double z) {
                return ExactHall3D::rho(x, y, z) * ExactHall3D::vy(x, y, z);
            });
            fillUsableField(patchLayout, model.state.rhoV(Component::Z), [](double x, double y, double z) {
                return ExactHall3D::rho(x, y, z) * ExactHall3D::vz(x, y, z);
            });
            fillUsableField(patchLayout, model.state.Etot, ExactHall3D::etot);
            
            // Check initial state accuracy (should be machine precision)
            auto norms_rho = computeFieldError(patchLayout, model.state.rho, ExactHall3D::rho, 6);
            auto norms_V = computeFieldError(patchLayout, model.state.V(Component::X), ExactHall3D::vx, 6);
            auto norms_B = computeFieldError(patchLayout, model.state.B(Component::X), ExactHall3D::bx, 6);
            auto norms_P = computeFieldError(patchLayout, model.state.P, ExactHall3D::pressure, 6);
            
            study.recordError("Stage1_rho_initial", n, norms_rho.l2);
            study.recordError("Stage1_V_initial", n, norms_V.l2);
            study.recordError("Stage1_B_initial", n, norms_B.l2);
            study.recordError("Stage1_P_initial", n, norms_P.l2);
            
            std::cout << "    Initial state L2 errors: rho=" << std::scientific << norms_rho.l2
                      << ", V=" << norms_V.l2 << ", B=" << norms_B.l2 << ", P=" << norms_P.l2 << std::endl;
        }

        // === STAGE 2: Compute fluxes ===
        std::cout << "  Stage 2: Computing fluxes..." << std::endl;
        computeFluxes(model, model.state, fluxes, bc, *level, 0.0);

        // Check flux accuracy by direction
        for (auto& patch : *level)
        {
            auto guard = model.resourcesManager->setOnPatch(*patch, model.state, fluxes);
            auto patchLayout = PHARE::amr::layoutFromPatch<Layout>(*patch);
            fillFluxGhosts(patchLayout, fluxes);

            // Mass fluxes by direction
            auto err_rho_x = l2FluxError(patchLayout, fluxes.rho_fx, [](double x, double y, double z) {
                if constexpr (EnableHall)
                    return ExactHall3D::flux(Direction::X, x, y, z)[0];
                else
                    return ExactIdealMHD3D::flux(Direction::X, x, y, z)[0];
            });
            auto err_rho_y = l2FluxError(patchLayout, fluxes.rho_fy, [](double x, double y, double z) {
                if constexpr (EnableHall)
                    return ExactHall3D::flux(Direction::Y, x, y, z)[0];
                else
                    return ExactIdealMHD3D::flux(Direction::Y, x, y, z)[0];
            });
            auto err_rho_z = l2FluxError(patchLayout, fluxes.rho_fz, [](double x, double y, double z) {
                if constexpr (EnableHall)
                    return ExactHall3D::flux(Direction::Z, x, y, z)[0];
                else
                    return ExactIdealMHD3D::flux(Direction::Z, x, y, z)[0];
            });
            
            study.recordError("Stage2_flux_rho_X", n, err_rho_x);
            study.recordError("Stage2_flux_rho_Y", n, err_rho_y);
            study.recordError("Stage2_flux_rho_Z", n, err_rho_z);
            
            // Energy fluxes by direction
            auto err_etot_x = l2FluxError(patchLayout, fluxes.Etot_fx, [](double x, double y, double z) {
                if constexpr (EnableHall)
                    return ExactHall3D::flux(Direction::X, x, y, z)[4];
                else
                    return ExactIdealMHD3D::flux(Direction::X, x, y, z)[4];
            });
            auto err_etot_y = l2FluxError(patchLayout, fluxes.Etot_fy, [](double x, double y, double z) {
                if constexpr (EnableHall)
                    return ExactHall3D::flux(Direction::Y, x, y, z)[4];
                else
                    return ExactIdealMHD3D::flux(Direction::Y, x, y, z)[4];
            });
            auto err_etot_z = l2FluxError(patchLayout, fluxes.Etot_fz, [](double x, double y, double z) {
                if constexpr (EnableHall)
                    return ExactHall3D::flux(Direction::Z, x, y, z)[4];
                else
                    return ExactIdealMHD3D::flux(Direction::Z, x, y, z)[4];
            });
            
            study.recordError("Stage2_flux_Etot_X", n, err_etot_x);
            study.recordError("Stage2_flux_Etot_Y", n, err_etot_y);
            study.recordError("Stage2_flux_Etot_Z", n, err_etot_z);
            
            std::cout << "    Flux errors (rho): X=" << std::scientific << err_rho_x
                      << ", Y=" << err_rho_y << ", Z=" << err_rho_z << std::endl;
            std::cout << "    Flux errors (Etot): X=" << err_etot_x
                      << ", Y=" << err_etot_y << ", Z=" << err_etot_z << std::endl;
        }

        // === STAGE 3: Check E-field ===
        std::cout << "  Stage 3: Checking E-field..." << std::endl;
        
        for (auto& patch : *level)
        {
            auto guard = model.resourcesManager->setOnPatch(*patch, model.state);
            auto patchLayout = PHARE::amr::layoutFromPatch<Layout>(*patch);
            periodicFillGhostsVec(patchLayout, model.state.E);

            auto err_ex = l2FluxError(patchLayout, model.state.E(Component::X), [](double x, double y, double z) {
                if constexpr (EnableHall)
                    return ExactHall3D::electric(x, y, z)[0];
                else
                    return ExactIdealMHD3D::electric(x, y, z)[0];
            });
            auto err_ey = l2FluxError(patchLayout, model.state.E(Component::Y), [](double x, double y, double z) {
                if constexpr (EnableHall)
                    return ExactHall3D::electric(x, y, z)[1];
                else
                    return ExactIdealMHD3D::electric(x, y, z)[1];
            });
            auto err_ez = l2FluxError(patchLayout, model.state.E(Component::Z), [](double x, double y, double z) {
                if constexpr (EnableHall)
                    return ExactHall3D::electric(x, y, z)[2];
                else
                    return ExactIdealMHD3D::electric(x, y, z)[2];
            });
            
            study.recordError("Stage3_E_X", n, err_ex);
            study.recordError("Stage3_E_Y", n, err_ey);
            study.recordError("Stage3_E_Z", n, err_ez);
            
            std::cout << "    E-field errors: Ex=" << std::scientific << err_ex
                      << ", Ey=" << err_ey << ", Ez=" << err_ez << std::endl;
        }
    }

    // === ANALYZE RESULTS ===
    study.computeOrders();
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "CONVERGENCE ORDER ANALYSIS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    study.printSummary();
    
    // Identify problem areas
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "DIAGNOSTICS:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    auto failures = study.getFailures(3.5);
    if (failures.empty())
    {
        std::cout << "✅ ALL STAGES PASS (order > 3.5)" << std::endl;
    }
    else
    {
        std::cout << "❌ FAILING STAGES (order < 3.5):" << std::endl;
        for (auto const& name : failures)
        {
            auto const& result = study.getResult(name);
            std::cout << "  - " << name << ": min order = " 
                      << std::fixed << std::setprecision(2) << result.min_order() << std::endl;
        }
    }
    
    // Check directional bias
    std::cout << "\nDIRECTIONAL BIAS CHECK:" << std::endl;
    auto rho_x_order = study.getResult("Stage2_flux_rho_X").avg_order();
    auto rho_y_order = study.getResult("Stage2_flux_rho_Y").avg_order();
    auto rho_z_order = study.getResult("Stage2_flux_rho_Z").avg_order();
    
    std::cout << "  Mass flux convergence: X=" << rho_x_order 
              << ", Y=" << rho_y_order << ", Z=" << rho_z_order << std::endl;
    
    if (std::abs(rho_y_order - rho_x_order) > 0.5 || std::abs(rho_y_order - rho_z_order) > 0.5)
    {
        std::cout << "  ⚠️  SIGNIFICANT DIRECTIONAL BIAS DETECTED!" << std::endl;
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
