/**
 * @file test_decomposed_flux_pipeline.cpp
 * 
 * DECOMPOSED PIPELINE CONVERGENCE TEST
 * 
 * This test manually calls each sub-component of the ComputeFluxes pipeline
 * and checks convergence order after each step. This definitively identifies
 * which operator causes convergence degradation.
 * 
 * Pipeline decomposition:
 * 1. Initial state (rho, V, B, P) - should be exact
 * 2. Ampere (J = curl B) - if Hall/Resistivity
 * 3. ToPointValue (cell-average → point-value)
 * 4. ToPrimitive (conservative → primitive)
 * 5. FVM/Godunov (flux computation) ← PRIMARY SUSPECT
 * 6. CT (constrained transport, E-field)
 * 7. ToIntegral (point-value fluxes → cell-average)
 * 
 * Each step is tested for spatial convergence to isolate the bottleneck.
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
using PHARE::test::MultiQuantityConvergenceStudy;
using PHARE::test::computeFieldError;
using PHARE::test::ExactIdealMHD3D;
using PHARE::test::fillCellAveragedField;
using PHARE::test::fillFaceAveragedField;
using PHARE::test::l2FaceAveragedFluxError;
using PHARE::test::l2EdgeAveragedError;

namespace
{

// Helper function for Ampere (from test_hall_compute_flux_convergence.cpp)
template<typename Layout, typename MHDModelT>
void computeSecondOrderAmpere(MHDModelT& model, typename MHDModelT::state_type& state,
                              typename MHDModelT::level_t& level)
{
    for (auto const& patch : level)
    {
        auto guard  = model.resourcesManager->setOnPatch(*patch, state);
        auto layout = PHARE::amr::layoutFromPatch<Layout>(*patch);

        auto const& Bx = state.B(Component::X);
        auto const& By = state.B(Component::Y);
        auto const& Bz = state.B(Component::Z);
        auto& Jx       = state.J(Component::X);
        auto& Jy       = state.J(Component::Y);
        auto& Jz       = state.J(Component::Z);

        fillUsableField(layout, Jx, [](double x, double y, double z) {
            return ExactHall3D::current(x, y, z)[0];
        });
        fillUsableField(layout, Jy, [](double x, double y, double z) {
            return ExactHall3D::current(x, y, z)[1];
        });
        fillUsableField(layout, Jz, [](double x, double y, double z) {
            return ExactHall3D::current(x, y, z)[2];
        });

        auto const& cJx = layout.centering(MHDQuantity::Scalar::Jx);
        auto const& cJy = layout.centering(MHDQuantity::Scalar::Jy);
        auto const& cJz = layout.centering(MHDQuantity::Scalar::Jz);

        for (auto i = layout.ghostStartIndex(cJx[0], Direction::X);
             i <= layout.ghostEndIndex(cJx[0], Direction::X); ++i)
            for (auto j = layout.ghostStartIndex(cJx[1], Direction::Y) + 1;
                 j <= layout.ghostEndIndex(cJx[1], Direction::Y) - 1; ++j)
                for (auto k = layout.ghostStartIndex(cJx[2], Direction::Z) + 1;
                     k <= layout.ghostEndIndex(cJx[2], Direction::Z) - 1; ++k)
                {
                    MeshIndex<3> idx{i, j, k};
                    Jx(i, j, k)
                        = layout.template deriv<Direction::Z, 2>(By, idx)
                          - layout.template deriv<Direction::Y, 2>(Bz, idx);
                }

        for (auto i = layout.ghostStartIndex(cJy[0], Direction::X) + 1;
             i <= layout.ghostEndIndex(cJy[0], Direction::X) - 1; ++i)
            for (auto j = layout.ghostStartIndex(cJy[1], Direction::Y);
                 j <= layout.ghostEndIndex(cJy[1], Direction::Y); ++j)
                for (auto k = layout.ghostStartIndex(cJy[2], Direction::Z) + 1;
                     k <= layout.ghostEndIndex(cJy[2], Direction::Z) - 1; ++k)
                {
                    MeshIndex<3> idx{i, j, k};
                    Jy(i, j, k)
                        = layout.template deriv<Direction::X, 2>(Bz, idx)
                          - layout.template deriv<Direction::Z, 2>(Bx, idx);
                }

        for (auto i = layout.ghostStartIndex(cJz[0], Direction::X) + 1;
             i <= layout.ghostEndIndex(cJz[0], Direction::X) - 1; ++i)
            for (auto j = layout.ghostStartIndex(cJz[1], Direction::Y) + 1;
                 j <= layout.ghostEndIndex(cJz[1], Direction::Y) - 1; ++j)
                for (auto k = layout.ghostStartIndex(cJz[2], Direction::Z);
                     k <= layout.ghostEndIndex(cJz[2], Direction::Z); ++k)
                {
                    MeshIndex<3> idx{i, j, k};
                    Jz(i, j, k)
                        = layout.template deriv<Direction::X, 2>(By, idx)
                          - layout.template deriv<Direction::Y, 2>(Bx, idx);
                }
    }
}

// Test both ideal and Hall MHD
template<bool EnableHall, typename MHDModelT>
struct FVMethod3D
{
    template<typename GridLayoutT>
    using type = Godunov<GridLayoutT, MHDModelT, WENOZReconstruction, Rusanov<true>,
                         MHDEquations<EnableHall, false, false>>;
};

/**
 * @brief Run decomposed pipeline test
 * 
 * Calls each pipeline component individually and checks convergence.
 * 
 * @tparam EnableHall If true, test Hall MHD; if false, test ideal MHD
 */
template<bool EnableHall>
void runDecomposedPipelineTest()
{
    using Layout = GridLayout<GridLayoutImplYeeMHD<3, 2>>;
    using Array3D = NdArrayVector<3>;
    using Grid3D = Grid<Array3D, MHDQuantity::Scalar>;
    using Field3D = Field<3, MHDQuantity::Scalar>;
    using VecField3D = VecField<Field3D, MHDQuantity>;
    using ResourcesManagerT = PHARE::amr::ResourcesManager<Layout, Grid3D>;
    using MHDModelT = PHARE::solver::MHDModel<Layout, VecField3D, PHARE::amr::SAMRAI_Types, Grid3D>;
    using FluxesT = AllFluxes<Field3D, VecField3D>;
    
    // Individual pipeline components
    using DispatchersT = PHARE::solver::Dispatchers<Layout>;
    using ToPointValueT = typename DispatchersT::template ToPointValue_t<MHDModelT>;
    using ToPrimitiveT = typename DispatchersT::ToPrimitiveConverter_t;
    using ToConservativeT = typename DispatchersT::ToConservativeConverter_t;
    using AmpereT = typename DispatchersT::Ampere_t;
    using FVMethodT = typename DispatchersT::template FVMethod_t<FVMethod3D<EnableHall, MHDModelT>::template type>;
    using CTT = typename DispatchersT::template ConstrainedTransport_t<
        MHDModelT, FVMethodT::template Rec, EnableHall, false, false>;

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "DECOMPOSED PIPELINE: " << (EnableHall ? "HALL MHD" : "IDEAL MHD") << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    MultiQuantityConvergenceStudy study;
    
    // Define all quantities we'll track
    study.addQuantity("Step0_rho_initial");
    study.addQuantity("Step0_V_initial");
    study.addQuantity("Step0_B_initial");
    study.addQuantity("Step0_P_initial");
    
    study.addQuantity("Step1_rho_after_point_value");
    study.addQuantity("Step1_V_after_point_value");
    study.addQuantity("Step1_B_after_point_value");
    
    if constexpr (EnableHall)
    {
        study.addQuantity("Step2_J_after_ampere");
    }
    
    study.addQuantity("Step3_rho_after_primitive");
    study.addQuantity("Step3_V_after_primitive");
    study.addQuantity("Step3_B_after_primitive");
    study.addQuantity("Step3_P_after_primitive");
    
    study.addQuantity("Step4_flux_rho_X");
    study.addQuantity("Step4_flux_rho_Y");
    study.addQuantity("Step4_flux_rho_Z");
    study.addQuantity("Step4_flux_Etot_X");
    study.addQuantity("Step4_flux_Etot_Y");
    study.addQuantity("Step4_flux_Etot_Z");
    
    study.addQuantity("Step5_E_after_CT");
    
    study.addQuantity("Step6_integral_flux_rho_X");
    study.addQuantity("Step6_integral_flux_rho_Y");
    study.addQuantity("Step6_integral_flux_rho_Z");
    study.addQuantity("Step6_integral_flux_Etot_X");
    study.addQuantity("Step6_integral_flux_Etot_Y");
    study.addQuantity("Step6_integral_flux_Etot_Z");
    
    std::vector<int> nCells = {16, 32, 64};

    for (int n : nCells)
    {
        std::cout << "\n--- Testing resolution: " << n << "³ ---" << std::endl;

        auto hierarchy = makePeriodicHierarchy3D(n);
        auto level = hierarchy->getPatchLevel(0);
        auto resman = std::make_shared<ResourcesManagerT>();
        auto modelDict = makeHall3DMHDModelDict();
        auto fluxDict = makeHall3DComputeFluxDict();
        MHDModelT model{modelDict, resman};

        FluxesT fluxes{
                       {"test_rho_fx", MHDQuantity::Scalar::ScalarFlux_x},
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

        // Create individual pipeline components
        AmpereT ampere{};
        ToPointValueT point_value{};
        ToPrimitiveT to_primitive{fluxDict["to_primitive"]};
        ToConservativeT to_conservative{fluxDict["to_conservative"]};
        FVMethodT fvm{fluxDict["fv_method"]};
        CTT ct{fluxDict["constrained_transport"]};

        // Register and allocate resources
        ct.constrained_transport_.registerResources(model);
        fvm.finite_volume_method_.registerResources(model);
        point_value.to_point_value_.registerResources(model);

        for (auto& patch : *level)
        {
            model.allocate(*patch, 0.0);
            ct.constrained_transport_.allocate(model, *patch, 0.0);
            fvm.finite_volume_method_.allocate(model, *patch, 0.0);
            point_value.to_point_value_.allocate(model, *patch, 0.0);
            model.resourcesManager->allocate(fluxes, *patch, 0.0);
        }

        // ========================================================================
        // STEP 0: Initialize with exact solution
        // ========================================================================
        std::cout << "  Step 0: Initializing state..." << std::endl;
        
        for (auto& patch : *level)
        {
            auto guard = model.resourcesManager->setOnPatch(*patch, model.state, fluxes);
            auto patchLayout = PHARE::amr::layoutFromPatch<Layout>(*patch);

            // Fill with CELL-AVERAGED exact solution (not point-values!)
            // Cell-centered quantities (rho, P, Etot, rhoV): use cell-average
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
            
            // V is cell-centered (only B, E, J are on Yee grid)
            fillCellAveragedField(patchLayout, model.state.V(Component::X), ExactHall3D::vx);
            fillCellAveragedField(patchLayout, model.state.V(Component::Y), ExactHall3D::vy);
            fillCellAveragedField(patchLayout, model.state.V(Component::Z), ExactHall3D::vz);
            
            // B is face-centered (electromagnetic on Yee grid)
            fillFaceAveragedField<Layout, decltype(model.state.B(Component::X)), decltype(ExactHall3D::bx), Direction::X>(
                patchLayout, model.state.B(Component::X), ExactHall3D::bx);
            fillFaceAveragedField<Layout, decltype(model.state.B(Component::Y)), decltype(ExactHall3D::by), Direction::Y>(
                patchLayout, model.state.B(Component::Y), ExactHall3D::by);
            fillFaceAveragedField<Layout, decltype(model.state.B(Component::Z)), decltype(ExactHall3D::bz), Direction::Z>(
                patchLayout, model.state.B(Component::Z), ExactHall3D::bz);
            
            // Check initial state
            // Cell-centered: compare cell-averaged field vs cell-averaged exact
            auto err_rho = computeFieldError(patchLayout, model.state.rho, ExactHall3D::rho, 6);
            auto err_P = computeFieldError(patchLayout, model.state.P, ExactHall3D::pressure, 6);
            
            // V is also cell-centered
            auto err_V = computeFieldError(patchLayout, model.state.V(Component::X), ExactHall3D::vx, 6);
            
            // B is face-centered: compare face-averaged field vs face-averaged exact
            auto err_Bx = l2FaceAveragedFluxError<Layout, Direction::X>(patchLayout, model.state.B(Component::X), ExactHall3D::bx);
            auto err_By = l2FaceAveragedFluxError<Layout, Direction::Y>(patchLayout, model.state.B(Component::Y), ExactHall3D::by);
            auto err_Bz = l2FaceAveragedFluxError<Layout, Direction::Z>(patchLayout, model.state.B(Component::Z), ExactHall3D::bz);
            auto err_B = std::max({err_Bx, err_By, err_Bz});
            
            study.recordError("Step0_rho_initial", n, err_rho.l2);
            study.recordError("Step0_V_initial", n, err_V.l2);
            study.recordError("Step0_B_initial", n, err_B);
            study.recordError("Step0_P_initial", n, err_P.l2);
            
            std::cout << "    ✓ Initial state errors (cell-avg vs cell-avg): "
                      << "rho=" << std::scientific << err_rho.l2 << ", "
                      << "V=" << err_V.l2 << ", "
                      << "B=" << err_B << ", "
                      << "P=" << err_P.l2 << std::endl;
        }

        // ========================================================================
        // STEP 1: ToPointValue (cell-average → point-value, INCLUDING B)
        // ========================================================================
        std::cout << "  Step 1: Converting to point values..." << std::endl;
        
        point_value(*level, model, 0.0, model.state);
        bc.fillMagneticPointGhosts(point_value.to_point_value_.B, *level, 0.0);
        
        // ========================================================================
        // STEP 2: Ampere (J = curl B) on POINT-VALUE B - only if Hall/Resistivity
        // ========================================================================
        if constexpr (EnableHall)
        {
            std::cout << "  Step 2: Computing J = curl(point-value B) [Ampere]..." << std::endl;
            
            // Ampere now operates on point_value.to_point_value_.B (not face-averaged state.B)
            ampere(*level, model, 0.0, point_value.to_point_value_);
            bc.fillCurrentPointGhosts(point_value.to_point_value_.J, *level, 0.0);
            
            for (auto& patch : *level)
            {
                auto guard = model.resourcesManager->setOnPatch(*patch, point_value.to_point_value_);
                auto patchLayout = PHARE::amr::layoutFromPatch<Layout>(*patch);
                
                // J is already point-value from Ampere on point-value B
                auto err_J = computeFieldError(patchLayout, point_value.to_point_value_.J(Component::X), 
                                              [](double x, double y, double z) {
                                                  return ExactHall3D::current(x, y, z)[0];
                                              }, 6);
                study.recordError("Step2_J_after_ampere", n, err_J.l2);
                
                std::cout << "    ✓ J error after Ampere(point-value B): " << std::scientific << err_J.l2 << std::endl;
            }
        }
        
        for (auto& patch : *level)
        {
            // Include point_value.to_point_value_ in the guard so we can access its fields
            auto guard = model.resourcesManager->setOnPatch(*patch, model.state, point_value.to_point_value_);
            auto patchLayout = PHARE::amr::layoutFromPatch<Layout>(*patch);
            
            // Point-value fields should still match exact solution
            auto err_rho = computeFieldError(patchLayout, point_value.to_point_value_.rho, 
                                            ExactHall3D::rho, 6);
            auto err_V = computeFieldError(patchLayout, point_value.to_point_value_.V(Component::X), 
                                          ExactHall3D::vx, 6);
            auto err_B = computeFieldError(patchLayout, point_value.to_point_value_.B(Component::X), 
                                          ExactHall3D::bx, 6);
            
            study.recordError("Step1_rho_after_point_value", n, err_rho.l2);
            study.recordError("Step1_V_after_point_value", n, err_V.l2);
            study.recordError("Step1_B_after_point_value", n, err_B.l2);
            
            std::cout << "    ✓ Point-value errors: "
                      << "rho=" << std::scientific << err_rho.l2 << ", "
                      << "V=" << err_V.l2 << ", "
                      << "B=" << err_B.l2 << std::endl;
        }

        // ========================================================================
        // STEP 3: ToPrimitive (conservative → primitive)
        // ========================================================================
        std::cout << "  Step 3: Converting to primitive variables..." << std::endl;
        
        to_primitive(*level, model, 0.0, point_value.to_point_value_);
        bc.fillPrimitivePointGhosts(point_value.to_point_value_, *level, 0.0);
        
        for (auto& patch : *level)
        {
            auto guard = model.resourcesManager->setOnPatch(*patch, model.state, point_value.to_point_value_);
            auto patchLayout = PHARE::amr::layoutFromPatch<Layout>(*patch);
            
            auto err_rho = computeFieldError(patchLayout, point_value.to_point_value_.rho, 
                                            ExactHall3D::rho, 6);
            auto err_V = computeFieldError(patchLayout, point_value.to_point_value_.V(Component::X), 
                                          ExactHall3D::vx, 6);
            auto err_B = computeFieldError(patchLayout, point_value.to_point_value_.B(Component::X), 
                                          ExactHall3D::bx, 6);
            auto err_P = computeFieldError(patchLayout, point_value.to_point_value_.P, 
                                          ExactHall3D::pressure, 6);
            
            study.recordError("Step3_rho_after_primitive", n, err_rho.l2);
            study.recordError("Step3_V_after_primitive", n, err_V.l2);
            study.recordError("Step3_B_after_primitive", n, err_B.l2);
            study.recordError("Step3_P_after_primitive", n, err_P.l2);
            
            std::cout << "    ✓ Primitive errors: "
                      << "rho=" << std::scientific << err_rho.l2 << ", "
                      << "V=" << err_V.l2 << ", "
                      << "B=" << err_B.l2 << ", "
                      << "P=" << err_P.l2 << std::endl;
        }

        // ========================================================================
        // STEP 4: FVM/Godunov (flux computation) ← THE SUSPECT!
        // ========================================================================
        std::cout << "  Step 4: Computing fluxes [Godunov]... *** PRIMARY SUSPECT ***" << std::endl;
        
        fvm(*level, model, 0.0, ct.constrained_transport_, point_value.to_point_value_, fluxes);
        
        for (auto& patch : *level)
        {
            auto guard = model.resourcesManager->setOnPatch(*patch, fluxes);
            auto patchLayout = PHARE::amr::layoutFromPatch<Layout>(*patch);
            
            // Check flux errors by direction
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
            
            study.recordError("Step4_flux_rho_X", n, err_rho_x);
            study.recordError("Step4_flux_rho_Y", n, err_rho_y);
            study.recordError("Step4_flux_rho_Z", n, err_rho_z);
            study.recordError("Step4_flux_Etot_X", n, err_etot_x);
            study.recordError("Step4_flux_Etot_Y", n, err_etot_y);
            study.recordError("Step4_flux_Etot_Z", n, err_etot_z);
            
            std::cout << "    ⚠️  Flux errors (rho): "
                      << "X=" << std::scientific << err_rho_x << ", "
                      << "Y=" << err_rho_y << ", "
                      << "Z=" << err_rho_z << std::endl;
            std::cout << "    ⚠️  Flux errors (Etot): "
                      << "X=" << std::scientific << err_etot_x << ", "
                      << "Y=" << err_etot_y << ", "
                      << "Z=" << err_etot_z << std::endl;
        }

        // ========================================================================
        // STEP 5: CT (constrained transport, E-field)
        // ========================================================================
        std::cout << "  Step 5: Computing E-field [CT]..." << std::endl;
        
        ct(*level, model, point_value.to_point_value_, model.state.E);
        bc.fillElectricGhosts(model.state.E, *level, 0.0);
        
        for (auto& patch : *level)
        {
            auto guard = model.resourcesManager->setOnPatch(*patch, model.state);
            auto patchLayout = PHARE::amr::layoutFromPatch<Layout>(*patch);
            
            // CT outputs POINT-VALUE E, so compare against point-value exact (not edge-averaged)
            auto err_E = l2FluxError(patchLayout, model.state.E(Component::X), [](double x, double y, double z) {
                if constexpr (EnableHall)
                    return ExactHall3D::electric(x, y, z)[0];
                else
                    return ExactIdealMHD3D::electric(x, y, z)[0];
            });
            
            study.recordError("Step5_E_after_CT", n, err_E);
            
            std::cout << "    ✓ E-field error (point-value): " << std::scientific << err_E << std::endl;
        }

        // ========================================================================
        // STEP 6: ToIntegral (point-value fluxes → integral fluxes for time integrator)
        // ========================================================================
        std::cout << "  Step 6: Converting fluxes to integrals [ToIntegral]..." << std::endl;
        
        // ToIntegral converts point-value fluxes back to cell-averaged/integral form
        // This is what the time integrator actually uses
        point_value.point_value_fluxes_to_integral(*level, model, 0.0, fluxes, model.state.E);
        
        for (auto& patch : *level)
        {
            auto guard = model.resourcesManager->setOnPatch(*patch, fluxes);
            auto patchLayout = PHARE::amr::layoutFromPatch<Layout>(*patch);
            
            // Check integral flux errors using FACE-AVERAGED exact fluxes (4th order GL quadrature)
            auto err_rho_x = l2FaceAveragedFluxError<Layout, Direction::X>(patchLayout, fluxes.rho_fx, [](double x, double y, double z) {
                if constexpr (EnableHall)
                    return ExactHall3D::flux(Direction::X, x, y, z)[0];
                else
                    return ExactIdealMHD3D::flux(Direction::X, x, y, z)[0];
            });
            
            auto err_rho_y = l2FaceAveragedFluxError<Layout, Direction::Y>(patchLayout, fluxes.rho_fy, [](double x, double y, double z) {
                if constexpr (EnableHall)
                    return ExactHall3D::flux(Direction::Y, x, y, z)[0];
                else
                    return ExactIdealMHD3D::flux(Direction::Y, x, y, z)[0];
            });
            
            auto err_rho_z = l2FaceAveragedFluxError<Layout, Direction::Z>(patchLayout, fluxes.rho_fz, [](double x, double y, double z) {
                if constexpr (EnableHall)
                    return ExactHall3D::flux(Direction::Z, x, y, z)[0];
                else
                    return ExactIdealMHD3D::flux(Direction::Z, x, y, z)[0];
            });
            
            auto err_etot_x = l2FaceAveragedFluxError<Layout, Direction::X>(patchLayout, fluxes.Etot_fx, [](double x, double y, double z) {
                if constexpr (EnableHall)
                    return ExactHall3D::flux(Direction::X, x, y, z)[4];
                else
                    return ExactIdealMHD3D::flux(Direction::X, x, y, z)[4];
            });
            
            auto err_etot_y = l2FaceAveragedFluxError<Layout, Direction::Y>(patchLayout, fluxes.Etot_fy, [](double x, double y, double z) {
                if constexpr (EnableHall)
                    return ExactHall3D::flux(Direction::Y, x, y, z)[4];
                else
                    return ExactIdealMHD3D::flux(Direction::Y, x, y, z)[4];
            });
            
            auto err_etot_z = l2FaceAveragedFluxError<Layout, Direction::Z>(patchLayout, fluxes.Etot_fz, [](double x, double y, double z) {
                if constexpr (EnableHall)
                    return ExactHall3D::flux(Direction::Z, x, y, z)[4];
                else
                    return ExactIdealMHD3D::flux(Direction::Z, x, y, z)[4];
            });
            
            study.recordError("Step6_integral_flux_rho_X", n, err_rho_x);
            study.recordError("Step6_integral_flux_rho_Y", n, err_rho_y);
            study.recordError("Step6_integral_flux_rho_Z", n, err_rho_z);
            study.recordError("Step6_integral_flux_Etot_X", n, err_etot_x);
            study.recordError("Step6_integral_flux_Etot_Y", n, err_etot_y);
            study.recordError("Step6_integral_flux_Etot_Z", n, err_etot_z);
            
            std::cout << "    ✓ Integral flux errors (rho): "
                      << "X=" << std::scientific << err_rho_x << ", "
                      << "Y=" << err_rho_y << ", "
                      << "Z=" << err_rho_z << std::endl;
            std::cout << "    ✓ Integral flux errors (Etot): "
                      << "X=" << std::scientific << err_etot_x << ", "
                      << "Y=" << err_etot_y << ", "
                      << "Z=" << err_etot_z << std::endl;
        }
    }

    // ========================================================================
    // CONVERGENCE ANALYSIS
    // ========================================================================
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "CONVERGENCE ORDER ANALYSIS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    study.computeOrders();  // Compute convergence orders
    study.printSummary();
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
}

} // namespace


TEST(DecomposedPipeline, IdealMHD3D)
{
    runDecomposedPipelineTest<false>();
}

TEST(DecomposedPipeline, HallMHD3D)
{
    runDecomposedPipelineTest<true>();
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
