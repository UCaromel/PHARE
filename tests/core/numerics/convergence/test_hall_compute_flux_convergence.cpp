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
template<typename FVMethodT, typename MHDModelT, typename DispatchersT>
struct CTTypeBuilder
{
    template<typename T>
    using Rec = typename FVMethodT::template Rec<T>;
    using type = typename DispatchersT::template ConstrainedTransport_t<MHDModelT, Rec, FVMethodT::Hall,
                                                                        FVMethodT::Resistivity, FVMethodT::HyperResistivity>;
};

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
                        = layout.template deriv<Direction::Y, 2>(Bz, idx)
                          - layout.template deriv<Direction::Z, 2>(By, idx);
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
                        = layout.template deriv<Direction::Z, 2>(Bx, idx)
                          - layout.template deriv<Direction::X, 2>(Bz, idx);
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

template<bool UseSecondOrderAmpere>
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
        = PHARE::solver::ComputeFluxes<HallFVMethod3D<MHDModelT>::template type, MHDModelT>;
    using DispatchersT = PHARE::solver::Dispatchers<Layout>;
    using ToPointValueT = typename DispatchersT::template ToPointValue_t<MHDModelT>;
    using ToPrimitiveT = typename DispatchersT::ToPrimitiveConverter_t;
    using FVMethodT = typename DispatchersT::template FVMethod_t<HallFVMethod3D<MHDModelT>::template type>;
    using CTT = typename CTTypeBuilder<FVMethodT, MHDModelT, DispatchersT>::type;

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

        if constexpr (UseSecondOrderAmpere)
        {
            ToPointValueT point_value{};
            ToPrimitiveT to_primitive{fluxDict["to_primitive"]};
            FVMethodT fvm{fluxDict["fv_method"]};
            CTT ct{fluxDict["constrained_transport"]};

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
            }

            computeSecondOrderAmpere<Layout>(model, model.state, *level);
            point_value.point_value_J(*level, model, 0.0, model.state.J);
            bc.fillCurrentPointGhosts(point_value.to_point_value_.J, *level, 0.0);
            point_value(*level, model, 0.0, model.state);
            bc.fillMagneticPointGhosts(point_value.to_point_value_.B, *level, 0.0);
            to_primitive(*level, model, 0.0, point_value.to_point_value_);
            bc.fillPrimitivePointGhosts(point_value.to_point_value_, *level, 0.0);
            fvm(*level, model, 0.0, ct.constrained_transport_, point_value.to_point_value_, fluxes);
            ct(*level, model, point_value.to_point_value_, model.state.E);
            bc.fillElectricGhosts(model.state.E, *level, 0.0);
            point_value.point_value_fluxes_to_integral(*level, model, 0.0, fluxes, model.state.E);
        }
        else
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
    auto [nCells, errors] = runFullFluxConvergence<false>();

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

TEST(HallConvergence, FullComputeFluxHall3DPeriodicSecondOrderAmperePath)
{
    std::cout << "\n=== TEST 11: full Hall 3D pipeline with explicit 2nd-order Ampere ==="
              << std::endl;
    auto [nCells, errors] = runFullFluxConvergence<true>();

    std::cout << "  Observed orders (manual 2nd-order Ampere path):" << std::endl;
    for (auto const& [name, err] : errors)
    {
        ASSERT_EQ(err.size(), nCells.size());
        std::cout << "    " << name << ": ";
        for (std::size_t i = 1; i < err.size(); ++i)
        {
            auto ord = convergenceOrder(err[i - 1], err[i]);
            std::cout << std::fixed << std::setprecision(2) << ord
                      << (i + 1 < err.size() ? ", " : "");
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
