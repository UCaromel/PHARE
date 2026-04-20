/**
 * @file test_mhd_init_convergence.cpp
 *
 * Convergence test for MHDState::initialize().
 *
 * FieldUserFunctionInitializer uses 2-point GL quadrature in each *dual*
 * dimension, which covers all MHD grid centerings:
 *   - rho, V, P, rhoV, Etot : {dual,dual,dual} → 8-point GL cell averages
 *   - Bx                    : {primal,dual,dual} → 4-point GL face averages (y-z)
 *   - By                    : {dual,primal,dual} → 4-point GL face averages (x-z)
 *   - Bz                    : {dual,dual,primal} → 4-point GL face averages (x-y)
 *
 * Primitive fields (rho, V, B, P) are initialized directly from the user
 * functions and match the GL reference to machine precision.
 *
 * Conservative fields are computed from primitives:
 *   rhoV(i,j,k) = <rho>(i,j,k) * <V>(i,j,k)   (product of averages)
 *   Etot(i,j,k) = f(<rho>, <V>, <B>, <P>)      (function of averages)
 *
 * The product of cell averages is NOT the cell average of the product:
 *   <rho*V> - <rho>*<V> = O(dx²)
 *
 * Hence WITHOUT fix/init-pointvalue-conversion:
 *   rhoV, Etot error vs GL cell-averaged exact → O(dx²)  (min_order ≈ 2)
 *
 * WITH fix/init-pointvalue-conversion (transverse-Laplacian/24 correction):
 *   rhoV, Etot error vs GL cell-averaged exact → O(dx⁴)  (min_order ≈ 4)
 *
 * Assertions:
 *   Primitives (rho, V, B, P): error < 1e-10  (machine-precision agreement)
 *   Conservative (rhoV, Etot): min_order ≥ 3.5 — FAILS without the fix
 */

#include "gtest/gtest.h"

#include <iomanip>
#include <iostream>

#include "convergence_test_framework.hpp"
#include "exact_solutions.hpp"
#include "hall_convergence_test_common.hpp"

#include "amr/resources_manager/amr_utils.hpp"

#include <SAMRAI/tbox/SAMRAIManager.h>
#include <SAMRAI/tbox/SAMRAI_MPI.h>

using namespace PHARE::core;
using PHARE::test::MultiQuantityConvergenceStudy;
using PHARE::test::l2CellAveragedError;
using PHARE::test::l2FaceAveragedFluxError;
// ExactHall3D from global namespace (hall_convergence_test_common.hpp)

namespace
{

TEST(MHDInitConvergence, AllFieldsHaveCorrectAccuracy)
{
    using Layout          = GridLayout<GridLayoutImplYeeMHD<3, 2>>;
    using Array3D         = NdArrayVector<3>;
    using Grid3D          = Grid<Array3D, MHDQuantity::Scalar>;
    using Field3D         = Field<3, MHDQuantity::Scalar>;
    using VecField3D      = VecField<Field3D, MHDQuantity>;
    using ResourcesManagerT =
        PHARE::amr::ResourcesManager<Layout, Grid3D>;
    using MHDModelT = PHARE::solver::MHDModel<Layout, VecField3D,
                                              PHARE::amr::SAMRAI_Types, Grid3D>;

    // Track convergence for conservative fields (rhoV, Etot).
    MultiQuantityConvergenceStudy study;
    study.addQuantity("rhoVx");
    study.addQuantity("rhoVy");
    study.addQuantity("rhoVz");
    study.addQuantity("Etot");

    // Track max errors for primitive fields (should be machine-precision).
    std::map<std::string, double> prim_max_err;
    for (auto const& name : {"rho", "vx", "vy", "vz", "bx", "by", "bz", "p"})
        prim_max_err[name] = 0.0;

    // N=16 has only 4 interior cells per dim in y/z after ghost stripping (margin=6),
    // placing it outside the asymptotic regime.  Start from N=32.
    std::vector<int> nCells = {32, 64, 128};

    for (int n : nCells)
    {
        std::cout << "\n--- N=" << n << " ---\n";

        auto hierarchy = makePeriodicHierarchy3D(n);
        auto level     = hierarchy->getPatchLevel(0);
        auto resman    = std::make_shared<ResourcesManagerT>();
        auto modelDict = makeHall3DMHDModelDict();
        MHDModelT model{modelDict, resman};

        model.resourcesManager->registerResources(model.state);

        for (auto& patch : *level)
            model.allocate(*patch, 0.0);

        // Call the production initialize() — this is the function under test.
        for (auto& patch : *level)
        {
            auto guard       = model.resourcesManager->setOnPatch(*patch, model.state);
            auto patchLayout = PHARE::amr::layoutFromPatch<Layout>(*patch);
            model.state.initialize(patchLayout);
        }

        // Measure errors for all fields.
        for (auto& patch : *level)
        {
            auto guard       = model.resourcesManager->setOnPatch(*patch, model.state);
            auto patchLayout = PHARE::amr::layoutFromPatch<Layout>(*patch);

            // ---------------------------------------------------------------
            // Primitives: {dual,dual,dual} cell-averaged by GL init
            // → machine-precision agreement vs l2CellAveragedError
            // ---------------------------------------------------------------
            auto rho_err = l2CellAveragedError(patchLayout, model.state.rho, ExactHall3D::rho);
            auto vx_err  = l2CellAveragedError(patchLayout, model.state.V(Component::X),  ExactHall3D::vx);
            auto vy_err  = l2CellAveragedError(patchLayout, model.state.V(Component::Y),  ExactHall3D::vy);
            auto vz_err  = l2CellAveragedError(patchLayout, model.state.V(Component::Z),  ExactHall3D::vz);
            auto p_err   = l2CellAveragedError(patchLayout, model.state.P,  ExactHall3D::pressure);

            // B: face-averaged (GL in transverse dual dims)
            // → machine-precision agreement vs l2FaceAveragedFluxError
            auto bx_err = l2FaceAveragedFluxError<Layout, Direction::X>(
                patchLayout, model.state.B(Component::X), ExactHall3D::bx);
            auto by_err = l2FaceAveragedFluxError<Layout, Direction::Y>(
                patchLayout, model.state.B(Component::Y), ExactHall3D::by);
            auto bz_err = l2FaceAveragedFluxError<Layout, Direction::Z>(
                patchLayout, model.state.B(Component::Z), ExactHall3D::bz);

            prim_max_err["rho"] = std::max(prim_max_err["rho"], rho_err);
            prim_max_err["vx"]  = std::max(prim_max_err["vx"],  vx_err);
            prim_max_err["vy"]  = std::max(prim_max_err["vy"],  vy_err);
            prim_max_err["vz"]  = std::max(prim_max_err["vz"],  vz_err);
            prim_max_err["bx"]  = std::max(prim_max_err["bx"],  bx_err);
            prim_max_err["by"]  = std::max(prim_max_err["by"],  by_err);
            prim_max_err["bz"]  = std::max(prim_max_err["bz"],  bz_err);
            prim_max_err["p"]   = std::max(prim_max_err["p"],   p_err);

            // ---------------------------------------------------------------
            // Conservative: product of cell averages ≠ cell average of product
            // → O(dx²) WITHOUT fix, O(dx⁴) WITH fix
            // ---------------------------------------------------------------
            auto rhoVx_err = l2CellAveragedError(
                patchLayout, model.state.rhoV(Component::X),
                [](double x, double y, double z) {
                    return ExactHall3D::rho(x, y, z) * ExactHall3D::vx(x, y, z);
                });
            auto rhoVy_err = l2CellAveragedError(
                patchLayout, model.state.rhoV(Component::Y),
                [](double x, double y, double z) {
                    return ExactHall3D::rho(x, y, z) * ExactHall3D::vy(x, y, z);
                });
            auto rhoVz_err = l2CellAveragedError(
                patchLayout, model.state.rhoV(Component::Z),
                [](double x, double y, double z) {
                    return ExactHall3D::rho(x, y, z) * ExactHall3D::vz(x, y, z);
                });
            auto Etot_err = l2CellAveragedError(
                patchLayout, model.state.Etot, ExactHall3D::etot);

            study.recordError("rhoVx", n, rhoVx_err);
            study.recordError("rhoVy", n, rhoVy_err);
            study.recordError("rhoVz", n, rhoVz_err);
            study.recordError("Etot",  n, Etot_err);

            std::cout << std::scientific << std::setprecision(3);
            std::cout << "  rho:   " << rho_err  << "  (machine-precision expected)\n";
            std::cout << "  vx:    " << vx_err   << "  (machine-precision expected)\n";
            std::cout << "  vy:    " << vy_err   << "  (machine-precision expected)\n";
            std::cout << "  vz:    " << vz_err   << "  (machine-precision expected)\n";
            std::cout << "  bx:    " << bx_err   << "  (machine-precision expected)\n";
            std::cout << "  by:    " << by_err   << "  (machine-precision expected)\n";
            std::cout << "  bz:    " << bz_err   << "  (machine-precision expected)\n";
            std::cout << "  p:     " << p_err    << "  (machine-precision expected)\n";
            std::cout << "  rhoVx: " << rhoVx_err << "\n";
            std::cout << "  rhoVy: " << rhoVy_err << "\n";
            std::cout << "  rhoVz: " << rhoVz_err << "\n";
            std::cout << "  Etot:  " << Etot_err  << "\n";
        }
    }

    study.computeOrders();
    study.printSummary();

    std::cout << "\n--- Primitive field max errors (all N) ---\n";
    for (auto const& [name, err] : prim_max_err)
        std::cout << "  " << std::setw(4) << name << ": " << std::scientific
                  << std::setprecision(3) << err << "\n";

    // Primitive fields: GL init → machine-precision agreement with GL reference.
    constexpr double machine_tol = 1e-10;
    for (auto const& [name, err] : prim_max_err)
        EXPECT_LT(err, machine_tol)
            << name << " primitive init error exceeds machine-precision tolerance";

    // Conservative fields: must be 4th order WITH fix/init-pointvalue-conversion.
    // Without the fix these converge at O(dx²) and the assertions fail.
    constexpr double min_conservative_order = 3.5;
    EXPECT_GE(study.getResult("rhoVx").min_order(), min_conservative_order)
        << "rhoVx not 4th order — cherry-pick fix/init-pointvalue-conversion";
    EXPECT_GE(study.getResult("rhoVy").min_order(), min_conservative_order)
        << "rhoVy not 4th order — cherry-pick fix/init-pointvalue-conversion";
    EXPECT_GE(study.getResult("rhoVz").min_order(), min_conservative_order)
        << "rhoVz not 4th order — cherry-pick fix/init-pointvalue-conversion";
    EXPECT_GE(study.getResult("Etot").min_order(), min_conservative_order)
        << "Etot not 4th order — cherry-pick fix/init-pointvalue-conversion";
}

} // namespace


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    SAMRAI::tbox::SAMRAI_MPI::init(&argc, &argv);
    SAMRAI::tbox::SAMRAIManager::initialize();
    SAMRAI::tbox::SAMRAIManager::startup();

    int result = RUN_ALL_TESTS();

    SAMRAI::tbox::SAMRAIManager::shutdown();
    SAMRAI::tbox::SAMRAIManager::finalize();
    SAMRAI::tbox::SAMRAI_MPI::finalize();

    return result;
}
