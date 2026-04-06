/**
 * @file test_hall_convergence.cpp
 *
 * Isolated convergence tests for the 3D Hall MHD pipeline.
 * Tests each component to find where order reduction occurs.
 *
 * Test sequence:
 * 1. WENO-Z reconstruct on cell-centered data (baseline)
 * 2. Projection: edge → cell center (4th order interpolation)
 * 3. Ampere: curl(B) with 4th order derivatives
 * 4. center_reconstruct: projection + WENO combined
 * 5. Double reconstruction: UCT pattern (center_reconstruct → store → reconstruct)
 */

#include "gtest/gtest.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <array>
#include <map>
#include <memory>
#include <string>

#include "amr/types/amr_types.hpp"
#include "amr/physical_models/mhd_model.hpp"
#include "amr/resources_manager/resources_manager.hpp"
#include "amr/resources_manager/amr_utils.hpp"
#include "amr/solvers/time_integrator/compute_fluxes.hpp"
#include "core/data/grid/grid.hpp"
#include "core/data/grid/gridlayout.hpp"
#include "core/data/grid/gridlayout_impl.hpp"
#include "core/data/grid/gridlayoutimplyee_mhd.hpp"
#include "core/data/ndarray/ndarray_vector.hpp"
#include "core/numerics/ampere/ampere.hpp"
#include "core/numerics/MHD_equations/MHD_equations.hpp"
#include "core/numerics/godunov_fluxes/godunov_fluxes.hpp"
#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/riemann_solvers/rusanov.hpp"
#include "core/numerics/reconstructions/wenoz.hpp"
#include "core/numerics/primite_conservative_converter/to_primitive_converter.hpp"
#include "core/utilities/point/point.hpp"
#include "core/utilities/span.hpp"
#include "tests/core/data/field/test_usable_field_fixtures_mhd.hpp"
#include "tests/core/data/vecfield/test_vecfield_fixtures_mhd.hpp"
#include <SAMRAI/tbox/MemoryDatabase.h>
#include <SAMRAI/tbox/SAMRAIManager.h>
#include <SAMRAI/tbox/SAMRAI_MPI.h>

#include "tests/core/numerics/convergence/hall_convergence_test_common.hpp"

using namespace PHARE::core;

//==============================================================================
// TEST 1: Plain WENO-Z reconstruction on cell-centered data (baseline)
//==============================================================================

TEST(HallConvergence, WENOZReconstruct1D)
{
    std::cout << "\n=== TEST 1: WENO-Z reconstruct (cell-centered) 1D ===" << std::endl;

    std::vector<std::size_t> nCells = {32, 64, 128, 256};
    std::vector<double> errors;

    for (auto n : nCells)
    {
        double dx = 1.0 / n;
        using Layout = GridLayout<GridLayoutImplYeeMHD<1, 2>>;
        Layout layout{{{dx}}, {{static_cast<unsigned int>(n)}}, Point{0.0}};

        UsableFieldMHD<1> field{"f", layout, MHDQuantity::Scalar::rho};

        auto centering = layout.centering(MHDQuantity::Scalar::rho);
        auto gsi       = layout.ghostStartIndex(centering[0], Direction::X);
        auto gei       = layout.ghostEndIndex(centering[0], Direction::X);
        for (auto i = gsi; i <= gei; ++i)
        {
            auto c   = layout.fieldNodeCoordinates(field, layout.localToAMR(Point{i}.as_signed()));
            field(i) = f(c[0]);
        }

        using WENOZ_t = WENOZReconstruction<Layout>;
        double error  = 0.0;
        std::size_t count = 0;

        auto psi = layout.physicalStartIndex(QtyCentering::primal, Direction::X);
        auto pei = layout.physicalEndIndex(QtyCentering::primal, Direction::X);

        for (auto i = psi + 3; i <= pei - 3; ++i)
        {
            MeshIndex<1> idx{i};
            auto [fL, fR] = WENOZ_t::template reconstruct<Direction::X>(field, idx);

            double x_interface = (i - psi) * dx;
            double exact       = f(x_interface);
            double avg         = 0.5 * (fL + fR);
            error += (avg - exact) * (avg - exact);
            count++;
        }
        error = std::sqrt(error / count);
        errors.push_back(error);
        std::cout << "  n=" << n << " error=" << std::scientific << error << std::endl;
    }

    std::cout << "  Convergence orders:" << std::endl;
    for (std::size_t i = 1; i < errors.size(); ++i)
    {
        double order = convergenceOrder(errors[i - 1], errors[i]);
        std::cout << "    " << nCells[i - 1] << "->" << nCells[i] << ": " << std::fixed
                  << std::setprecision(2) << order << std::endl;
    }
}

//==============================================================================
// TEST 2: Edge-to-cell projection (4th order interpolation)
//==============================================================================

TEST(HallConvergence, EdgeToCellProjection3D)
{
    std::cout << "\n=== TEST 2: Edge-to-cell projection 3D ===" << std::endl;

    std::vector<std::size_t> nCells = {16, 32, 64};
    std::vector<double> errors;

    for (auto n : nCells)
    {
        double dx = 1.0 / n;
        using Layout = GridLayout<GridLayoutImplYeeMHD<3, 2>>;
        Layout layout{{{dx, dx, dx}}, {{static_cast<unsigned int>(n), static_cast<unsigned int>(n), static_cast<unsigned int>(n)}}, Point{0.0, 0.0, 0.0}};

        // Jx at edge centering dPP
        UsableFieldMHD<3> Jx{"Jx", layout, MHDQuantity::Scalar::Jx};

        auto centering = layout.centering(MHDQuantity::Scalar::Jx);
        auto gsi_x     = layout.ghostStartIndex(centering[0], Direction::X);
        auto gei_x     = layout.ghostEndIndex(centering[0], Direction::X);
        auto gsi_y     = layout.ghostStartIndex(centering[1], Direction::Y);
        auto gei_y     = layout.ghostEndIndex(centering[1], Direction::Y);
        auto gsi_z     = layout.ghostStartIndex(centering[2], Direction::Z);
        auto gei_z     = layout.ghostEndIndex(centering[2], Direction::Z);

        for (auto i = gsi_x; i <= gei_x; ++i)
            for (auto j = gsi_y; j <= gei_y; ++j)
                for (auto kk = gsi_z; kk <= gei_z; ++kk)
                {
                    auto c       = layout.fieldNodeCoordinates(Jx, layout.localToAMR(Point{i, j, kk}.as_signed()));
                    Jx(i, j, kk) = f3d(c[0], c[1], c[2]);
                }

        auto projection = Layout::edgeXToCellCenter();

        double error      = 0.0;
        std::size_t count = 0;

        // Cell center indices
        auto cell_centering = layout.centering(MHDQuantity::Scalar::rho);
        auto psi_x          = layout.physicalStartIndex(cell_centering[0], Direction::X);
        auto pei_x          = layout.physicalEndIndex(cell_centering[0], Direction::X);
        auto psi_y          = layout.physicalStartIndex(cell_centering[1], Direction::Y);
        auto pei_y          = layout.physicalEndIndex(cell_centering[1], Direction::Y);
        auto psi_z          = layout.physicalStartIndex(cell_centering[2], Direction::Z);
        auto pei_z          = layout.physicalEndIndex(cell_centering[2], Direction::Z);

        for (auto i = psi_x + 2; i <= pei_x - 2; ++i)
            for (auto j = psi_y + 2; j <= pei_y - 2; ++j)
                for (auto kk = psi_z + 2; kk <= pei_z - 2; ++kk)
                {
                    MeshIndex<3> idx{i, j, kk};
                    double projected = Layout::project(Jx, idx, projection);

                    // Get cell center coordinates
                    UsableFieldMHD<3> dummy{"d", layout, MHDQuantity::Scalar::rho};
                    auto c = layout.fieldNodeCoordinates(dummy, layout.localToAMR(Point{i, j, kk}.as_signed()));

                    double exact = f3d(c[0], c[1], c[2]);
                    error += (projected - exact) * (projected - exact);
                    count++;
                }

        error = std::sqrt(error / count);
        errors.push_back(error);
        std::cout << "  n=" << n << " error=" << std::scientific << error << std::endl;
    }

    std::cout << "  Convergence orders:" << std::endl;
    for (std::size_t i = 1; i < errors.size(); ++i)
    {
        double order = convergenceOrder(errors[i - 1], errors[i]);
        std::cout << "    " << nCells[i - 1] << "->" << nCells[i] << ": " << std::fixed
                  << std::setprecision(2) << order << std::endl;
    }
}

//==============================================================================
// TEST 3: center_reconstruct (projection + WENO combined)
//==============================================================================

TEST(HallConvergence, CenterReconstruct3D)
{
    std::cout << "\n=== TEST 3: center_reconstruct (edge->cell + WENO) 3D ===" << std::endl;

    std::vector<std::size_t> nCells = {16, 32, 64};
    std::vector<double> errors;

    for (auto n : nCells)
    {
        double dx = 1.0 / n;
        using Layout = GridLayout<GridLayoutImplYeeMHD<3, 2>>;
        Layout layout{{{dx, dx, dx}}, {{static_cast<unsigned int>(n), static_cast<unsigned int>(n), static_cast<unsigned int>(n)}}, Point{0.0, 0.0, 0.0}};

        UsableFieldMHD<3> Jx{"Jx", layout, MHDQuantity::Scalar::Jx};

        auto centering = layout.centering(MHDQuantity::Scalar::Jx);
        auto gsi_x     = layout.ghostStartIndex(centering[0], Direction::X);
        auto gei_x     = layout.ghostEndIndex(centering[0], Direction::X);
        auto gsi_y     = layout.ghostStartIndex(centering[1], Direction::Y);
        auto gei_y     = layout.ghostEndIndex(centering[1], Direction::Y);
        auto gsi_z     = layout.ghostStartIndex(centering[2], Direction::Z);
        auto gei_z     = layout.ghostEndIndex(centering[2], Direction::Z);

        for (auto i = gsi_x; i <= gei_x; ++i)
            for (auto j = gsi_y; j <= gei_y; ++j)
                for (auto kk = gsi_z; kk <= gei_z; ++kk)
                {
                    auto c       = layout.fieldNodeCoordinates(Jx, layout.localToAMR(Point{i, j, kk}.as_signed()));
                    Jx(i, j, kk) = f3d(c[0], c[1], c[2]);
                }

        using WENOZ_t   = WENOZReconstruction<Layout>;
        auto projection = Layout::edgeXToCellCenter();

        double error      = 0.0;
        std::size_t count = 0;

        auto flux_centering = layout.centering(MHDQuantity::Scalar::ScalarFlux_x);
        auto psi_x          = layout.physicalStartIndex(flux_centering[0], Direction::X);
        auto pei_x          = layout.physicalEndIndex(flux_centering[0], Direction::X);
        auto psi_y          = layout.physicalStartIndex(flux_centering[1], Direction::Y);
        auto pei_y          = layout.physicalEndIndex(flux_centering[1], Direction::Y);
        auto psi_z          = layout.physicalStartIndex(flux_centering[2], Direction::Z);
        auto pei_z          = layout.physicalEndIndex(flux_centering[2], Direction::Z);

        // Create dummy field at flux face centering to get correct coordinates
        UsableFieldMHD<3> flux_face{"flux", layout, MHDQuantity::Scalar::ScalarFlux_x};
        
        for (auto i = psi_x + 4; i <= pei_x - 4; ++i)
            for (auto j = psi_y + 4; j <= pei_y - 4; ++j)
                for (auto kk = psi_z + 4; kk <= pei_z - 4; ++kk)
                {
                    MeshIndex<3> idx{i, j, kk};
                    auto [fL, fR]
                        = WENOZ_t::template center_reconstruct<Direction::X>(Jx, idx, projection);

                    // Get coordinates at flux face location (Pdd), not edge location (dPP)
                    auto c = layout.fieldNodeCoordinates(flux_face, layout.localToAMR(Point{i, j, kk}.as_signed()));

                    double exact = f3d(c[0], c[1], c[2]);
                    double avg   = 0.5 * (fL + fR);
                    error += (avg - exact) * (avg - exact);
                    count++;
                }

        error = std::sqrt(error / count);
        errors.push_back(error);
        std::cout << "  n=" << n << " error=" << std::scientific << error << std::endl;
    }

    std::cout << "  Convergence orders:" << std::endl;
    for (std::size_t i = 1; i < errors.size(); ++i)
    {
        double order = convergenceOrder(errors[i - 1], errors[i]);
        std::cout << "    " << nCells[i - 1] << "->" << nCells[i] << ": " << std::fixed
                  << std::setprecision(2) << order << std::endl;
    }
}

//==============================================================================
// TEST 4: Double reconstruction (UCT pattern)
// Step 1: center_reconstruct J to X-flux interface → store as jt_x
// Step 2: reconstruct jt_x in Y direction (what UCT does for EMF)
//==============================================================================

TEST(HallConvergence, DoubleReconstruction3D)
{
    std::cout << "\n=== TEST 4: Double reconstruction (UCT pattern) 3D ===" << std::endl;
    std::cout << "    Step 1: edge J -> X-flux interface (center_reconstruct)" << std::endl;
    std::cout << "    Step 2: X-flux jt -> Y edge (reconstruct in Y)" << std::endl;

    std::vector<std::size_t> nCells = {16, 32, 64};
    std::vector<double> errors;

    for (auto n : nCells)
    {
        double dx = 1.0 / n;
        using Layout = GridLayout<GridLayoutImplYeeMHD<3, 2>>;
        Layout layout{{{dx, dx, dx}}, {{static_cast<unsigned int>(n), static_cast<unsigned int>(n), static_cast<unsigned int>(n)}}, Point{0.0, 0.0, 0.0}};
        using WENOZ_t = WENOZReconstruction<Layout>;

        // Jy at edge centering PdP
        UsableFieldMHD<3> Jy{"Jy", layout, MHDQuantity::Scalar::Jy};

        auto Jy_centering = layout.centering(MHDQuantity::Scalar::Jy);
        auto gsi_x        = layout.ghostStartIndex(Jy_centering[0], Direction::X);
        auto gei_x        = layout.ghostEndIndex(Jy_centering[0], Direction::X);
        auto gsi_y        = layout.ghostStartIndex(Jy_centering[1], Direction::Y);
        auto gei_y        = layout.ghostEndIndex(Jy_centering[1], Direction::Y);
        auto gsi_z        = layout.ghostStartIndex(Jy_centering[2], Direction::Z);
        auto gei_z        = layout.ghostEndIndex(Jy_centering[2], Direction::Z);

        for (auto i = gsi_x; i <= gei_x; ++i)
            for (auto j = gsi_y; j <= gei_y; ++j)
                for (auto kk = gsi_z; kk <= gei_z; ++kk)
                {
                    auto c       = layout.fieldNodeCoordinates(Jy, layout.localToAMR(Point{i, j, kk}.as_signed()));
                    Jy(i, j, kk) = f3d(c[0], c[1], c[2]);
                }

        // jt_x at X-flux interface (VecFluxY_x centering: Pdd)
        UsableFieldMHD<3> jt_x{"jt_x", layout, MHDQuantity::Scalar::VecFluxY_x};

        auto projection   = Layout::edgeYToCellCenter();
        auto jt_centering = layout.centering(MHDQuantity::Scalar::VecFluxY_x);
        auto jt_psi_x     = layout.physicalStartIndex(jt_centering[0], Direction::X);
        auto jt_pei_x     = layout.physicalEndIndex(jt_centering[0], Direction::X);
        auto jt_psi_y     = layout.physicalStartIndex(jt_centering[1], Direction::Y);
        auto jt_pei_y     = layout.physicalEndIndex(jt_centering[1], Direction::Y);
        auto jt_psi_z     = layout.physicalStartIndex(jt_centering[2], Direction::Z);
        auto jt_pei_z     = layout.physicalEndIndex(jt_centering[2], Direction::Z);

        // First reconstruction: Jy → jt_x
        for (auto i = jt_psi_x; i <= jt_pei_x; ++i)
            for (auto j = jt_psi_y; j <= jt_pei_y; ++j)
                for (auto kk = jt_psi_z; kk <= jt_pei_z; ++kk)
                {
                    MeshIndex<3> idx{i, j, kk};
                    auto [fL, fR]
                        = WENOZ_t::template center_reconstruct<Direction::X>(Jy, idx, projection);
                    jt_x(i, j, kk) = 0.5 * (fL + fR);
                }

        // Second reconstruction: jt_x in Y direction
        double error      = 0.0;
        std::size_t count = 0;

        auto Ez_centering = layout.centering(MHDQuantity::Scalar::Ez);
        auto psi_x        = layout.physicalStartIndex(Ez_centering[0], Direction::X);
        auto pei_x        = layout.physicalEndIndex(Ez_centering[0], Direction::X);
        auto psi_y        = layout.physicalStartIndex(Ez_centering[1], Direction::Y);
        auto pei_y        = layout.physicalEndIndex(Ez_centering[1], Direction::Y);
        auto psi_z        = layout.physicalStartIndex(Ez_centering[2], Direction::Z);
        auto pei_z        = layout.physicalEndIndex(Ez_centering[2], Direction::Z);

        // Create dummy field at Ez centering to get correct coordinates
        UsableFieldMHD<3> Ez_edge{"Ez", layout, MHDQuantity::Scalar::Ez};

        for (auto i = psi_x + 5; i <= pei_x - 5; ++i)
            for (auto j = psi_y + 5; j <= pei_y - 5; ++j)
                for (auto kk = psi_z + 5; kk <= pei_z - 5; ++kk)
                {
                    MeshIndex<3> idx{i, j, kk};
                    auto [jS, jN] = WENOZ_t::template reconstruct<Direction::Y>(jt_x, idx);

                    // Get coordinates at Ez edge location (PPd), where final EMF lives
                    auto c = layout.fieldNodeCoordinates(Ez_edge, layout.localToAMR(Point{i, j, kk}.as_signed()));

                    double exact = f3d(c[0], c[1], c[2]);
                    double avg   = 0.5 * (jS + jN);
                    error += (avg - exact) * (avg - exact);
                    count++;
                }

        error = std::sqrt(error / count);
        errors.push_back(error);
        std::cout << "  n=" << n << " error=" << std::scientific << error << std::endl;
    }

    std::cout << "  Convergence orders:" << std::endl;
    for (std::size_t i = 1; i < errors.size(); ++i)
    {
        double order = convergenceOrder(errors[i - 1], errors[i]);
        std::cout << "    " << nCells[i - 1] << "->" << nCells[i] << ": " << std::fixed
                  << std::setprecision(2) << order << std::endl;
    }
}

//==============================================================================
// TEST 5: Point-value B + 4th-order curl -> point-value J
//==============================================================================
TEST(HallConvergence, PointValueBPlusAmpere4thOrderToPointJ3D)
{
    std::cout << "\n=== TEST 5: point-value B + 4th-order curl -> point-value J (Jx) 3D ==="
              << std::endl;

    std::vector<std::size_t> nCells = {16, 32, 64};
    std::vector<double> errors;

    for (auto n : nCells)
    {
        double dx = 1.0 / n;
        using Layout = GridLayout<GridLayoutImplYeeMHD<3, 2>>;
        Layout layout{{{dx, dx, dx}},
                      {{static_cast<unsigned int>(n), static_cast<unsigned int>(n),
                        static_cast<unsigned int>(n)}},
                      Point{0.0, 0.0, 0.0}};

        UsableVecFieldMHD<3> B{"B", layout, MHDQuantity::Vector::B};
        UsableVecFieldMHD<3> J{"J", layout, MHDQuantity::Vector::J};

        auto const& [Bx, By, Bz] = B();
        auto const& [Jx, Jy, Jz] = J();

        auto fill_field = [&](auto& field, auto func) {
            auto cent = layout.centering(field.physicalQuantity());
            for (auto i = layout.ghostStartIndex(cent[0], Direction::X);
                 i <= layout.ghostEndIndex(cent[0], Direction::X); ++i)
                for (auto j = layout.ghostStartIndex(cent[1], Direction::Y);
                     j <= layout.ghostEndIndex(cent[1], Direction::Y); ++j)
                    for (auto kk = layout.ghostStartIndex(cent[2], Direction::Z);
                         kk <= layout.ghostEndIndex(cent[2], Direction::Z); ++kk)
                    {
                        auto c = layout.fieldNodeCoordinates(
                            field, layout.localToAMR(Point{i, j, kk}.as_signed()));
                        field(i, j, kk) = func(c[0], c[1], c[2]);
                    }
        };

        fill_field(Bx, Bx_func);
        fill_field(By, By_func);
        fill_field(Bz, Bz_func);

        Ampere<Layout> ampere;
        ampere.setLayout(&layout);
        ampere(B, J);

        double error      = 0.0;
        std::size_t count = 0;
        auto Jx_cent      = layout.centering(MHDQuantity::Scalar::Jx);
        auto psi_x        = layout.physicalStartIndex(Jx_cent[0], Direction::X);
        auto pei_x        = layout.physicalEndIndex(Jx_cent[0], Direction::X);
        auto psi_y        = layout.physicalStartIndex(Jx_cent[1], Direction::Y);
        auto pei_y        = layout.physicalEndIndex(Jx_cent[1], Direction::Y);
        auto psi_z        = layout.physicalStartIndex(Jx_cent[2], Direction::Z);
        auto pei_z        = layout.physicalEndIndex(Jx_cent[2], Direction::Z);

        for (auto i = psi_x + 2; i <= pei_x - 2; ++i)
            for (auto j = psi_y + 2; j <= pei_y - 2; ++j)
                for (auto kk = psi_z + 2; kk <= pei_z - 2; ++kk)
                {
                    auto c    = layout.fieldNodeCoordinates(
                        Jx, layout.localToAMR(Point{i, j, kk}.as_signed()));
                    auto diff = Jx(i, j, kk) - Jx_exact(c[0], c[1], c[2]);
                    error += diff * diff;
                    ++count;
                }

        error = std::sqrt(error / count);
        errors.push_back(error);
        std::cout << "  n=" << n << " error=" << std::scientific << error << std::endl;
    }

    std::cout << "  Convergence orders:" << std::endl;
    for (std::size_t i = 1; i < errors.size(); ++i)
    {
        double order = convergenceOrder(errors[i - 1], errors[i]);
        std::cout << "    " << nCells[i - 1] << "->" << nCells[i] << ": " << std::fixed
                  << std::setprecision(2) << order << std::endl;
    }
}

//==============================================================================
// TEST 6: Face-average B + 4th-order curl -> average J -> point-value J
//==============================================================================

TEST(HallConvergence, FaceAverageBPlusAmpere4thOrderToPointJ3D)
{
    std::cout << "\n=== TEST 6: face-average B + 4th-order curl(avg J) -> point-value J (Jx) 3D ==="
              << std::endl;

    std::vector<std::size_t> nCells = {16, 32, 64};
    std::vector<double> errors;

    for (auto n : nCells)
    {
        double dx = 1.0 / n;
        using Layout = GridLayout<GridLayoutImplYeeMHD<3, 2>>;
        Layout layout{{{dx, dx, dx}}, {{static_cast<unsigned int>(n), static_cast<unsigned int>(n), static_cast<unsigned int>(n)}}, Point{0.0, 0.0, 0.0}};

        UsableFieldMHD<3> By{"By", layout, MHDQuantity::Scalar::By};
        UsableFieldMHD<3> Bz{"Bz", layout, MHDQuantity::Scalar::Bz};
        UsableFieldMHD<3> Jx_avg{"Jx_avg", layout, MHDQuantity::Scalar::Jx};
        UsableFieldMHD<3> Jx_pv{"Jx_pv", layout, MHDQuantity::Scalar::Jx};

        auto fill_face_average = [&](auto& field, auto point_fn) {
            auto cent = layout.centering(field.physicalQuantity());
            for (auto i = layout.ghostStartIndex(cent[0], Direction::X);
                 i <= layout.ghostEndIndex(cent[0], Direction::X); ++i)
                for (auto j = layout.ghostStartIndex(cent[1], Direction::Y);
                     j <= layout.ghostEndIndex(cent[1], Direction::Y); ++j)
                    for (auto kk = layout.ghostStartIndex(cent[2], Direction::Z);
                         kk <= layout.ghostEndIndex(cent[2], Direction::Z); ++kk)
                    {
                        auto c = layout.fieldNodeCoordinates(
                            field, layout.localToAMR(Point{i, j, kk}.as_signed()));
                        double point = point_fn(c[0], c[1], c[2]);
                        // For the manufactured By/Bz fields used here, transverse Laplacian = -2*k^2*point.
                        double lapl_t_scaled = (-2.0 * k * k * point) * (dx * dx);
                        field(i, j, kk)      = point + lapl_t_scaled / 24.0;
                    }
        };
        fill_face_average(By, By_func);
        fill_face_average(Bz, Bz_func);

        auto Jx_cent = layout.centering(MHDQuantity::Scalar::Jx);
        auto psi_x   = layout.physicalStartIndex(Jx_cent[0], Direction::X);
        auto pei_x   = layout.physicalEndIndex(Jx_cent[0], Direction::X);
        auto psi_y   = layout.physicalStartIndex(Jx_cent[1], Direction::Y);
        auto pei_y   = layout.physicalEndIndex(Jx_cent[1], Direction::Y);
        auto psi_z   = layout.physicalStartIndex(Jx_cent[2], Direction::Z);
        auto pei_z   = layout.physicalEndIndex(Jx_cent[2], Direction::Z);
        
        for (auto i = psi_x + 4; i <= pei_x - 4; ++i)
            for (auto j = psi_y + 4; j <= pei_y - 4; ++j)
                for (auto kk = psi_z + 4; kk <= pei_z - 4; ++kk)
                {
                    MeshIndex<3> idx{i, j, kk};
                    Jx_avg(i, j, kk) = layout.template deriv<Direction::Y, 4>(Bz, idx)
                                     - layout.template deriv<Direction::Z, 4>(By, idx);
                }

        double error      = 0.0;
        std::size_t count = 0;
        for (auto i = psi_x + 5; i <= pei_x - 5; ++i)
            for (auto j = psi_y + 5; j <= pei_y - 5; ++j)
                for (auto kk = psi_z + 5; kk <= pei_z - 5; ++kk)
                {
                    MeshIndex<3> idx{i, j, kk};
                    auto lapl_x = layout.template directionalLapl<Direction::X>(Jx_avg, idx);
                    Jx_pv(i, j, kk) = Jx_avg(i, j, kk) - lapl_x / 24.0;
                    auto c = layout.fieldNodeCoordinates(
                        Jx_pv, layout.localToAMR(Point{i, j, kk}.as_signed()));
                    double exact = Jx_exact(c[0], c[1], c[2]);
                    error += (Jx_pv(i, j, kk) - exact) * (Jx_pv(i, j, kk) - exact);
                    ++count;
                }

        error = std::sqrt(error / count);
        errors.push_back(error);
        std::cout << "  n=" << n << " error=" << std::scientific << error << std::endl;
    }

    std::cout << "  Convergence orders:" << std::endl;
    for (std::size_t i = 1; i < errors.size(); ++i)
    {
        double order = convergenceOrder(errors[i - 1], errors[i]);
        std::cout << "    " << nCells[i - 1] << "->" << nCells[i] << ": " << std::fixed
                  << std::setprecision(2) << order << std::endl;
    }
}

//==============================================================================
// TEST 7: Face-average B + 2nd-order curl -> average J -> point-value J
//==============================================================================
TEST(HallConvergence, FaceAverageBPlusAmpere2ndOrderToPointJ3D)
{
    std::cout << "\n=== TEST 7: face-average B + 2nd-order curl(avg J) -> point-value J (Jx) 3D ==="
              << std::endl;

    std::vector<std::size_t> nCells = {16, 32, 64};
    std::vector<double> errors;

    for (auto n : nCells)
    {
        double dx = 1.0 / n;
        using Layout = GridLayout<GridLayoutImplYeeMHD<3, 2>>;
        Layout layout{{{dx, dx, dx}},
                      {{static_cast<unsigned int>(n), static_cast<unsigned int>(n),
                        static_cast<unsigned int>(n)}},
                      Point{0.0, 0.0, 0.0}};

        UsableFieldMHD<3> By{"By", layout, MHDQuantity::Scalar::By};
        UsableFieldMHD<3> Bz{"Bz", layout, MHDQuantity::Scalar::Bz};
        UsableFieldMHD<3> Jx_avg{"Jx_avg", layout, MHDQuantity::Scalar::Jx};
        UsableFieldMHD<3> Jx_pv{"Jx_pv", layout, MHDQuantity::Scalar::Jx};

        auto fill_face_average = [&](auto& field, auto point_fn) {
            auto cent = layout.centering(field.physicalQuantity());
            for (auto i = layout.ghostStartIndex(cent[0], Direction::X);
                 i <= layout.ghostEndIndex(cent[0], Direction::X); ++i)
                for (auto j = layout.ghostStartIndex(cent[1], Direction::Y);
                     j <= layout.ghostEndIndex(cent[1], Direction::Y); ++j)
                    for (auto kk = layout.ghostStartIndex(cent[2], Direction::Z);
                         kk <= layout.ghostEndIndex(cent[2], Direction::Z); ++kk)
                    {
                        auto c = layout.fieldNodeCoordinates(
                            field, layout.localToAMR(Point{i, j, kk}.as_signed()));
                        double point = point_fn(c[0], c[1], c[2]);
                        double lapl_t_scaled = (-2.0 * k * k * point) * (dx * dx);
                        field(i, j, kk)      = point + lapl_t_scaled / 24.0;
                    }
        };
        fill_face_average(By, By_func);
        fill_face_average(Bz, Bz_func);

        auto jcent = layout.centering(MHDQuantity::Scalar::Jx);
        auto psi_x = layout.physicalStartIndex(jcent[0], Direction::X);
        auto pei_x = layout.physicalEndIndex(jcent[0], Direction::X);
        auto psi_y = layout.physicalStartIndex(jcent[1], Direction::Y);
        auto pei_y = layout.physicalEndIndex(jcent[1], Direction::Y);
        auto psi_z = layout.physicalStartIndex(jcent[2], Direction::Z);
        auto pei_z = layout.physicalEndIndex(jcent[2], Direction::Z);

        for (auto i = psi_x + 3; i <= pei_x - 3; ++i)
            for (auto j = psi_y + 3; j <= pei_y - 3; ++j)
                for (auto kk = psi_z + 3; kk <= pei_z - 3; ++kk)
                {
                    MeshIndex<3> idx{i, j, kk};
                    Jx_avg(i, j, kk) = layout.template deriv<Direction::Y, 2>(Bz, idx)
                                     - layout.template deriv<Direction::Z, 2>(By, idx);
                }

        double err = 0.0;
        std::size_t cnt = 0;
        for (auto i = psi_x + 4; i <= pei_x - 4; ++i)
            for (auto j = psi_y + 4; j <= pei_y - 4; ++j)
                for (auto kk = psi_z + 4; kk <= pei_z - 4; ++kk)
                {
                    MeshIndex<3> idx{i, j, kk};
                    auto lapl_x = layout.template directionalLapl<Direction::X>(Jx_avg, idx);
                    Jx_pv(i, j, kk) = Jx_avg(i, j, kk) - lapl_x / 24.0;
                    auto c = layout.fieldNodeCoordinates(
                        Jx_pv, layout.localToAMR(Point{i, j, kk}.as_signed()));
                    auto diff = Jx_pv(i, j, kk) - Jx_exact(c[0], c[1], c[2]);
                    err += diff * diff;
                    ++cnt;
                }

        errors.push_back(std::sqrt(err / cnt));
        std::cout << "  n=" << n << " error=" << std::scientific << errors.back() << std::endl;
    }

    std::cout << "  Convergence orders:" << std::endl;
    for (std::size_t i = 1; i < errors.size(); ++i)
    {
        std::cout << "    " << nCells[i - 1] << "->" << nCells[i] << ": " << std::fixed
                  << std::setprecision(2) << convergenceOrder(errors[i - 1], errors[i]) << std::endl;
    }
}

//==============================================================================
// TEST 8: Point value J conversion (edge-average → point-value)
//==============================================================================

TEST(HallConvergence, PointValueJ3D)
{
    std::cout << "\n=== TEST 8: Point value J conversion (edge avg → point value) 3D ===" << std::endl;

    std::vector<std::size_t> nCells = {16, 32, 64};
    std::vector<double> errors;

    for (auto n : nCells)
    {
        double dx = 1.0 / n;
        using Layout = GridLayout<GridLayoutImplYeeMHD<3, 2>>;
        Layout layout{{{dx, dx, dx}}, {{static_cast<unsigned int>(n), static_cast<unsigned int>(n), static_cast<unsigned int>(n)}}, Point{0.0, 0.0, 0.0}};

        // Edge-averaged J (input)
        UsableFieldMHD<3> Jx_avg{"Jx_avg", layout, MHDQuantity::Scalar::Jx};
        
        // Manufacture line-averaged Jx along X from an analytic point field.
        // For a line average over X:
        //   Jx_avg = Jx_point + (dx^2/24) * d2Jx/dx2 + O(dx^4)
        // and for our trigonometric manufactured solution d2/dx2 = -k^2 * value.
        auto centering = layout.centering(MHDQuantity::Scalar::Jx);
        auto gsi_x     = layout.ghostStartIndex(centering[0], Direction::X);
        auto gei_x     = layout.ghostEndIndex(centering[0], Direction::X);
        auto gsi_y     = layout.ghostStartIndex(centering[1], Direction::Y);
        auto gei_y     = layout.ghostEndIndex(centering[1], Direction::Y);
        auto gsi_z     = layout.ghostStartIndex(centering[2], Direction::Z);
        auto gei_z     = layout.ghostEndIndex(centering[2], Direction::Z);

        for (auto i = gsi_x; i <= gei_x; ++i)
            for (auto j = gsi_y; j <= gei_y; ++j)
                for (auto kk = gsi_z; kk <= gei_z; ++kk)
                {
                    auto c = layout.fieldNodeCoordinates(
                        Jx_avg, layout.localToAMR(Point{i, j, kk}.as_signed()));
                    double point = f3d(c[0], c[1], c[2]);
                    double d2x_scaled = (-k * k * point) * (dx * dx);
                    Jx_avg(i, j, kk)  = point + d2x_scaled / 24.0;
                }

        // Point value J (output) - allocate at same centering
        UsableFieldMHD<3> Jx_pv{"Jx_pv", layout, MHDQuantity::Scalar::Jx};
        
        // Apply point value conversion: f_pv = f_avg - (1/24) * Laplacian(f)
        // This adds a 4th order correction
        auto psi_x = layout.physicalStartIndex(centering[0], Direction::X);
        auto pei_x = layout.physicalEndIndex(centering[0], Direction::X);
        auto psi_y = layout.physicalStartIndex(centering[1], Direction::Y);
        auto pei_y = layout.physicalEndIndex(centering[1], Direction::Y);
        auto psi_z = layout.physicalStartIndex(centering[2], Direction::Z);
        auto pei_z = layout.physicalEndIndex(centering[2], Direction::Z);
        
        double error      = 0.0;
        std::size_t count = 0;
        
        for (auto i = psi_x + 2; i <= pei_x - 2; ++i)
            for (auto j = psi_y + 2; j <= pei_y - 2; ++j)
                for (auto kk = psi_z + 2; kk <= pei_z - 2; ++kk)
                {
                    MeshIndex<3> idx{i, j, kk};
                    
                    // Edge conversion in production for Jx uses the parallel directional Laplacian.
                    auto lapl_x = layout.template directionalLapl<Direction::X>(Jx_avg, idx);
                    Jx_pv(i, j, kk) = Jx_avg(i, j, kk) - lapl_x / 24.0;
                    
                    // For exact solution, point value equals the function itself (no averaging)
                    auto c = layout.fieldNodeCoordinates(Jx_avg, layout.localToAMR(Point{i, j, kk}.as_signed()));
                    double exact = f3d(c[0], c[1], c[2]);
                    
                    error += (Jx_pv(i, j, kk) - exact) * (Jx_pv(i, j, kk) - exact);
                    count++;
                }

        error = std::sqrt(error / count);
        errors.push_back(error);
        std::cout << "  n=" << n << " error=" << std::scientific << error << std::endl;
    }

    std::cout << "  Convergence orders:" << std::endl;
    for (std::size_t i = 1; i < errors.size(); ++i)
    {
        double order = convergenceOrder(errors[i - 1], errors[i]);
        std::cout << "    " << nCells[i - 1] << "->" << nCells[i] << ": " << std::fixed
                  << std::setprecision(2) << order << std::endl;
    }
}

//==============================================================================
// TEST 9: Cell-centered point-value sanity check (rho)
//==============================================================================

TEST(HallConvergence, PointValueCellCenteredSanity3D)
{
    std::cout << "\n=== TEST 9: Cell-centered point-value sanity (rho) 3D ===" << std::endl;

    std::vector<std::size_t> nCells = {16, 32, 64};
    std::vector<double> errors;

    for (auto n : nCells)
    {
        double dx = 1.0 / n;
        using Layout = GridLayout<GridLayoutImplYeeMHD<3, 2>>;
        Layout layout{{{dx, dx, dx}},
                      {{static_cast<unsigned int>(n), static_cast<unsigned int>(n),
                        static_cast<unsigned int>(n)}},
                      Point{0.0, 0.0, 0.0}};

        UsableFieldMHD<3> rho_avg{"rho_avg", layout, MHDQuantity::Scalar::rho};
        UsableFieldMHD<3> rho_pv{"rho_pv", layout, MHDQuantity::Scalar::rho};

        auto cent = layout.centering(MHDQuantity::Scalar::rho);
        auto gsi_x = layout.ghostStartIndex(cent[0], Direction::X);
        auto gei_x = layout.ghostEndIndex(cent[0], Direction::X);
        auto gsi_y = layout.ghostStartIndex(cent[1], Direction::Y);
        auto gei_y = layout.ghostEndIndex(cent[1], Direction::Y);
        auto gsi_z = layout.ghostStartIndex(cent[2], Direction::Z);
        auto gei_z = layout.ghostEndIndex(cent[2], Direction::Z);

        for (auto i = gsi_x; i <= gei_x; ++i)
            for (auto j = gsi_y; j <= gei_y; ++j)
                for (auto kk = gsi_z; kk <= gei_z; ++kk)
                {
                    auto c = layout.fieldNodeCoordinates(
                        rho_avg, layout.localToAMR(Point{i, j, kk}.as_signed()));
                    double point = f3d(c[0], c[1], c[2]);
                    // For this manufactured case, second derivative in each direction is analytic.
                    double lapl_analytic = d2f3d_dx2(c[0], c[1], c[2]) * (dx * dx) * 3.0;
                    rho_avg(i, j, kk)    = point + lapl_analytic / 24.0;
                }

        auto psi_x = layout.physicalStartIndex(cent[0], Direction::X);
        auto pei_x = layout.physicalEndIndex(cent[0], Direction::X);
        auto psi_y = layout.physicalStartIndex(cent[1], Direction::Y);
        auto pei_y = layout.physicalEndIndex(cent[1], Direction::Y);
        auto psi_z = layout.physicalStartIndex(cent[2], Direction::Z);
        auto pei_z = layout.physicalEndIndex(cent[2], Direction::Z);

        double error = 0.0;
        std::size_t count = 0;

        for (auto i = psi_x + 2; i <= pei_x - 2; ++i)
            for (auto j = psi_y + 2; j <= pei_y - 2; ++j)
                for (auto kk = psi_z + 2; kk <= pei_z - 2; ++kk)
                {
                    MeshIndex<3> idx{i, j, kk};
                    rho_pv(i, j, kk) = rho_avg(i, j, kk) - layout.lapl(rho_avg, idx) / 24.0;
                    auto c = layout.fieldNodeCoordinates(
                        rho_avg, layout.localToAMR(Point{i, j, kk}.as_signed()));
                    double exact = f3d(c[0], c[1], c[2]);
                    error += (rho_pv(i, j, kk) - exact) * (rho_pv(i, j, kk) - exact);
                    count++;
                }

        error = std::sqrt(error / count);
        errors.push_back(error);
        std::cout << "  n=" << n << " error=" << std::scientific << error << std::endl;
    }

    std::cout << "  Convergence orders:" << std::endl;
    for (std::size_t i = 1; i < errors.size(); ++i)
    {
        double order = convergenceOrder(errors[i - 1], errors[i]);
        std::cout << "    " << nCells[i - 1] << "->" << nCells[i] << ": " << std::fixed
                  << std::setprecision(2) << order << std::endl;
    }
}

//==============================================================================
// TEST 10: Face-centered point-value sanity check (Bx)
//==============================================================================

TEST(HallConvergence, PointValueFaceCenteredSanity3D)
{
    std::cout << "\n=== TEST 10: Face-centered point-value sanity (Bx) 3D ===" << std::endl;

    std::vector<std::size_t> nCells = {16, 32, 64};
    std::vector<double> errors;

    for (auto n : nCells)
    {
        double dx = 1.0 / n;
        using Layout = GridLayout<GridLayoutImplYeeMHD<3, 2>>;
        Layout layout{{{dx, dx, dx}},
                      {{static_cast<unsigned int>(n), static_cast<unsigned int>(n),
                        static_cast<unsigned int>(n)}},
                      Point{0.0, 0.0, 0.0}};

        UsableFieldMHD<3> Bx_avg{"Bx_avg", layout, MHDQuantity::Scalar::Bx};
        UsableFieldMHD<3> Bx_pv{"Bx_pv", layout, MHDQuantity::Scalar::Bx};

        auto cent = layout.centering(MHDQuantity::Scalar::Bx);
        auto gsi_x = layout.ghostStartIndex(cent[0], Direction::X);
        auto gei_x = layout.ghostEndIndex(cent[0], Direction::X);
        auto gsi_y = layout.ghostStartIndex(cent[1], Direction::Y);
        auto gei_y = layout.ghostEndIndex(cent[1], Direction::Y);
        auto gsi_z = layout.ghostStartIndex(cent[2], Direction::Z);
        auto gei_z = layout.ghostEndIndex(cent[2], Direction::Z);

        for (auto i = gsi_x; i <= gei_x; ++i)
            for (auto j = gsi_y; j <= gei_y; ++j)
                for (auto kk = gsi_z; kk <= gei_z; ++kk)
                {
                    auto c = layout.fieldNodeCoordinates(
                        Bx_avg, layout.localToAMR(Point{i, j, kk}.as_signed()));
                    double point = f3d(c[0], c[1], c[2]);
                    // For Bx face values, correction is transverse to X => Y+Z.
                    double lapl_t_analytic = d2f3d_dx2(c[0], c[1], c[2]) * (dx * dx) * 2.0;
                    Bx_avg(i, j, kk)       = point + lapl_t_analytic / 24.0;
                }

        auto psi_x = layout.physicalStartIndex(cent[0], Direction::X);
        auto pei_x = layout.physicalEndIndex(cent[0], Direction::X);
        auto psi_y = layout.physicalStartIndex(cent[1], Direction::Y);
        auto pei_y = layout.physicalEndIndex(cent[1], Direction::Y);
        auto psi_z = layout.physicalStartIndex(cent[2], Direction::Z);
        auto pei_z = layout.physicalEndIndex(cent[2], Direction::Z);

        double error = 0.0;
        std::size_t count = 0;

        for (auto i = psi_x + 2; i <= pei_x - 2; ++i)
            for (auto j = psi_y + 2; j <= pei_y - 2; ++j)
                for (auto kk = psi_z + 2; kk <= pei_z - 2; ++kk)
                {
                    MeshIndex<3> idx{i, j, kk};
                    Bx_pv(i, j, kk)
                        = Bx_avg(i, j, kk)
                          - layout.template tranverseLapl<Direction::X>(Bx_avg, idx) / 24.0;
                    auto c = layout.fieldNodeCoordinates(
                        Bx_avg, layout.localToAMR(Point{i, j, kk}.as_signed()));
                    double exact = f3d(c[0], c[1], c[2]);
                    error += (Bx_pv(i, j, kk) - exact) * (Bx_pv(i, j, kk) - exact);
                    count++;
                }

        error = std::sqrt(error / count);
        errors.push_back(error);
        std::cout << "  n=" << n << " error=" << std::scientific << error << std::endl;
    }

    std::cout << "  Convergence orders:" << std::endl;
    for (std::size_t i = 1; i < errors.size(); ++i)
    {
        double order = convergenceOrder(errors[i - 1], errors[i]);
        std::cout << "    " << nCells[i - 1] << "->" << nCells[i] << ": " << std::fixed
                  << std::setprecision(2) << order << std::endl;
    }
}

//==============================================================================
// TEST 11: Face-flux point-value sanity check (ScalarFlux_x)
//==============================================================================
TEST(HallConvergence, PointValueScalarFluxX3D)
{
    std::cout << "\n=== TEST 11: face-flux point-value sanity (ScalarFlux_x) 3D ===" << std::endl;

    std::vector<std::size_t> nCells = {16, 32, 64};
    std::vector<double> errors;

    for (auto n : nCells)
    {
        double dx = 1.0 / n;
        using Layout = GridLayout<GridLayoutImplYeeMHD<3, 2>>;
        Layout layout{{{dx, dx, dx}},
                      {{static_cast<unsigned int>(n), static_cast<unsigned int>(n),
                        static_cast<unsigned int>(n)}},
                      Point{0.0, 0.0, 0.0}};

        UsableFieldMHD<3> fx_avg{"fx_avg", layout, MHDQuantity::Scalar::ScalarFlux_x};
        UsableFieldMHD<3> fx_pv{"fx_pv", layout, MHDQuantity::Scalar::ScalarFlux_x};

        auto cent = layout.centering(MHDQuantity::Scalar::ScalarFlux_x);
        auto gsi_x = layout.ghostStartIndex(cent[0], Direction::X);
        auto gei_x = layout.ghostEndIndex(cent[0], Direction::X);
        auto gsi_y = layout.ghostStartIndex(cent[1], Direction::Y);
        auto gei_y = layout.ghostEndIndex(cent[1], Direction::Y);
        auto gsi_z = layout.ghostStartIndex(cent[2], Direction::Z);
        auto gei_z = layout.ghostEndIndex(cent[2], Direction::Z);

        for (auto i = gsi_x; i <= gei_x; ++i)
            for (auto j = gsi_y; j <= gei_y; ++j)
                for (auto kk = gsi_z; kk <= gei_z; ++kk)
                {
                    auto c = layout.fieldNodeCoordinates(
                        fx_avg, layout.localToAMR(Point{i, j, kk}.as_signed()));
                    double point = f3d(c[0], c[1], c[2]);
                    // X-face flux uses YZ-surface averaging => transverse (Y+Z) correction.
                    double lapl_t_analytic = d2f3d_dx2(c[0], c[1], c[2]) * (dx * dx) * 2.0;
                    fx_avg(i, j, kk)       = point + lapl_t_analytic / 24.0;
                }

        auto psi_x = layout.physicalStartIndex(cent[0], Direction::X);
        auto pei_x = layout.physicalEndIndex(cent[0], Direction::X);
        auto psi_y = layout.physicalStartIndex(cent[1], Direction::Y);
        auto pei_y = layout.physicalEndIndex(cent[1], Direction::Y);
        auto psi_z = layout.physicalStartIndex(cent[2], Direction::Z);
        auto pei_z = layout.physicalEndIndex(cent[2], Direction::Z);

        double error = 0.0;
        std::size_t count = 0;

        for (auto i = psi_x + 2; i <= pei_x - 2; ++i)
            for (auto j = psi_y + 2; j <= pei_y - 2; ++j)
                for (auto kk = psi_z + 2; kk <= pei_z - 2; ++kk)
                {
                    MeshIndex<3> idx{i, j, kk};
                    fx_pv(i, j, kk)
                        = fx_avg(i, j, kk)
                          - layout.template tranverseLapl<Direction::X>(fx_avg, idx) / 24.0;
                    auto c = layout.fieldNodeCoordinates(
                        fx_avg, layout.localToAMR(Point{i, j, kk}.as_signed()));
                    double exact = f3d(c[0], c[1], c[2]);
                    error += (fx_pv(i, j, kk) - exact) * (fx_pv(i, j, kk) - exact);
                    count++;
                }

        error = std::sqrt(error / count);
        errors.push_back(error);
        std::cout << "  n=" << n << " error=" << std::scientific << error << std::endl;
    }

    std::cout << "  Convergence orders:" << std::endl;
    for (std::size_t i = 1; i < errors.size(); ++i)
    {
        double order = convergenceOrder(errors[i - 1], errors[i]);
        std::cout << "    " << nCells[i - 1] << "->" << nCells[i] << ": " << std::fixed
                  << std::setprecision(2) << order << std::endl;
    }
}

//==============================================================================
// TEST 12: ToPrimitive converter on manufactured conservative state
//==============================================================================
TEST(HallConvergence, ToPrimitiveConverter3D)
{
    std::cout << "\n=== TEST 12: to_primitive converter 3D ===" << std::endl;

    std::vector<std::size_t> nCells = {16, 32, 64};
    std::vector<double> vErrors, pErrors;
    constexpr double gamma = 1.4;

    for (auto n : nCells)
    {
        double dx = 1.0 / n;
        using Layout = GridLayout<GridLayoutImplYeeMHD<3, 2>>;
        Layout layout{{{dx, dx, dx}},
                      {{static_cast<unsigned int>(n), static_cast<unsigned int>(n),
                        static_cast<unsigned int>(n)}},
                      Point{0.0, 0.0, 0.0}};

        UsableFieldMHD<3> rho{"rho", layout, MHDQuantity::Scalar::rho};
        UsableFieldMHD<3> Etot{"Etot", layout, MHDQuantity::Scalar::Etot};
        UsableFieldMHD<3> P{"P", layout, MHDQuantity::Scalar::P};
        UsableVecFieldMHD<3> rhoV{"rhoV", layout, MHDQuantity::Vector::rhoV};
        UsableVecFieldMHD<3> V{"V", layout, MHDQuantity::Vector::V};
        UsableVecFieldMHD<3> B{"B", layout, MHDQuantity::Vector::B};

        auto const& [rhoVx, rhoVy, rhoVz] = rhoV();
        auto const& [Vx, Vy, Vz]           = V();
        auto const& [Bx, By, Bz]           = B();

        // Fill primitive reference and conservative inputs.
        fillUsableField(layout, rho, [](double x, double y, double z) {
            return ExactHall3D::rho(x, y, z);
        });
        fillUsableField(layout, Bx, [](double x, double y, double z) {
            return ExactHall3D::bx(x, y, z);
        });
        fillUsableField(layout, By, [](double x, double y, double z) {
            return ExactHall3D::by(x, y, z);
        });
        fillUsableField(layout, Bz, [](double x, double y, double z) {
            return ExactHall3D::bz(x, y, z);
        });

        auto ccent = layout.centering(MHDQuantity::Scalar::rho);
        auto gsi_x = layout.ghostStartIndex(ccent[0], Direction::X);
        auto gei_x = layout.ghostEndIndex(ccent[0], Direction::X);
        auto gsi_y = layout.ghostStartIndex(ccent[1], Direction::Y);
        auto gei_y = layout.ghostEndIndex(ccent[1], Direction::Y);
        auto gsi_z = layout.ghostStartIndex(ccent[2], Direction::Z);
        auto gei_z = layout.ghostEndIndex(ccent[2], Direction::Z);

        for (auto i = gsi_x; i <= gei_x; ++i)
            for (auto j = gsi_y; j <= gei_y; ++j)
                for (auto kk = gsi_z; kk <= gei_z; ++kk)
                {
                    auto cc = layout.fieldNodeCoordinates(
                        rho, layout.localToAMR(Point{i, j, kk}.as_signed()));
                    auto r  = ExactHall3D::rho(cc[0], cc[1], cc[2]);
                    auto vx = ExactHall3D::vx(cc[0], cc[1], cc[2]);
                    auto vy = ExactHall3D::vy(cc[0], cc[1], cc[2]);
                    auto vz = ExactHall3D::vz(cc[0], cc[1], cc[2]);
                    rhoVx(i, j, kk) = r * vx;
                    rhoVy(i, j, kk) = r * vy;
                    rhoVz(i, j, kk) = r * vz;
                    Etot(i, j, kk)  = ExactHall3D::etot(cc[0], cc[1], cc[2]);
                }

        ToPrimitiveConverter_ref<Layout>{layout}(gamma, rho, rhoV, B, Etot, V, P);

        auto psi_x = layout.physicalStartIndex(ccent[0], Direction::X);
        auto pei_x = layout.physicalEndIndex(ccent[0], Direction::X);
        auto psi_y = layout.physicalStartIndex(ccent[1], Direction::Y);
        auto pei_y = layout.physicalEndIndex(ccent[1], Direction::Y);
        auto psi_z = layout.physicalStartIndex(ccent[2], Direction::Z);
        auto pei_z = layout.physicalEndIndex(ccent[2], Direction::Z);

        double verr = 0.0, perr = 0.0;
        std::size_t cnt = 0;
        for (auto i = psi_x + 4; i <= pei_x - 4; ++i)
            for (auto j = psi_y + 4; j <= pei_y - 4; ++j)
                for (auto kk = psi_z + 4; kk <= pei_z - 4; ++kk)
                {
                    auto cc = layout.fieldNodeCoordinates(
                        rho, layout.localToAMR(Point{i, j, kk}.as_signed()));
                    auto dvx = Vx(i, j, kk) - ExactHall3D::vx(cc[0], cc[1], cc[2]);
                    auto dvy = Vy(i, j, kk) - ExactHall3D::vy(cc[0], cc[1], cc[2]);
                    auto dvz = Vz(i, j, kk) - ExactHall3D::vz(cc[0], cc[1], cc[2]);
                    auto dp  = P(i, j, kk) - ExactHall3D::pressure(cc[0], cc[1], cc[2]);
                    verr += dvx * dvx + dvy * dvy + dvz * dvz;
                    perr += dp * dp;
                    ++cnt;
                }

        vErrors.push_back(std::sqrt(verr / (3.0 * cnt)));
        pErrors.push_back(std::sqrt(perr / cnt));
        std::cout << "  n=" << n << " Verr=" << std::scientific << vErrors.back()
                  << " Perr=" << pErrors.back() << std::endl;
    }

    std::cout << "  Convergence orders (V):" << std::endl;
    for (std::size_t i = 1; i < vErrors.size(); ++i)
    {
        std::cout << "    " << nCells[i - 1] << "->" << nCells[i] << ": " << std::fixed
                  << std::setprecision(2) << convergenceOrder(vErrors[i - 1], vErrors[i])
                  << std::endl;
    }
    std::cout << "  Convergence orders (P):" << std::endl;
    for (std::size_t i = 1; i < pErrors.size(); ++i)
    {
        std::cout << "    " << nCells[i - 1] << "->" << nCells[i] << ": " << std::fixed
                  << std::setprecision(2) << convergenceOrder(pErrors[i - 1], pErrors[i])
                  << std::endl;
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
