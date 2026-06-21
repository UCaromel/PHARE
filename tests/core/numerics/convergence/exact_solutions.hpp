/**
 * @file exact_solutions.hpp
 * 
 * Analytical solutions for MHD convergence testing.
 * 
 * Each solution provides primitive state (ρ, V, B, P), conserved quantities,
 * fluxes, and derived quantities (J, E) for use in convergence tests.
 * 
 * All solutions use domain [0,1]³ with periodic boundaries and k = 2π wavenumber.
 */

#pragma once

#include <array>
#include <cmath>
#include "core/def.hpp"
#include "core/utilities/types.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"

namespace PHARE::test
{

constexpr double k = 2.0 * M_PI;  ///< Wavenumber for periodic domain [0,1]

/**
 * @brief Simple sinusoidal functions for basic tests
 */
/**
 * @brief Fill field with cell-averaged values using 4th-order Gauss-Legendre quadrature
 * 
 * Uses 2-point Gauss-Legendre quadrature in each dimension (2³ = 8 points per cell)
 * This gives 4th-order accurate cell-averages for smooth functions.
 */
template<typename Layout, typename Field, typename Func>
inline void fillCellAveragedField(Layout const& layout, Field& field, Func&& fn)
{
    static constexpr double gl_pt = 0.28867513459481287; // 1 / (2*sqrt(3))
    static constexpr double w     = 0.5;
    static constexpr double www   = w * w * w; // 0.125
    
    auto cent = layout.centering(field.physicalQuantity());
    auto meshSize = layout.meshSize();
    double dx = meshSize[static_cast<int>(PHARE::core::Direction::X)];
    double dy = meshSize[static_cast<int>(PHARE::core::Direction::Y)];
    double dz = meshSize[static_cast<int>(PHARE::core::Direction::Z)];
    
    for (auto i = layout.ghostStartIndex(cent[0], PHARE::core::Direction::X);
         i <= layout.ghostEndIndex(cent[0], PHARE::core::Direction::X); ++i)
        for (auto j = layout.ghostStartIndex(cent[1], PHARE::core::Direction::Y);
             j <= layout.ghostEndIndex(cent[1], PHARE::core::Direction::Y); ++j)
            for (auto kk = layout.ghostStartIndex(cent[2], PHARE::core::Direction::Z);
                 kk <= layout.ghostEndIndex(cent[2], PHARE::core::Direction::Z); ++kk)
            {
                auto c = layout.fieldNodeCoordinates(
                    field, layout.localToAMR(PHARE::core::Point{i, j, kk}.as_signed()));
                
                // 2³ = 8 quadrature points per cell
                double avg = 0.0;
                for (double sx : {-gl_pt, +gl_pt})
                    for (double sy : {-gl_pt, +gl_pt})
                        for (double sz : {-gl_pt, +gl_pt})
                        {
                            double x = c[0] + sx * dx;
                            double y = c[1] + sy * dy;
                            double z = c[2] + sz * dz;
                            avg += www * fn(x, y, z);
                        }
                field(i, j, kk) = avg;
            }
}

/**
 * @brief Fill face-averaged field using 4th-order Gauss-Legendre quadrature
 * 
 * Uses 2-point Gauss-Legendre quadrature in transverse directions (2² = 4 points per face)
 * Face-average in direction dir integrates over the transverse plane.
 */
template<typename Layout, typename Field, typename Func, PHARE::core::Direction dir>
inline void fillFaceAveragedField(Layout const& layout, Field& field, Func&& fn)
{
    static constexpr double gl_pt = 0.28867513459481287; // 1 / (2*sqrt(3))
    static constexpr double w     = 0.5;
    static constexpr double ww    = w * w; // 0.25
    
    auto cent = layout.centering(field.physicalQuantity());
    auto meshSize = layout.meshSize();
    double dx = meshSize[static_cast<int>(PHARE::core::Direction::X)];
    double dy = meshSize[static_cast<int>(PHARE::core::Direction::Y)];
    double dz = meshSize[static_cast<int>(PHARE::core::Direction::Z)];
    
    for (auto i = layout.ghostStartIndex(cent[0], PHARE::core::Direction::X);
         i <= layout.ghostEndIndex(cent[0], PHARE::core::Direction::X); ++i)
        for (auto j = layout.ghostStartIndex(cent[1], PHARE::core::Direction::Y);
             j <= layout.ghostEndIndex(cent[1], PHARE::core::Direction::Y); ++j)
            for (auto kk = layout.ghostStartIndex(cent[2], PHARE::core::Direction::Z);
                 kk <= layout.ghostEndIndex(cent[2], PHARE::core::Direction::Z); ++kk)
            {
                auto c = layout.fieldNodeCoordinates(
                    field, layout.localToAMR(PHARE::core::Point{i, j, kk}.as_signed()));
                
                // 2² = 4 quadrature points per face (integrate in transverse directions)
                double avg = 0.0;
                
                if constexpr (dir == PHARE::core::Direction::X) {
                    // X-face: integrate over Y-Z plane
                    for (double sy : {-gl_pt, +gl_pt})
                        for (double sz : {-gl_pt, +gl_pt})
                            avg += ww * fn(c[0], c[1] + sy * dy, c[2] + sz * dz);
                }
                else if constexpr (dir == PHARE::core::Direction::Y) {
                    // Y-face: integrate over X-Z plane
                    for (double sx : {-gl_pt, +gl_pt})
                        for (double sz : {-gl_pt, +gl_pt})
                            avg += ww * fn(c[0] + sx * dx, c[1], c[2] + sz * dz);
                }
                else if constexpr (dir == PHARE::core::Direction::Z) {
                    // Z-face: integrate over X-Y plane
                    for (double sx : {-gl_pt, +gl_pt})
                        for (double sy : {-gl_pt, +gl_pt})
                            avg += ww * fn(c[0] + sx * dx, c[1] + sy * dy, c[2]);
                }
                
                field(i, j, kk) = avg;
            }
}

inline double f(double x) { return std::sin(k * x); }
inline double df(double x) { return k * std::cos(k * x); }
inline double f3d(double x, double y, double z)
{
    return std::sin(k * x) * std::sin(k * y) * std::sin(k * z);
}
inline double d2f3d_dx2(double x, double y, double z) { return -k * k * f3d(x, y, z); }

/**
 * @brief Simple B-field for curl tests
 */
inline double Bx_func(double x, double y, double z) { return std::sin(k * y) * std::sin(k * z); }
inline double By_func(double x, double y, double z) { return std::sin(k * x) * std::sin(k * z); }
inline double Bz_func(double x, double y, double z) { return std::sin(k * x) * std::sin(k * y); }

/**
 * @brief Exact current from curl of simple B-field
 * 
 * J = curl(B) where B = (sin(ky)sin(kz), sin(kx)sin(kz), sin(kx)sin(ky))
 */
inline double Jx_exact(double x, double y, double z)
{
    return k * std::cos(k * y) * std::sin(k * x) - k * std::cos(k * z) * std::sin(k * x);
}
inline double Jy_exact(double x, double y, double z)
{
    return k * std::cos(k * z) * std::sin(k * y) - k * std::cos(k * x) * std::sin(k * y);
}
inline double Jz_exact(double x, double y, double z)
{
    return k * std::cos(k * x) * std::sin(k * z) - k * std::cos(k * y) * std::sin(k * z);
}

/**
 * @brief Full 3D Hall MHD test case with smooth variations in all directions
 * 
 * This is a manufactured solution (not an MHD equilibrium) designed to test
 * pointwise flux accuracy. The state is smooth and periodic but does not
 * satisfy ∂U/∂t = 0 (it would evolve in time if integrated).
 * 
 * Primitive state:
 * - ρ = 1.5 + 0.1*sin(kx) + 0.07*cos(ky) + 0.05*sin(kz)
 * - V = smooth 3D variations with cross-coupling
 * - B = divergence-free: ∇·B = 0
 * - P = 1.2 + 0.1*cos(k(x+y)) + 0.05*sin(kz)
 * 
 * Derived quantities:
 * - J = curl(B) - exact analytical formula
 * - E_ideal = -V×B
 * - E_Hall = -V×B + (J×B)/ρ
 * - Fluxes: Full MHD with optional Hall energy correction
 */
struct ExactHall3D
{
    //-------------------------------------------------------------------------
    // Primitive State
    //-------------------------------------------------------------------------
    
    static double rho(double x, double y, double z)
    {
        return 1.5 + 0.1 * std::sin(k * x) + 0.07 * std::cos(k * y) + 0.05 * std::sin(k * z);
    }

    static double vx(double x, double y, double z)
    {
        return 0.2 * std::sin(k * x) + 0.1 * std::cos(k * y) + 0.03 * std::sin(k * z);
    }

    static double vy(double x, double y, double z)
    {
        return -0.15 * std::cos(k * y) + 0.08 * std::sin(k * z) + 0.02 * std::cos(k * x);
    }

    static double vz(double x, double y, double z)
    {
        return 0.12 * std::sin(k * z) + 0.06 * std::cos(k * x) + 0.02 * std::sin(k * y);
    }

    /**
     * @brief Magnetic field (divergence-free by construction)
     * 
     * Verification: ∂Bx/∂x + ∂By/∂y + ∂Bz/∂z = 0
     */
    static double bx(double, double y, double z) 
    { 
        return 0.3 + 0.1 * std::sin(k * y) * std::sin(k * z); 
    }
    
    static double by(double x, double, double z) 
    { 
        return 0.2 + 0.1 * std::sin(k * x) * std::sin(k * z); 
    }
    
    static double bz(double x, double y, double) 
    { 
        return -0.1 + 0.1 * std::sin(k * x) * std::sin(k * y); 
    }

    static double pressure(double x, double y, double z)
    {
        return 1.2 + 0.1 * std::cos(k * (x + y)) + 0.05 * std::sin(k * z);
    }

    //-------------------------------------------------------------------------
    // Derived Quantities
    //-------------------------------------------------------------------------
    
    /**
     * @brief Current density J = curl(B)
     * 
     * Derived from analytical curl of B field above.
     * Verified with SymPy symbolic differentiation.
     */
    static std::array<double, 3> current(double x, double y, double z)
    {
        auto const jx = 0.1 * k * std::sin(k * x) * (std::cos(k * y) - std::cos(k * z));
        auto const jy = 0.1 * k * std::sin(k * y) * (std::cos(k * z) - std::cos(k * x));
        auto const jz = 0.1 * k * std::sin(k * z) * (std::cos(k * x) - std::cos(k * y));
        return {jx, jy, jz};
    }

    /**
     * @brief Total energy E = P/(γ-1) + (1/2)ρv² + (1/2)B²
     */
    static double etot(double x, double y, double z)
    {
        auto const r  = rho(x, y, z);
        auto const vX = vx(x, y, z);
        auto const vY = vy(x, y, z);
        auto const vZ = vz(x, y, z);
        auto const bX = bx(x, y, z);
        auto const bY = by(x, y, z);
        auto const bZ = bz(x, y, z);
        auto const p  = pressure(x, y, z);
        constexpr double gamma = 1.4;
        return p / (gamma - 1.0) + 0.5 * r * (vX * vX + vY * vY + vZ * vZ)
               + 0.5 * (bX * bX + bY * bY + bZ * bZ);
    }

    //-------------------------------------------------------------------------
    // MHD Fluxes (with optional Hall energy correction)
    //-------------------------------------------------------------------------
    
    /**
     * @brief MHD flux in specified direction
     * 
     * Returns [F_rho, F_rhoVx, F_rhoVy, F_rhoVz, F_Etot]
     * 
     * Ideal MHD fluxes:
     * - F_rho = ρV_i
     * - F_rhoV = ρV_iV + (P + B²/2)I - B_iB
     * - F_E = (E + P + B²/2)V_i - B_i(V·B)
     * 
     * Hall energy correction (if include_hall = true):
     * - F_E += [(B·J)B_i - (B·B)J_i]/ρ
     * 
     * @param dir Direction (X, Y, or Z)
     * @param x,y,z Spatial coordinates
     * @param include_hall Include Hall energy correction
     * @return Array of 5 flux components
     */
    static std::array<double, 5> flux(PHARE::core::Direction dir, double x, double y, double z,
                                      bool include_hall = true)
    {
        using PHARE::core::Direction;
        
        auto const r  = rho(x, y, z);
        auto const vX = vx(x, y, z);
        auto const vY = vy(x, y, z);
        auto const vZ = vz(x, y, z);
        auto const bX = bx(x, y, z);
        auto const bY = by(x, y, z);
        auto const bZ = bz(x, y, z);
        auto const p  = pressure(x, y, z);
        auto const eT = etot(x, y, z);

        // Gas + magnetic pressure
        auto const gp = p + 0.5 * (bX * bX + bY * bY + bZ * bZ);

        // Ideal MHD fluxes
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
        else // Direction::Z
        {
            frho   = r * vZ;
            frhoVx = r * vZ * vX - bZ * bX;
            frhoVy = r * vZ * vY - bZ * bY;
            frhoVz = r * vZ * vZ + gp - bZ * bZ;
            fetot  = (eT + gp) * vZ - bZ * (vX * bX + vY * bY + vZ * bZ);
        }

        // Hall energy correction: [(B·J)B_i - (B·B)J_i]/ρ
        if (include_hall)
        {
            auto const j  = current(x, y, z);
            auto const jX = j[0];
            auto const jY = j[1];
            auto const jZ = j[2];
            
            auto const bdotJ = bX * jX + bY * jY + bZ * jZ;
            auto const bdotB = bX * bX + bY * bY + bZ * bZ;
            
            if (dir == Direction::X)
                fetot += (bdotJ * bX - bdotB * jX) / r;
            else if (dir == Direction::Y)
                fetot += (bdotJ * bY - bdotB * jY) / r;
            else
                fetot += (bdotJ * bZ - bdotB * jZ) / r;
        }

        return {frho, frhoVx, frhoVy, frhoVz, fetot};
    }

    //-------------------------------------------------------------------------
    // Electric Field
    //-------------------------------------------------------------------------
    
    /**
     * @brief Electric field E = -V×B + (J×B)/ρ (Hall MHD)
     * 
     * @param include_hall If false, returns only ideal term -V×B
     */
    static std::array<double, 3> electric(double x, double y, double z, bool include_hall = true)
    {
        auto const r  = rho(x, y, z);
        auto const vX = vx(x, y, z);
        auto const vY = vy(x, y, z);
        auto const vZ = vz(x, y, z);
        auto const bX = bx(x, y, z);
        auto const bY = by(x, y, z);
        auto const bZ = bz(x, y, z);

        // Ideal term: -V×B
        auto ex = -(vY * bZ - vZ * bY);
        auto ey = -(vZ * bX - vX * bZ);
        auto ez = -(vX * bY - vY * bX);

        // Hall term: (J×B)/ρ
        if (include_hall)
        {
            auto const j  = current(x, y, z);
            auto const jX = j[0];
            auto const jY = j[1];
            auto const jZ = j[2];
            
            ex += (jY * bZ - jZ * bY) / r;
            ey += (jZ * bX - jX * bZ) / r;
            ez += (jX * bY - jY * bX) / r;
        }

        return {ex, ey, ez};
    }
};

/**
 * @brief Ideal MHD version of ExactHall3D
 * 
 * Same primitive state but without Hall corrections.
 * Use for testing ideal MHD implementation.
 */
struct ExactIdealMHD3D
{
    // Primitive state identical to Hall case
    static double rho(double x, double y, double z) { return ExactHall3D::rho(x, y, z); }
    static double vx(double x, double y, double z) { return ExactHall3D::vx(x, y, z); }
    static double vy(double x, double y, double z) { return ExactHall3D::vy(x, y, z); }
    static double vz(double x, double y, double z) { return ExactHall3D::vz(x, y, z); }
    static double bx(double x, double y, double z) { return ExactHall3D::bx(x, y, z); }
    static double by(double x, double y, double z) { return ExactHall3D::by(x, y, z); }
    static double bz(double x, double y, double z) { return ExactHall3D::bz(x, y, z); }
    static double pressure(double x, double y, double z) { return ExactHall3D::pressure(x, y, z); }
    static double etot(double x, double y, double z) { return ExactHall3D::etot(x, y, z); }

    // Flux without Hall correction
    static std::array<double, 5> flux(PHARE::core::Direction dir, double x, double y, double z)
    {
        return ExactHall3D::flux(dir, x, y, z, false);  // include_hall = false
    }

    // E-field: ideal only (no Hall term)
    static std::array<double, 3> electric(double x, double y, double z)
    {
        return ExactHall3D::electric(x, y, z, false);  // include_hall = false
    }
};

/**
 * @brief Compute face-averaged exact flux using 4th-order Gauss-Legendre quadrature
 * 
 * For integral/cell-averaged fluxes, we need face-averaged exact values for proper comparison.
 * Uses 2-point GL quadrature in transverse directions (4 sample points per face).
 */
template<typename Layout, PHARE::core::Direction dir, typename FluxFunc>
inline double l2FaceAveragedFluxError(Layout const& layout, auto const& field, FluxFunc&& exactFlux)
{
    static constexpr double gl_pt = 0.28867513459481287; // 1 / (2*sqrt(3))
    static constexpr double w     = 0.5;
    static constexpr double ww    = w * w; // 0.25
    
    auto cent = layout.centering(field.physicalQuantity());
    auto meshSize = layout.meshSize();
    double dx = meshSize[static_cast<int>(PHARE::core::Direction::X)];
    double dy = meshSize[static_cast<int>(PHARE::core::Direction::Y)];
    double dz = meshSize[static_cast<int>(PHARE::core::Direction::Z)];
    
    auto psiX = layout.physicalStartIndex(cent[0], PHARE::core::Direction::X);
    auto peiX = layout.physicalEndIndex(cent[0], PHARE::core::Direction::X);
    auto psiY = layout.physicalStartIndex(cent[1], PHARE::core::Direction::Y);
    auto peiY = layout.physicalEndIndex(cent[1], PHARE::core::Direction::Y);
    auto psiZ = layout.physicalStartIndex(cent[2], PHARE::core::Direction::Z);
    auto peiZ = layout.physicalEndIndex(cent[2], PHARE::core::Direction::Z);
    
    double err = 0.0;
    std::size_t count = 0;
    constexpr int margin = 6;
    
    for (auto i = psiX + margin; i <= peiX - margin; ++i)
        for (auto j = psiY + margin; j <= peiY - margin; ++j)
            for (auto kk = psiZ + margin; kk <= peiZ - margin; ++kk)
            {
                auto c = layout.fieldNodeCoordinates(
                    field, layout.localToAMR(PHARE::core::Point{i, j, kk}.as_signed()));
                
                // Compute face-averaged exact flux using GL quadrature
                double exact_avg = 0.0;
                
                if constexpr (dir == PHARE::core::Direction::X) {
                    // X-face: integrate over Y-Z plane
                    for (double sy : {-gl_pt, +gl_pt})
                        for (double sz : {-gl_pt, +gl_pt})
                            exact_avg += ww * exactFlux(c[0], c[1] + sy * dy, c[2] + sz * dz);
                }
                else if constexpr (dir == PHARE::core::Direction::Y) {
                    // Y-face: integrate over X-Z plane
                    for (double sx : {-gl_pt, +gl_pt})
                        for (double sz : {-gl_pt, +gl_pt})
                            exact_avg += ww * exactFlux(c[0] + sx * dx, c[1], c[2] + sz * dz);
                }
                else if constexpr (dir == PHARE::core::Direction::Z) {
                    // Z-face: integrate over X-Y plane
                    for (double sx : {-gl_pt, +gl_pt})
                        for (double sy : {-gl_pt, +gl_pt})
                            exact_avg += ww * exactFlux(c[0] + sx * dx, c[1] + sy * dy, c[2]);
                }
                
                auto diff = field(i, j, kk) - exact_avg;
                err += diff * diff;
                ++count;
            }
    
    return std::sqrt(err / static_cast<double>(count));
}

/**
 * @brief Compute L2 error between cell-centered field and cell-averaged exact function
 *
 * Uses 8-point Gauss-Legendre quadrature (2³) to compute exact cell averages,
 * then compares against the stored field values.
 *
 * Without fix/init-pointvalue-conversion: rhoV and Etot are stored as point
 * values → O(dx²) error here.
 * With fix: they are stored as proper cell averages → O(dx⁴) error.
 */
template<typename Layout, typename ExactFunc>
inline double l2CellAveragedError(Layout const& layout, auto const& field, ExactFunc&& exactFn)
{
    static constexpr double gl_pt = 0.28867513459481287; // 1 / (2*sqrt(3))
    static constexpr double w     = 0.5;
    static constexpr double www   = w * w * w; // 0.125

    auto cent = layout.centering(field.physicalQuantity());
    auto meshSize = layout.meshSize();
    double dx = meshSize[static_cast<int>(PHARE::core::Direction::X)];
    double dy = meshSize[static_cast<int>(PHARE::core::Direction::Y)];
    double dz = meshSize[static_cast<int>(PHARE::core::Direction::Z)];

    auto psiX = layout.physicalStartIndex(cent[0], PHARE::core::Direction::X);
    auto peiX = layout.physicalEndIndex(cent[0], PHARE::core::Direction::X);
    auto psiY = layout.physicalStartIndex(cent[1], PHARE::core::Direction::Y);
    auto peiY = layout.physicalEndIndex(cent[1], PHARE::core::Direction::Y);
    auto psiZ = layout.physicalStartIndex(cent[2], PHARE::core::Direction::Z);
    auto peiZ = layout.physicalEndIndex(cent[2], PHARE::core::Direction::Z);

    double err   = 0.0;
    std::size_t count = 0;
    constexpr int margin = 6;

    for (auto i = psiX + margin; i <= peiX - margin; ++i)
        for (auto j = psiY + margin; j <= peiY - margin; ++j)
            for (auto kk = psiZ + margin; kk <= peiZ - margin; ++kk)
            {
                auto c = layout.fieldNodeCoordinates(
                    field, layout.localToAMR(PHARE::core::Point{i, j, kk}.as_signed()));

                // 8-point GL cell average of exact function
                double exact_avg = 0.0;
                for (double sx : {-gl_pt, +gl_pt})
                    for (double sy : {-gl_pt, +gl_pt})
                        for (double sz : {-gl_pt, +gl_pt})
                            exact_avg += www * exactFn(c[0] + sx * dx, c[1] + sy * dy, c[2] + sz * dz);

                auto diff = field(i, j, kk) - exact_avg;
                err += diff * diff;
                ++count;
            }

    return std::sqrt(err / static_cast<double>(count));
}

/**
 * @brief Compute edge-averaged exact value using 4th-order Gauss-Legendre quadrature
 * 
 * For edge-centered fields (like E), we need line-averaged exact values.
 * Uses 2-point GL quadrature along the edge direction (2 sample points per edge).
 */
template<typename Layout, PHARE::core::Direction dir, typename EdgeFunc>
inline double l2EdgeAveragedError(Layout const& layout, auto const& field, EdgeFunc&& exactEdge)
{
    static constexpr double gl_pt = 0.28867513459481287; // 1 / (2*sqrt(3))
    static constexpr double w     = 0.5;
    
    auto cent = layout.centering(field.physicalQuantity());
    auto meshSize = layout.meshSize();
    double dx = meshSize[static_cast<int>(PHARE::core::Direction::X)];
    double dy = meshSize[static_cast<int>(PHARE::core::Direction::Y)];
    double dz = meshSize[static_cast<int>(PHARE::core::Direction::Z)];
    
    auto psiX = layout.physicalStartIndex(cent[0], PHARE::core::Direction::X);
    auto peiX = layout.physicalEndIndex(cent[0], PHARE::core::Direction::X);
    auto psiY = layout.physicalStartIndex(cent[1], PHARE::core::Direction::Y);
    auto peiY = layout.physicalEndIndex(cent[1], PHARE::core::Direction::Y);
    auto psiZ = layout.physicalStartIndex(cent[2], PHARE::core::Direction::Z);
    auto peiZ = layout.physicalEndIndex(cent[2], PHARE::core::Direction::Z);
    
    double err = 0.0;
    std::size_t count = 0;
    constexpr int margin = 6;
    
    for (auto i = psiX + margin; i <= peiX - margin; ++i)
        for (auto j = psiY + margin; j <= peiY - margin; ++j)
            for (auto kk = psiZ + margin; kk <= peiZ - margin; ++kk)
            {
                auto c = layout.fieldNodeCoordinates(
                    field, layout.localToAMR(PHARE::core::Point{i, j, kk}.as_signed()));
                
                // Compute edge-averaged exact value using GL quadrature (line integral)
                double exact_avg = 0.0;
                
                if constexpr (dir == PHARE::core::Direction::X) {
                    // X-edge: integrate along X direction
                    for (double sx : {-gl_pt, +gl_pt})
                        exact_avg += w * exactEdge(c[0] + sx * dx, c[1], c[2]);
                }
                else if constexpr (dir == PHARE::core::Direction::Y) {
                    // Y-edge: integrate along Y direction
                    for (double sy : {-gl_pt, +gl_pt})
                        exact_avg += w * exactEdge(c[0], c[1] + sy * dy, c[2]);
                }
                else if constexpr (dir == PHARE::core::Direction::Z) {
                    // Z-edge: integrate along Z direction
                    for (double sz : {-gl_pt, +gl_pt})
                        exact_avg += w * exactEdge(c[0], c[1], c[2] + sz * dz);
                }
                
                auto diff = field(i, j, kk) - exact_avg;
                err += diff * diff;
                ++count;
            }
    
    return std::sqrt(err / static_cast<double>(count));
}

} // namespace PHARE::test
