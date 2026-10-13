/**
 * Analytical solutions for MHD convergence tests: primitive state (ρ, V, B, P), conserved
 * quantities, fluxes and derived quantities (J, E), on the periodic domain [0,1]³ with k = 2π.
 */

#pragma once

#include <array>
#include <cmath>
#include "core/def.hpp"
#include "core/utilities/types.hpp"
#include "core/data/grid/gridlayoutdefs.hpp"

namespace PHARE::test
{

constexpr double k = 2.0 * M_PI; ///< Wavenumber for periodic domain [0,1]

/**
 * @brief Fill field with cell averages: 2-point Gauss-Legendre per direction (8 nodes per cell),
 * exact for cubics, so fourth order.
 */
template<typename Layout, typename Field, typename Func>
inline void fillCellAveragedField(Layout const& layout, Field& field, Func&& fn)
{
    static constexpr double gl_pt = 0.28867513459481287; // 1 / (2*sqrt(3))
    static constexpr double w     = 0.5;
    static constexpr double www   = w * w * w; // 0.125

    auto cent     = layout.centering(field.physicalQuantity());
    auto meshSize = layout.meshSize();
    double dx     = meshSize[static_cast<int>(PHARE::core::Direction::X)];
    double dy     = meshSize[static_cast<int>(PHARE::core::Direction::Y)];
    double dz     = meshSize[static_cast<int>(PHARE::core::Direction::Z)];

    for (auto i = layout.ghostStartIndex(cent[0], PHARE::core::Direction::X);
         i <= layout.ghostEndIndex(cent[0], PHARE::core::Direction::X); ++i)
        for (auto j = layout.ghostStartIndex(cent[1], PHARE::core::Direction::Y);
             j <= layout.ghostEndIndex(cent[1], PHARE::core::Direction::Y); ++j)
            for (auto kk = layout.ghostStartIndex(cent[2], PHARE::core::Direction::Z);
                 kk <= layout.ghostEndIndex(cent[2], PHARE::core::Direction::Z); ++kk)
            {
                auto c = layout.fieldNodeCoordinates(
                    field, layout.localToAMR(PHARE::core::Point{i, j, kk}.as_signed()));

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
 * @brief Fill a `dir`-face-centred field with face averages: 2-point Gauss-Legendre in the two
 * transverse directions, fourth order.
 */
template<typename Layout, typename Field, typename Func, PHARE::core::Direction dir>
inline void fillFaceAveragedField(Layout const& layout, Field& field, Func&& fn)
{
    static constexpr double gl_pt = 0.28867513459481287; // 1 / (2*sqrt(3))
    static constexpr double w     = 0.5;
    static constexpr double ww    = w * w; // 0.25

    auto cent     = layout.centering(field.physicalQuantity());
    auto meshSize = layout.meshSize();
    double dx     = meshSize[static_cast<int>(PHARE::core::Direction::X)];
    double dy     = meshSize[static_cast<int>(PHARE::core::Direction::Y)];
    double dz     = meshSize[static_cast<int>(PHARE::core::Direction::Z)];

    for (auto i = layout.ghostStartIndex(cent[0], PHARE::core::Direction::X);
         i <= layout.ghostEndIndex(cent[0], PHARE::core::Direction::X); ++i)
        for (auto j = layout.ghostStartIndex(cent[1], PHARE::core::Direction::Y);
             j <= layout.ghostEndIndex(cent[1], PHARE::core::Direction::Y); ++j)
            for (auto kk = layout.ghostStartIndex(cent[2], PHARE::core::Direction::Z);
                 kk <= layout.ghostEndIndex(cent[2], PHARE::core::Direction::Z); ++kk)
            {
                auto c = layout.fieldNodeCoordinates(
                    field, layout.localToAMR(PHARE::core::Point{i, j, kk}.as_signed()));

                double avg = 0.0;

                if constexpr (dir == PHARE::core::Direction::X)
                {
                    for (double sy : {-gl_pt, +gl_pt})
                        for (double sz : {-gl_pt, +gl_pt})
                            avg += ww * fn(c[0], c[1] + sy * dy, c[2] + sz * dz);
                }
                else if constexpr (dir == PHARE::core::Direction::Y)
                {
                    for (double sx : {-gl_pt, +gl_pt})
                        for (double sz : {-gl_pt, +gl_pt})
                            avg += ww * fn(c[0] + sx * dx, c[1], c[2] + sz * dz);
                }
                else if constexpr (dir == PHARE::core::Direction::Z)
                {
                    for (double sx : {-gl_pt, +gl_pt})
                        for (double sy : {-gl_pt, +gl_pt})
                            avg += ww * fn(c[0] + sx * dx, c[1] + sy * dy, c[2]);
                }

                field(i, j, kk) = avg;
            }
}

/**
 * @brief Full 3D Hall MHD test case with smooth variations in all directions
 *
 * This is a manufactured solution (not an MHD equilibrium) designed to test
 * pointwise flux accuracy. The state is smooth and periodic but does not
 * satisfy ∂U/∂t = 0 (it would evolve in time if integrated).
 *
 * B is divergence-free; J, E and the fluxes are its exact derivatives.
 */
struct ExactHall3D
{
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

    // each component is independent of its own coordinate, so ∇·B = 0
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

    /** @brief Current density J = curl(B) */
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
        auto const r           = rho(x, y, z);
        auto const vX          = vx(x, y, z);
        auto const vY          = vy(x, y, z);
        auto const vZ          = vz(x, y, z);
        auto const bX          = bx(x, y, z);
        auto const bY          = by(x, y, z);
        auto const bZ          = bz(x, y, z);
        auto const p           = pressure(x, y, z);
        constexpr double gamma = 1.4;
        return p / (gamma - 1.0) + 0.5 * r * (vX * vX + vY * vY + vZ * vZ)
               + 0.5 * (bX * bX + bY * bY + bZ * bZ);
    }

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
        else // Direction::Z
        {
            frho   = r * vZ;
            frhoVx = r * vZ * vX - bZ * bX;
            frhoVy = r * vZ * vY - bZ * bY;
            frhoVz = r * vZ * vZ + gp - bZ * bZ;
            fetot  = (eT + gp) * vZ - bZ * (vX * bX + vY * bY + vZ * bZ);
        }

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

        auto ex = -(vY * bZ - vZ * bY);
        auto ey = -(vZ * bX - vX * bZ);
        auto ez = -(vX * bY - vY * bX);

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

/** @brief ExactHall3D's state, with ideal-MHD fluxes and electric field */
struct ExactIdealMHD3D
{
    static double rho(double x, double y, double z) { return ExactHall3D::rho(x, y, z); }
    static double vx(double x, double y, double z) { return ExactHall3D::vx(x, y, z); }
    static double vy(double x, double y, double z) { return ExactHall3D::vy(x, y, z); }
    static double vz(double x, double y, double z) { return ExactHall3D::vz(x, y, z); }
    static double bx(double x, double y, double z) { return ExactHall3D::bx(x, y, z); }
    static double by(double x, double y, double z) { return ExactHall3D::by(x, y, z); }
    static double bz(double x, double y, double z) { return ExactHall3D::bz(x, y, z); }
    static double pressure(double x, double y, double z) { return ExactHall3D::pressure(x, y, z); }
    static double etot(double x, double y, double z) { return ExactHall3D::etot(x, y, z); }

    static std::array<double, 5> flux(PHARE::core::Direction dir, double x, double y, double z)
    {
        return ExactHall3D::flux(dir, x, y, z, false); // include_hall = false
    }

    static std::array<double, 3> electric(double x, double y, double z)
    {
        return ExactHall3D::electric(x, y, z, false); // include_hall = false
    }
};

/**
 * @brief L2 error of a face-averaged flux against the exact face average (2-point Gauss-Legendre
 * in the transverse directions).
 */
template<typename Layout, PHARE::core::Direction dir, typename FluxFunc>
inline double l2FaceAveragedFluxError(Layout const& layout, auto const& field, FluxFunc&& exactFlux)
{
    static constexpr double gl_pt = 0.28867513459481287; // 1 / (2*sqrt(3))
    static constexpr double w     = 0.5;
    static constexpr double ww    = w * w; // 0.25

    auto cent     = layout.centering(field.physicalQuantity());
    auto meshSize = layout.meshSize();
    double dx     = meshSize[static_cast<int>(PHARE::core::Direction::X)];
    double dy     = meshSize[static_cast<int>(PHARE::core::Direction::Y)];
    double dz     = meshSize[static_cast<int>(PHARE::core::Direction::Z)];

    auto psiX = layout.physicalStartIndex(cent[0], PHARE::core::Direction::X);
    auto peiX = layout.physicalEndIndex(cent[0], PHARE::core::Direction::X);
    auto psiY = layout.physicalStartIndex(cent[1], PHARE::core::Direction::Y);
    auto peiY = layout.physicalEndIndex(cent[1], PHARE::core::Direction::Y);
    auto psiZ = layout.physicalStartIndex(cent[2], PHARE::core::Direction::Z);
    auto peiZ = layout.physicalEndIndex(cent[2], PHARE::core::Direction::Z);

    double err           = 0.0;
    std::size_t count    = 0;
    constexpr int margin = 0;

    for (auto i = psiX + margin; i <= peiX - margin; ++i)
        for (auto j = psiY + margin; j <= peiY - margin; ++j)
            for (auto kk = psiZ + margin; kk <= peiZ - margin; ++kk)
            {
                auto c = layout.fieldNodeCoordinates(
                    field, layout.localToAMR(PHARE::core::Point{i, j, kk}.as_signed()));

                double exact_avg = 0.0;

                if constexpr (dir == PHARE::core::Direction::X)
                {
                    for (double sy : {-gl_pt, +gl_pt})
                        for (double sz : {-gl_pt, +gl_pt})
                            exact_avg += ww * exactFlux(c[0], c[1] + sy * dy, c[2] + sz * dz);
                }
                else if constexpr (dir == PHARE::core::Direction::Y)
                {
                    for (double sx : {-gl_pt, +gl_pt})
                        for (double sz : {-gl_pt, +gl_pt})
                            exact_avg += ww * exactFlux(c[0] + sx * dx, c[1], c[2] + sz * dz);
                }
                else if constexpr (dir == PHARE::core::Direction::Z)
                {
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
 * @brief L2 error of an edge-averaged field (E) against the exact edge average (2-point
 * Gauss-Legendre along the edge).
 */
template<typename Layout, PHARE::core::Direction dir, typename EdgeFunc>
inline double l2EdgeAveragedError(Layout const& layout, auto const& field, EdgeFunc&& exactEdge)
{
    static constexpr double gl_pt = 0.28867513459481287; // 1 / (2*sqrt(3))
    static constexpr double w     = 0.5;

    auto cent     = layout.centering(field.physicalQuantity());
    auto meshSize = layout.meshSize();
    double dx     = meshSize[static_cast<int>(PHARE::core::Direction::X)];
    double dy     = meshSize[static_cast<int>(PHARE::core::Direction::Y)];
    double dz     = meshSize[static_cast<int>(PHARE::core::Direction::Z)];

    auto psiX = layout.physicalStartIndex(cent[0], PHARE::core::Direction::X);
    auto peiX = layout.physicalEndIndex(cent[0], PHARE::core::Direction::X);
    auto psiY = layout.physicalStartIndex(cent[1], PHARE::core::Direction::Y);
    auto peiY = layout.physicalEndIndex(cent[1], PHARE::core::Direction::Y);
    auto psiZ = layout.physicalStartIndex(cent[2], PHARE::core::Direction::Z);
    auto peiZ = layout.physicalEndIndex(cent[2], PHARE::core::Direction::Z);

    double err           = 0.0;
    std::size_t count    = 0;
    constexpr int margin = 0;

    for (auto i = psiX + margin; i <= peiX - margin; ++i)
        for (auto j = psiY + margin; j <= peiY - margin; ++j)
            for (auto kk = psiZ + margin; kk <= peiZ - margin; ++kk)
            {
                auto c = layout.fieldNodeCoordinates(
                    field, layout.localToAMR(PHARE::core::Point{i, j, kk}.as_signed()));

                double exact_avg = 0.0;

                if constexpr (dir == PHARE::core::Direction::X)
                {
                    for (double sx : {-gl_pt, +gl_pt})
                        exact_avg += w * exactEdge(c[0] + sx * dx, c[1], c[2]);
                }
                else if constexpr (dir == PHARE::core::Direction::Y)
                {
                    for (double sy : {-gl_pt, +gl_pt})
                        exact_avg += w * exactEdge(c[0], c[1] + sy * dy, c[2]);
                }
                else if constexpr (dir == PHARE::core::Direction::Z)
                {
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
