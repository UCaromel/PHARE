// Signal speeds of the Hall Riemann solvers: the magnetic subsystem's speed is bounded by both
// the fast and the grid-scale whistler waves, |v| + max(c_f, c_w), with the whistler phase speed
// evaluated at the actual density.

#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/riemann_solvers/hll.hpp"
#include "core/numerics/riemann_solvers/mhd_speeds.hpp"
#include "core/numerics/riemann_solvers/rusanov.hpp"
#include "core/utilities/types.hpp"

#include "gtest/gtest.h"

#include <cmath>

using namespace PHARE::core;

namespace
{
double constexpr gamma_ = 5. / 3.;

PerIndex<double> state(double rho, double vx, double bx, double by, double P)
{
    return PerIndex<double>{rho, {vx, 0., 0.}, {bx, by, 0.}, P};
}

PerIndex<double> zeroFlux()
{
    return PerIndex<double>{0., {0., 0., 0.}, {0., 0., 0.}, 0.};
}

PerIndexVector<double> const noCurrent{0., 0., 0.};
} // namespace


TEST(WhistlerPhaseSpeed, isTheHallDispersionRelationAtTheActualDensity)
{
    double const B = 1.3;
    for (double const rho : {0.25, 1., 4.})
        for (double const k : {0.1, 1., 10., 100.})
        {
            double const cA   = B / std::sqrt(rho);
            double const di   = 1. / std::sqrt(rho);
            double const half = 0.5 * k * di;
            EXPECT_NEAR(compute_whistler_phase_speed_(k, rho, B * B),
                        cA * (std::sqrt(1. + half * half) + half), 1e-12 * (1. + B * k / rho));
        }
}

TEST(WhistlerPhaseSpeed, matchesTheUnitDensityFormAtUnitDensity)
{
    for (double const k : {0.1, 1., 10.})
        EXPECT_NEAR(compute_whistler_phase_speed_(k, 1., 2.), compute_whistler_(k, 1., 2.), 1e-12);
}

TEST(WhistlerPhaseSpeed, tendsToAlfvenAtLowKAndToEtaHKAtHighK)
{
    double const B = 2., rho = 4.;
    EXPECT_NEAR(compute_whistler_phase_speed_(1e-8, rho, B * B), B / std::sqrt(rho), 1e-7);
    double const k = 1e6;
    EXPECT_NEAR(compute_whistler_phase_speed_(k, rho, B * B) / (B * k / rho), 1., 1e-6);
}


TEST(RusanovHall, magneticSpeedNeverDropsBelowTheFastSpeed)
{
    // high beta, weak field: c_f >> c_w at this mesh size
    double const rho = 1., P = 100., bx = 0.01, invMesh = 1.;
    auto uL = state(rho, 0., bx, 0., P), uR = state(rho, 0., bx, 0., P);
    double const cf = compute_fast_magnetosonic_(gamma_, rho, bx, bx * bx, P);
    double const cw = compute_whistler_phase_speed_(invMesh, rho, bx * bx);
    ASSERT_GT(cf, cw);

    Rusanov<true> rusanov{gamma_, /*upwindWhistler=*/true};
    rusanov.solve<Direction::X>(uL, uR, zeroFlux(), zeroFlux(), noCurrent, noCurrent, invMesh);
    EXPECT_NEAR(rusanov.uct_coefs[2], 0.5 * cf, 1e-12);
}

TEST(RusanovHall, magneticSpeedTakesTheWhistlerWhenItDominates)
{
    double const rho = 4., P = 1e-3, bx = 1., invMesh = 50.;
    auto uL = state(rho, 0., bx, 0., P), uR = state(rho, 0., bx, 0., P);
    double const cw = compute_whistler_phase_speed_(invMesh, rho, bx * bx);
    ASSERT_GT(cw, compute_fast_magnetosonic_(gamma_, rho, bx, bx * bx, P));

    Rusanov<true> rusanov{gamma_, /*upwindWhistler=*/true};
    rusanov.solve<Direction::X>(uL, uR, zeroFlux(), zeroFlux(), noCurrent, noCurrent, invMesh);
    EXPECT_NEAR(rusanov.uct_coefs[2], 0.5 * cw, 1e-12);
}

TEST(RusanovHall, whistlerIsOffWhenHyperResistivityIsOn)
{
    double const rho = 4., P = 1e-3, bx = 1., invMesh = 50.;
    auto uL = state(rho, 0., bx, 0., P), uR = state(rho, 0., bx, 0., P);

    Rusanov<true> rusanov{gamma_, /*upwindWhistler=*/false};
    rusanov.solve<Direction::X>(uL, uR, zeroFlux(), zeroFlux(), noCurrent, noCurrent, invMesh);
    EXPECT_NEAR(rusanov.uct_coefs[2], 0.5 * compute_fast_magnetosonic_(gamma_, rho, bx, 1., P),
                1e-12);
}


TEST(HLLHall, staysFiniteWithoutMagneticFieldOrFlow)
{
    // B = 0 gives c_w = 0: the magnetic fan must still be bounded by the sound speed
    double const rho = 1., P = 1., invMesh = 10.;
    auto uL = state(rho, 0., 0., 0., P), uR = state(rho, 0., 0., 0., P);

    HLL<true> hll{gamma_, /*upwindWhistler=*/true};
    auto const F
        = hll.solve<Direction::X>(uL, uR, zeroFlux(), zeroFlux(), noCurrent, noCurrent, invMesh);

    for (auto const c : hll.uct_coefs)
        EXPECT_TRUE(std::isfinite(c));
    auto const [a, b, c, d, e, f, g, h] = F.as_tuple();
    for (auto const v : {a, b, c, d, e, f, g, h})
        EXPECT_TRUE(std::isfinite(v));

    double const cs = std::sqrt(gamma_ * P / rho);
    EXPECT_NEAR(hll.uct_coefs[2], 0.5 * cs, 1e-12); // -SR SL / (SR - SL) with SR = -SL = c_s
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
