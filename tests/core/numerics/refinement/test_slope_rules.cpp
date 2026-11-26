#include "core/numerics/refinement/slope_rules.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <vector>

using namespace PHARE::core::refinement;

namespace
{
// five coarse values u[-2..+2] (anchor u[0]) and the expected slope
struct Row
{
    std::array<double, 5> u;
    double slope;
};

double unlimitedSlope(std::array<double, 5> const& u)
{
    return (u[3] - u[1]) / 2.;
}

template<SlopeRule Rule>
double slopeOf(std::array<double, 5> const& u, double param = Rule::defaultParam)
{
    std::array<double, 2 * Rule::width + 1> line{};
    std::copy_n(u.begin() + 2 - Rule::width, line.size(), line.begin());
    return Rule::slope(line, unlimitedSlope(u), param);
}

double scaleOf(std::array<double, 5> const& u)
{
    return std::abs(*std::ranges::max_element(u, {}, [](double x) { return std::abs(x); }));
}

// Expected slopes computed by the offline study scripts (eps_study, mc_study, mcd_study, mp_study);
// generator:
// refinement-weno-limiter/2026-11-26-pluggable-refinement-limiters/gates/slope_numbers.py.
// Tolerance 1e-14 relative to the data: the scripts reach the slope through other round-off paths
// (one-sided weights, children differences).
template<SlopeRule Rule>
void expectTable(std::vector<Row> const& rows, double param = Rule::defaultParam)
{
    for (auto const& row : rows)
        EXPECT_NEAR(slopeOf<Rule>(row.u, param), row.slope,
                    1e-14 * std::max(std::abs(row.slope), scaleOf(row.u)))
            << "u = {" << row.u[0] << ", " << row.u[1] << ", " << row.u[2] << ", " << row.u[3]
            << ", " << row.u[4] << "}";
}

std::array<double, 5> linear(double a, double b)
{
    return {a - 2 * b, a - b, a, a + b, a + 2 * b};
}
} // namespace


TEST(SlopeRules, LinearDataGivesTheUnlimitedSlope)
{
    for (auto const& [a, b] : std::vector<std::pair<double, double>>{
             {1., 0.25}, {-3., 0.7}, {10., -2.}, {0., 1e-3}, {1e5, 3.}})
    {
        auto const u   = linear(a, b);
        auto const s   = unlimitedSlope(u);
        auto const tol = 1e-14 * std::max(std::abs(s), scaleOf(u));
        EXPECT_NEAR(slopeOf<CentralRule>(u), s, tol);
        EXPECT_NEAR(slopeOf<VanLeerRule>(u), s, tol);
        EXPECT_NEAR(slopeOf<MCRule>(u), s, tol);
        EXPECT_NEAR(slopeOf<MCGateRule>(u), s, tol);
        EXPECT_NEAR(slopeOf<MCGateRule>(u, 0.), s, tol);
        EXPECT_NEAR(slopeOf<AO5ZRule>(u), s, tol);
        EXPECT_NEAR(slopeOf<MPRule>(u), s, tol);
    }
}


TEST(SlopeRules, VanLeerAndMCAreZeroAtAnExtremum)
{
    for (auto const& u : std::vector<std::array<double, 5>>{
             {0., 1., 2., 1.5, 0.}, {3., 2., 1., 1.2, 5.}, {0., 1., 1., 0., 0.}})
    {
        EXPECT_EQ(slopeOf<VanLeerRule>(u), 0.);
        EXPECT_EQ(slopeOf<MCRule>(u), 0.);
    }
}


TEST(SlopeRules, MCGateKeepsTheUnlimitedSlopeBelowTheDistanceThreshold)
{
    // D- = 0.1, D+ = 0.4: unlimited 0.25, MC 0.2; |unlimited|/4 = 0.0625 = d |u0| at d = 0.015625
    std::array<double, 5> const u{3.5, 3.9, 4., 4.4, 4.6};
    double const s = unlimitedSlope(u), mc = slopeOf<MCRule>(u);
    ASSERT_NEAR(mc, 0.2, 1e-14);
    EXPECT_EQ(slopeOf<MCGateRule>(u, 0.0625 / 4. * 1.001), s);
    EXPECT_EQ(slopeOf<MCGateRule>(u, 0.0625 / 4. * 0.999), mc);

    // extremum: MC gives 0, the open gate keeps the unlimited slope
    std::array<double, 5> const e{1., 1.1, 1.5, 1.4, 1.2};
    EXPECT_EQ(slopeOf<MCGateRule>(e, 0.05), unlimitedSlope(e));
    EXPECT_EQ(slopeOf<MCGateRule>(e, 0.0125), 0.);
}


TEST(SlopeRules, FivePointRulesAreUnlimitedOnAResolvedSine)
{
    // cell averages of sin over 64 cells per wavelength. ao5z deviates by O(h^p) through its Z
    // weights (offline: 1.8e-10 of max |s| at 64 cells); mp keeps the unlimited slope exactly,
    // extrema included.
    std::size_t constexpr n = 64;
    double const h          = 2. * std::numbers::pi / n;
    auto const average      = [&](long i) {
        double const x = (static_cast<double>(i) + 0.5) * h;
        return (std::cos(x - h / 2.) - std::cos(x + h / 2.)) / h;
    };
    double maxSlope = 0.;
    std::vector<std::array<double, 5>> lines;
    for (std::size_t i = 0; i < n; ++i)
    {
        std::array<double, 5> u{};
        for (long o = -2; o <= 2; ++o)
            u[static_cast<std::size_t>(o + 2)] = average(static_cast<long>(i) + o);
        maxSlope = std::max(maxSlope, std::abs(unlimitedSlope(u)));
        lines.push_back(u);
    }
    for (auto const& u : lines)
    {
        EXPECT_NEAR(slopeOf<AO5ZRule>(u), unlimitedSlope(u), 1e-9 * maxSlope);
        EXPECT_NEAR(slopeOf<MPRule>(u), unlimitedSlope(u), 1e-13 * maxSlope);
    }
}


TEST(SlopeRules, FivePointRulesLimitAStep)
{
    for (auto const& u :
         std::vector<std::array<double, 5>>{{0., 0., 0., 1., 1.}, {0., 0., 1., 1., 1.}})
    {
        EXPECT_LT(std::abs(slopeOf<AO5ZRule>(u)), 1e-6 * std::abs(unlimitedSlope(u)));
        EXPECT_EQ(slopeOf<MPRule>(u), 0.);
    }
}


TEST(SlopeRules, MinMod4)
{
    EXPECT_EQ(minmod4(1., 2.), 1.);
    EXPECT_EQ(minmod4(-1., -2.), -1.);
    EXPECT_EQ(minmod4(1., 5.), 0.); // 4a - b < 0
    EXPECT_EQ(minmod4(1., -1.), 0.);
    EXPECT_EQ(minmod4(0., 0.), 0.);
    EXPECT_EQ(minmod4(1., 0.5), 0.5);
    EXPECT_DOUBLE_EQ(minmod4(1., 0.3), 0.2); // 4b - a
}


TEST(SlopeRules, VanLeerMatchesTheOfflineScripts)
{
    expectTable<VanLeerRule>({
        {{0.29999999999999999, 1, 1.25, 2, 2.2000000000000002}, 0.375},
        {{2, 1.7, 1, 0.90000000000000002, 0.20000000000000001}, -0.17499999999999996},
        {{1, 1.1000000000000001, 1.5, 1.3999999999999999, 1.2}, 0},
        {{0.10000000000000001, 0.20000000000000001, 0.40000000000000002, 1.5, 3},
         0.33846153846153848},
        {{5, 4, 3.8999999999999999, 3.8500000000000001, 1}, -0.066666666666666527},
        {{-1, -0.5, 0.25, 0.29999999999999999, 2}, 0.093749999999999972},
        {{1, 1, 1, 2, 2}, 0},
        {{0.20000000000000001, -0.29999999999999999, 0.69999999999999996, 0.75,
          -1.1000000000000001},
         0.095238095238095316},
        {{3.5, 3.8999999999999999, 4, 4.4000000000000004, 4.5999999999999996}, 0.16000000000000014},
    });
}

TEST(SlopeRules, MCMatchesTheOfflineScripts)
{
    expectTable<MCRule>({
        {{0.29999999999999999, 1, 1.25, 2, 2.2000000000000002}, 0.5},
        {{2, 1.7, 1, 0.90000000000000002, 0.20000000000000001}, -0.19999999999999996},
        {{1, 1.1000000000000001, 1.5, 1.3999999999999999, 1.2}, 0},
        {{0.10000000000000001, 0.20000000000000001, 0.40000000000000002, 1.5, 3},
         0.40000000000000002},
        {{5, 4, 3.8999999999999999, 3.8500000000000001, 1}, -0.074999999999999956},
        {{-1, -0.5, 0.25, 0.29999999999999999, 2}, 0.099999999999999978},
        {{1, 1, 1, 2, 2}, 0},
        {{0.20000000000000001, -0.29999999999999999, 0.69999999999999996, 0.75,
          -1.1000000000000001},
         0.10000000000000009},
        {{3.5, 3.8999999999999999, 4, 4.4000000000000004, 4.5999999999999996}, 0.20000000000000018},
    });
}

TEST(SlopeRules, MCGateMatchesTheOfflineScripts)
{
    expectTable<MCGateRule>(
        {
            {{0.29999999999999999, 1, 1.25, 2, 2.2000000000000002}, 0.5},
            {{2, 1.7, 1, 0.90000000000000002, 0.20000000000000001}, -0.19999999999999996},
            {{1, 1.1000000000000001, 1.5, 1.3999999999999999, 1.2}, 0.14999999999999991},
            {{0.10000000000000001, 0.20000000000000001, 0.40000000000000002, 1.5, 3},
             0.40000000000000002},
            {{5, 4, 3.8999999999999999, 3.8500000000000001, 1}, -0.074999999999999956},
            {{-1, -0.5, 0.25, 0.29999999999999999, 2}, 0.099999999999999978},
            {{1, 1, 1, 2, 2}, 0},
            {{0.20000000000000001, -0.29999999999999999, 0.69999999999999996, 0.75,
              -1.1000000000000001},
             0.10000000000000009},
            {{3.5, 3.8999999999999999, 4, 4.4000000000000004, 4.5999999999999996},
             0.25000000000000022},
        },
        0.05);
    expectTable<MCGateRule>(
        {
            {{0.29999999999999999, 1, 1.25, 2, 2.2000000000000002}, 0.5},
            {{2, 1.7, 1, 0.90000000000000002, 0.20000000000000001}, -0.19999999999999996},
            {{1, 1.1000000000000001, 1.5, 1.3999999999999999, 1.2}, 0},
            {{0.10000000000000001, 0.20000000000000001, 0.40000000000000002, 1.5, 3},
             0.40000000000000002},
            {{5, 4, 3.8999999999999999, 3.8500000000000001, 1}, -0.074999999999999956},
            {{-1, -0.5, 0.25, 0.29999999999999999, 2}, 0.099999999999999978},
            {{1, 1, 1, 2, 2}, 0},
            {{0.20000000000000001, -0.29999999999999999, 0.69999999999999996, 0.75,
              -1.1000000000000001},
             0.10000000000000009},
            {{3.5, 3.8999999999999999, 4, 4.4000000000000004, 4.5999999999999996},
             0.20000000000000018},
        },
        0.0125);
}

TEST(SlopeRules, AO5ZMatchesTheOfflineScripts)
{
    expectTable<AO5ZRule>({
        {{0.29999999999999999, 1, 1.25, 2, 2.2000000000000002}, 0.42931255432315335},
        {{2, 1.7, 1, 0.90000000000000002, 0.20000000000000001}, -0.38930106592466296},
        {{1, 1.1000000000000001, 1.5, 1.3999999999999999, 1.2}, -0.037684797072576291},
        {{0.10000000000000001, 0.20000000000000001, 0.40000000000000002, 1.5, 3},
         0.3217450419445913},
        {{5, 4, 3.8999999999999999, 3.8500000000000001, 1}, -0.074978261940912963},
        {{-1, -0.5, 0.25, 0.29999999999999999, 2}, 0.43828485406891454},
        {{1, 1, 1, 2, 2}, 6.1312704275976737e-56},
        {{0.20000000000000001, -0.29999999999999999, 0.69999999999999996, 0.75,
          -1.1000000000000001},
         0.52456389221704369},
        {{3.5, 3.8999999999999999, 4, 4.4000000000000004, 4.5999999999999996}, 0.24011438977919439},
    });
}

TEST(SlopeRules, MPMatchesTheOfflineScripts)
{
    expectTable<MPRule>(
        {
            {{0.29999999999999999, 1, 1.25, 2, 2.2000000000000002}, 0.5},
            {{2, 1.7, 1, 0.90000000000000002, 0.20000000000000001}, -0.39999999999999991},
            {{1, 1.1000000000000001, 1.5, 1.3999999999999999, 1.2}, 0},
            {{0.10000000000000001, 0.20000000000000001, 0.40000000000000002, 1.5, 3},
             0.64999999999999991},
            {{5, 4, 3.8999999999999999, 3.8500000000000001, 1}, -0.074999999999999289},
            {{-1, -0.5, 0.25, 0.29999999999999999, 2}, 0.19999999999999996},
            {{1, 1, 1, 2, 2}, 0},
            {{0.20000000000000001, -0.29999999999999999, 0.69999999999999996, 0.75,
              -1.1000000000000001},
             0.52499999999999991},
            {{3.5, 3.8999999999999999, 4, 4.4000000000000004, 4.5999999999999996}, 0.25},
        },
        3. / 8.);
    expectTable<MPRule>(
        {
            {{0.29999999999999999, 1, 1.25, 2, 2.2000000000000002}, 0.5},
            {{2, 1.7, 1, 0.90000000000000002, 0.20000000000000001}, -0.39999999999999991},
            {{1, 1.1000000000000001, 1.5, 1.3999999999999999, 1.2}, 0},
            {{0.10000000000000001, 0.20000000000000001, 0.40000000000000002, 1.5, 3},
             0.64999999999999991},
            {{5, 4, 3.8999999999999999, 3.8500000000000001, 1}, -0.074999999999999289},
            {{-1, -0.5, 0.25, 0.29999999999999999, 2}, 0.19999999999999996},
            {{1, 1, 1, 2, 2}, 0},
            {{0.20000000000000001, -0.29999999999999999, 0.69999999999999996, 0.75,
              -1.1000000000000001},
             0.40625},
            {{3.5, 3.8999999999999999, 4, 4.4000000000000004, 4.5999999999999996}, 0.25},
        },
        3. / 32.);
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
