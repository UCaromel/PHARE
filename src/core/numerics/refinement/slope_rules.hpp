#ifndef PHARE_CORE_NUMERICS_REFINEMENT_SLOPE_RULES_HPP
#define PHARE_CORE_NUMERICS_REFINEMENT_SLOPE_RULES_HPP

#include "core/numerics/slope_limiters/min_mod.hpp"
#include "core/numerics/slope_limiters/van_leer.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <concepts>

// Slope rules for the order-2 refinement of a dual (cell-centred) row: a coarse cell of value u0
// gets the two children u0 - s/4 and u0 + s/4, whose mean is u0 whatever the slope s.
namespace PHARE::core::refinement
{
// u holds the coarse values at offsets -width..+width along one direction (u[width] is the anchor).
// `unlimited` is the slope of the unlimited rule of the kernel's order (order 2: (u[+1] - u[-1]) /
// 2), so the same rule serves another order. `param` is the rule's runtime parameter.
template<typename Rule>
concept SlopeRule = requires(std::array<double, 2 * Rule::width + 1> const& u, double s, double p) {
    { Rule::width } -> std::convertible_to<int>;
    { Rule::defaultParam } -> std::convertible_to<double>;
    { Rule::slope(u, s, p) } -> std::same_as<double>;
};


// Suresh & Huynh's d^M4: minmod of a, b, 4a - b and 4b - a.
inline double minmod4(double a, double b) noexcept
{
    auto const sign = [](double x) { return static_cast<double>((x > 0.) - (x < 0.)); };
    double const c1 = 4. * a - b;
    double const c2 = 4. * b - a;
    double const sa = sign(a);
    if (sign(b) != sa or sign(c1) != sa or sign(c2) != sa)
        return 0.;
    return sa * std::min(std::min(std::abs(a), std::abs(b)), std::min(std::abs(c1), std::abs(c2)));
}


struct CentralRule
{
    static constexpr int width           = 1;
    static constexpr double defaultParam = 0.;

    static double slope(std::array<double, 3> const& /*u*/, double unlimited,
                        double /*param*/) noexcept
    {
        return unlimited;
    }
};


// s = 2 D- D+ / (D- + D+) if D- D+ > 0, else 0
struct VanLeerRule
{
    static constexpr int width           = 1;
    static constexpr double defaultParam = 0.;

    static double slope(std::array<double, 3> const& u, double /*unlimited*/,
                        double /*param*/) noexcept
    {
        return VanLeerLimiter::limit(u[1] - u[0], u[2] - u[1]);
    }
};


// monotonized central, k = 2: minmod(2 D-, unlimited, 2 D+)
struct MCRule
{
    static constexpr int width           = 1;
    static constexpr double defaultParam = 0.;

    static double slope(std::array<double, 3> const& u, double unlimited, double /*param*/) noexcept
    {
        return MinModLimiter::limit(2. * (u[1] - u[0]), unlimited, 2. * (u[2] - u[1]));
    }
};


// unlimited while the children stay within a fraction `param` of their parent (|s|/4 <= param
// |u0|), MCRule otherwise; the same test for every variable
struct MCGateRule
{
    static constexpr int width           = 1;
    static constexpr double defaultParam = 0.05;

    static double slope(std::array<double, 3> const& u, double unlimited, double param) noexcept
    {
        if (std::abs(unlimited) / 4. <= param * std::abs(u[1]))
            return unlimited;
        return MCRule::slope(u, unlimited, param);
    }
};


// WENO-AO(5,3)-Z weights: the unlimited slope (5-point quartic indicator) and the slopes of the
// three 3-point quadratics, epsilon at round-off level only.
struct AO5ZRule
{
    static constexpr int width           = 2;
    static constexpr double defaultParam = 0.;

    static double slope(std::array<double, 5> const& u, double unlimited, double /*param*/) noexcept
    {
        double const um2 = u[0], um1 = u[1], u0 = u[2], u1 = u[3], u2 = u[4];

        // smoothness indicator of the quartic through the five cell averages
        double const m1 = 41. * u1 / 60. - 11. * u2 / 120. - 41. * um1 / 60. + 11. * um2 / 120.;
        double const m2
            = -37. * u0 / 28. + 5. * u1 / 7. - 3. * u2 / 56. + 5. * um1 / 7. - 3. * um2 / 56.;
        double const m3 = -u1 / 6. + u2 / 12. + um1 / 6. - um2 / 12.;
        double const ib
            = (m1 + m3 / 10.) * (m1 + m3 / 10.) + 13. / 3. * (m2 * m2) + 781. / 20. * (m3 * m3);

        // slope and curvature of the left, centred and right 3-point quadratics
        std::array<std::array<double, 2>, 3> const subs{{
            {3. * u0 / 2. - 2. * um1 + um2 / 2., u0 / 2. - um1 + um2 / 2.},
            {u1 / 2. - um1 / 2., -u0 + u1 / 2. + um1 / 2.},
            {-3. * u0 / 2. + 2. * u1 - u2 / 2., u0 / 2. - u1 + u2 / 2.},
        }};
        std::array<double, 3> isub{};
        for (std::size_t i = 0; i < 3; ++i)
            isub[i] = subs[i][0] * subs[i][0] + 13. / 3. * subs[i][1] * subs[i][1];

        double const eps = 1e-28 * u0 * u0 + DBL_MIN;
        double const tau
            = (std::abs(ib - isub[0]) + std::abs(ib - isub[1]) + std::abs(ib - isub[2])) / 3.;

        constexpr double gBig = 0.85, gSub = 0.05;
        auto const zWeight = [&](double gamma, double is) {
            double const r = tau / (is + eps);
            return gamma * (1. + r * r);
        };
        double const a0 = zWeight(gBig, ib);
        std::array<double, 3> a{};
        for (std::size_t i = 0; i < 3; ++i)
            a[i] = zWeight(gSub, isub[i]);
        double const S = a0 + (a[0] + a[1] + a[2]);

        double const subSum = subs[0][0] + subs[1][0] + subs[2][0];
        double const s      = (a0 / S / gBig) * (unlimited - gSub * subSum)
                         + (a[0] / S * subs[0][0] + a[1] / S * subs[1][0] + a[2] / S * subs[2][0]);
        // A sub-stencil with a zero indicator (flat data, u0 = 0 so eps ~ DBL_MIN) overflows its
        // weight: the limit takes that sub-stencil's slope, which is 0.
        return std::isfinite(s) ? s : 0.;
    }
};


// Suresh-Huynh monotonicity-preserving bounds for a child average at +-h/4; `param` = curvature
// factor c. child+ = median(u0 + s/4, max(min A, min B), min(max A, max B)) with
//   A = hull(u0, u+1, u0 + D+/4 - c dP),  B = hull(u0, u0 + D-, u0 + D-/4 + c dM)
// (B mirrors child-'s constraint onto child+), d_j = u_{j+1} - 2 u_j + u_{j-1},
// dP = minmod4(d_0, d_1), dM = minmod4(d_-1, d_0).
struct MPRule
{
    static constexpr int width           = 2;
    static constexpr double defaultParam = 3. / 8.;

    static double slope(std::array<double, 5> const& u, double unlimited, double param) noexcept
    {
        double const um2 = u[0], um1 = u[1], u0 = u[2], u1 = u[3], u2 = u[4];
        double const L = u0 - um1, R = u1 - u0;
        double const dm1 = L - (um1 - um2);
        double const d0  = R - L;
        double const d1  = (u2 - u1) - R;
        double const dP  = minmod4(d0, d1);
        double const dM  = minmod4(dm1, d0);

        double const a2 = u0 + R / 4. - param * dP;
        double const b1 = u0 + L;
        double const b2 = u0 + L / 4. + param * dM;
        double const lo = std::max(std::min(std::min(u0, u1), a2), std::min(std::min(u0, b1), b2));
        double const hi = std::min(std::max(std::max(u0, u1), a2), std::max(std::max(u0, b1), b2));

        double v = u0 + unlimited / 4.;
        if ((lo - v) * (hi - v) > 0.)
            v += std::abs(lo - v) < std::abs(hi - v) ? lo - v : hi - v;
        return 4. * (v - u0);
    }
};


static_assert(SlopeRule<CentralRule>);
static_assert(SlopeRule<VanLeerRule>);
static_assert(SlopeRule<MCRule>);
static_assert(SlopeRule<MCGateRule>);
static_assert(SlopeRule<AO5ZRule>);
static_assert(SlopeRule<MPRule>);

} // namespace PHARE::core::refinement

#endif
