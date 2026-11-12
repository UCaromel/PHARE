#ifndef PHARE_CORE_NUMERICS_WENO3Z_FALLBACK_HPP
#define PHARE_CORE_NUMERICS_WENO3Z_FALLBACK_HPP
// EXPERIMENT PHARE_O4_RECON_FALLBACK=1: WENO3Z reconstruction on troubled faces (port of
// exp/high-order-limit-pointvalue 752fb9321 weno3z.hpp + reconstruct_with_fallback_).
#include <cmath>
#include <tuple>
#include <utility>

namespace PHARE::core::o4fb
{
inline auto weno3zWeights(double ul, double u, double ur, double d0, double d1)
{
    constexpr double eps = 1.e-12;
    double const b0 = (u - ul) * (u - ul), b1 = (ur - u) * (ur - u), t3 = std::abs(b0 - b1);
    double const a0 = d0 * (1. + t3 / (b0 + eps)), a1 = d1 * (1. + t3 / (b1 + eps));
    return std::make_pair(a0 / (a0 + a1), a1 / (a0 + a1));
}
// face i between cells i-1 and i: L from (u_2, u_1, u), R from (u_1, u, u1)
inline double weno3zL(double ul, double u, double ur)
{
    auto [w0, w1] = weno3zWeights(ul, u, ur, 3. / 4., 1. / 4.);
    return w0 * (-0.5 * ul + 1.5 * u) + w1 * (0.5 * u + 0.5 * ur);
}
inline double weno3zR(double ul, double u, double ur)
{
    auto [w0, w1] = weno3zWeights(ul, u, ur, 1. / 4., 3. / 4.);
    return w0 * (0.5 * u + 0.5 * ul) + w1 * (-0.5 * ur + 1.5 * u);
}
} // namespace PHARE::core::o4fb
#endif
