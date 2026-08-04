#ifndef PHARE_CORE_NUMERICS_MHD_SPEEDS_HPP
#define PHARE_CORE_NUMERICS_MHD_SPEEDS_HPP

#include <cmath>

namespace PHARE::core
{
auto compute_fast_magnetosonic_(auto gamma, auto const& rho, auto const& B, auto const& BdotB,
                                auto const& P)
{
    auto const Sound     = std::sqrt((gamma * P) / rho);
    auto const AlfvenDir = std::sqrt(B * B / rho); // directionnal alfven
    auto const Alfven    = std::sqrt(BdotB / rho);

    auto const c02    = Sound * Sound;
    auto const cA2    = Alfven * Alfven;
    auto const cAdir2 = AlfvenDir * AlfvenDir;

    // Discriminant is >= 0 in exact arithmetic (cAdir2 <= cA2); clamp against roundoff-negative
    // when c02 == cA2 == cAdir2.
    return std::sqrt((c02 + cA2) * 0.5
                     + std::sqrt(std::max(0.0, (c02 + cA2) * (c02 + cA2) - 4.0 * c02 * cAdir2))
                           * 0.5);
}

auto compute_whistler_(auto const& invMeshSize, auto const& rho, auto const& BdotB)
{
    auto const vw = std::sqrt(1 + 0.25 * invMeshSize * invMeshSize) + 0.5 * invMeshSize;
    return std::sqrt(BdotB) * vw / rho;
}
} // namespace PHARE::core


#endif
