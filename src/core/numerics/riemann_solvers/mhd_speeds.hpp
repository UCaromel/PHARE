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

    return std::sqrt((c02 + cA2) * 0.5
                     + std::sqrt((c02 + cA2) * (c02 + cA2) - 4.0 * c02 * cAdir2) * 0.5);
}

auto compute_whistler_(auto const& invMeshSize, auto const& rho, auto const& BdotB)
{
    auto const vw = std::sqrt(1 + 0.25 * invMeshSize * invMeshSize) + 0.5 * invMeshSize;
    return std::sqrt(BdotB) * vw / rho;
}

// Whistler phase speed at wavenumber k, with d_i = 1/sqrt(rho) in code units:
//   c_w = eta_H k / 2 + sqrt(c_A^2 + (eta_H k / 2)^2),  eta_H = |B| / rho,  c_A^2 = |B|^2 / rho.
// It tends to c_A at low k and to eta_H k at high k. compute_whistler_ above hard-codes d_i = 1,
// which is exact only at rho = 1; the MHD and hybrid time-step estimates still use it.
auto compute_whistler_phase_speed_(auto const& k, auto const& rho, auto const& BdotB)
{
    auto const halfEtaHk = 0.5 * k * std::sqrt(BdotB) / rho;
    return halfEtaHk + std::sqrt(BdotB / rho + halfEtaHk * halfEtaHk);
}
} // namespace PHARE::core


#endif
