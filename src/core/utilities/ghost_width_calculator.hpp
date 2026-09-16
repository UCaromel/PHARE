#ifndef PHARE_CORE_UTILITIES_GHOST_WIDTH_CALCULATOR_HPP
#define PHARE_CORE_UTILITIES_GHOST_WIDTH_CALCULATOR_HPP

#include <cstdint>

namespace PHARE::core
{

// ============================================================================
// Utility: Round up to nearest even number
// ============================================================================

constexpr std::uint32_t roundUpToEven(std::uint32_t n)
{
    return (n % 2 == 0) ? n : n + 1;
}


// ============================================================================
// Ghost Width Computation Functions
// ============================================================================

/**
 * @brief Compute ghost width for Hybrid PIC model based on interpolation order.
 *
 * Ghost cells are needed for:
 * - Particle-mesh interpolation: (interp_order + 1) / 2
 * - One extra layer for particles that may leave cells
 * - Rounded to even for Toth & Roe (2002) magnetic refinement formulas
 */
template<std::uint32_t interp_order>
constexpr std::uint32_t nbrGhostsFromInterpOrder()
{
    if constexpr (interp_order == 1)
        return 2;
    else if constexpr (interp_order == 2)
        return 4;
    else if constexpr (interp_order == 3)
        return 4;
    else
        return roundUpToEven((interp_order + 1) / 2 + 1);
}


/**
 * @brief Compute ghost width for MHD model based on reconstruction stencil.
 *
 * Ghost cells are needed for:
 * - Reconstruction stencil width, which the transverse grow of the Godunov flux loop reads out
 *   to in full
 * - One layer for J, always: Ampere writes it on the ghost box shrunk by one, so the outermost
 *   ring is never valid and the reconstruction must stay one layer inside it
 * - One more layer only under hyper-resistivity: its flux loop grows by one in the flux
 *   direction so that laplacian(Jt) can be taken at face +/-1, reaching J one layer further out
 * - Rounded to even for Toth & Roe (2002) magnetic refinement formulas
 *
 * The hyper-resistivity layer is gated on the same flag that gates the grow shell itself
 * (getGrow<direction, dimension, HyperResistivity> in godunov_fluxes.hpp), so a build without
 * hyper-resistivity does not pay for it. At WENOZ this is the difference between 6 and 4.
 */
template<std::uint32_t reconstruction_nghosts, bool hyperResistivity>
constexpr std::uint32_t nbrGhostsFromReconstruction()
{
    return roundUpToEven(reconstruction_nghosts + 1 + (hyperResistivity ? 1 : 0));
}


/**
 * @brief For particles, ghost width depends on interpolation order.
 *
 * This is the same as the Hybrid field ghost width.
 */
template<std::uint32_t interp_order>
constexpr std::uint32_t particleGhostWidth()
{
    return nbrGhostsFromInterpOrder<interp_order>();
}


} // namespace PHARE::core

#endif // PHARE_CORE_UTILITIES_GHOST_WIDTH_CALCULATOR_HPP
