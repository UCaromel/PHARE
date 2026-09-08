#ifndef PHARE_CORE_UTILITIES_GHOST_WIDTH_CALCULATOR_HPP
#define PHARE_CORE_UTILITIES_GHOST_WIDTH_CALCULATOR_HPP

#include <cstdint>

namespace PHARE::core
{

// ============================================================================
// Utility: Round up to nearest even number
// ============================================================================

constexpr inline std::uint32_t roundUpToEven(std::uint32_t n)
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
 * - Rounded to even so the ghost box stays a whole-coarse-cell union (lower even / upper odd) at
 *   refinement ratio 2 — the invariant the ADPT magnetic touch-up's fill-box round-out clips
 *   against (see coarse_cell_round_out.hpp and
 *   ADPTMagneticRefinePatchStrategy::reconstructionRegion)
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
 * Ghost cells cover reconstruction and current stencils. Fourth-order point-value conversion
 * requires two additional layers for split Ampere and self-sufficient electric-field averaging.
 * Width stays even so ratio-2 refinement operates on whole coarse-cell unions.
 */
template<std::uint32_t reconstruction_nghosts, std::uint32_t mhd_order = 2>
constexpr std::uint32_t nbrGhostsFromReconstruction()
{
    static_assert(mhd_order == 2 || mhd_order == 4);
    constexpr std::uint32_t extra = mhd_order == 4 ? 4 : 2;
    return roundUpToEven(reconstruction_nghosts + extra);
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
