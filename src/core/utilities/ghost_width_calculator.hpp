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
 *   refinement ratio 2 — the invariant the magnetic touch-up's fill-box round-out clips
 *   against (see coarse_cell_round_out.hpp and
 *   MagneticRefinePatchStrategy::reconstructionRegion)
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
 * - Reconstruction stencil width
 * - One layer because ampere computes J on the ghost box shrinked by one, so J is only valid on
 *   ghost_width - 1 layers while the reconstruction reads it out to the stencil width
 * - At fourth order, two more layers: point-value conversion (1) and split Ampere (1). The
 *   edge averaging of the EMF needs none: it averages each component along its own edge
 *   direction only, and CT evaluates the EMF one layer further along that axis alone
 * - Rounded to even so the ghost box stays a whole-coarse-cell union (lower even / upper odd) at
 *   refinement ratio 2 — the invariant the magnetic touch-up's fill-box round-out clips
 *   against (see coarse_cell_round_out.hpp and
 *   MagneticRefinePatchStrategy::reconstructionRegion)
 *
 * There is no hyper-resistivity layer: the dissipative electric field takes its Laplacian on J at
 * its native edge location, and the flux contributions project it to the face, so they stay within
 * the reconstruction reach. See DissipativeElectricField and Godunov::transverse_on_face_.
 */
template<std::uint32_t reconstruction_nghosts, std::uint32_t mhd_order = 2>
constexpr std::uint32_t nbrGhostsFromReconstruction()
{
    static_assert(mhd_order == 2 || mhd_order == 4);
    constexpr std::uint32_t extra = mhd_order == 4 ? 3 : 1;
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
