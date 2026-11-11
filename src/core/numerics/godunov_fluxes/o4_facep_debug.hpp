#ifndef PHARE_CORE_NUMERICS_O4_FACEP_DEBUG_HPP
#define PHARE_CORE_NUMERICS_O4_FACEP_DEBUG_HPP
// EXPERIMENT PHARE_O4_FACEP_DUMP=<dir>: the first non-positive reconstructed face pressure of a
// rank is recorded by Godunov; compute_fluxes then dumps that stage's arrays (diagnostic only).
#include <array>
#include <cstdlib>
#include <string>

namespace PHARE::core::o4dbg
{
struct FaceBad
{
    bool pending = false, done = false;
    int dir = -1, side = -1;               // side 0 = uL, 1 = uR
    std::array<int, 2> amr{};              // AMR face index
    double P = 0, rho = 0, PL = 0, PR = 0; // reconstructed
    std::array<double, 6> Pst{}, rhost{};  // stencil u_3..u2 of the reconstructed cell fields
};
inline FaceBad faceBad;
inline char const* dumpDir()
{
    static char const* d = std::getenv("PHARE_O4_FACEP_DUMP");
    return d;
}
} // namespace PHARE::core::o4dbg
#endif
