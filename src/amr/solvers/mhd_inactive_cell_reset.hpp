#ifndef PHARE_AMR_SOLVERS_MHD_INACTIVE_CELL_RESET_HPP
#define PHARE_AMR_SOLVERS_MHD_INACTIVE_CELL_RESET_HPP

#include "core/data/vecfield/vecfield_component.hpp"
#include "core/numerics/primite_conservative_converter/conversion_utils.hpp"
#include "core/utilities/index/index.hpp"


namespace PHARE::solver
{
/**
 * @brief Reset a single inactive/ghost MHD cell to a physically safe state.
 *
 * Inactive cells sit deep inside an embedded body and play no role in the fluid
 * solve, but their conservative values still flow through to_primitive_, mixing
 * steps, and diagnostics. We pin them to (rho=1, P=1, V=0) and recompute Etot
 * from the current face-centered B at the cell centre. B itself is kept as-is
 * to preserve div(B) = 0.
 *
 * The caller is responsible for checking cellStatus(idx) > Cut before invoking.
 */
template<typename Layout, typename State, typename Thermo>
inline void safeResetInactiveMHDCell(core::MeshIndex<Layout::dimension> const& idx, State& state,
                                     Thermo& thermo)
{
    constexpr double safeRho = 1.0;
    constexpr double safeP   = 1.0;

    state.rho(idx) = safeRho;
    state.P(idx)   = safeP;

    state.V(core::Component::X)(idx) = 0.0;
    state.V(core::Component::Y)(idx) = 0.0;
    state.V(core::Component::Z)(idx) = 0.0;

    state.rhoV(core::Component::X)(idx) = 0.0;
    state.rhoV(core::Component::Y)(idx) = 0.0;
    state.rhoV(core::Component::Z)(idx) = 0.0;

    auto const bx = Layout::project(state.B(core::Component::X), idx, Layout::faceXToCellCenter());
    auto const by = Layout::project(state.B(core::Component::Y), idx, Layout::faceYToCellCenter());
    auto const bz = Layout::project(state.B(core::Component::Z), idx, Layout::faceZToCellCenter());

    thermo.setState_DP(safeRho, safeP);
    auto const e_int = safeRho * thermo.internalEnergy();
    state.Etot(idx)  = core::totalEnergyFromInternalEnergy(e_int, safeRho, 0., 0., 0., bx, by, bz);
}

} // namespace PHARE::solver

#endif
