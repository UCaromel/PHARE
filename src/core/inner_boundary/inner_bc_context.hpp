#ifndef PHARE_CORE_INNER_BOUNDARY_INNER_BC_CONTEXT_HPP
#define PHARE_CORE_INNER_BOUNDARY_INNER_BC_CONTEXT_HPP

namespace PHARE::core
{

/**
 * @brief Bundle of state information passed to inner boundary condition appliers.
 *
 * Naming mirrors EulerUsingComputedFlux: @p statenew is the state being actively
 * updated (primary), @p state is the previous state available for future
 * complex boundary conditions (e.g. Robin-type or zone-dependent BCs).
 *
 * When there is no distinct previous state (e.g. in ComputeFluxes), the caller
 * simply passes the same reference for both @p statenew and @p state.
 *
 * @tparam PhysicalStateT  Physical state type (MHDState, HybridState, …).
 */
template<typename PhysicalStateT>
struct InnerBCContext
{
    PhysicalStateT const& statenew; ///< Updated state (primary reference for BC interpolation).
    PhysicalStateT const& state;    ///< Previous state, may alias statenew when not applicable.
    double                time{0.}; ///< Current simulation time.
    double                dt{0.};   ///< Time-step size (zero until needed).
};

} // namespace PHARE::core

#endif // PHARE_CORE_INNER_BOUNDARY_INNER_BC_CONTEXT_HPP
