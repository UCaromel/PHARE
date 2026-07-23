#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_HPP

#include "initializer/data_provider.hpp"
#include "amr/solvers/time_integrator/compute_fluxes.hpp"
#include "amr/solvers/time_integrator/euler_using_computed_flux.hpp"

namespace PHARE::solver
{
template<typename FVMethodStrategy, typename MHDModel>
class Euler
{
    using level_t                  = MHDModel::level_t;
    using ComputeFluxes_t          = ComputeFluxes<FVMethodStrategy, MHDModel>;
    using EulerUsingComputedFlux_t = EulerUsingComputedFlux<MHDModel>;

public:
    Euler(PHARE::initializer::PHAREDict const& dict)
        : compute_fluxes_{dict}
    {
    }

    void operator()(MHDModel& model, auto& state, auto& statenew, auto& fluxes, auto& bc,
                    level_t& level, double const currentTime, double const newTime,
                    double dt = std::nan(""))
    {
        if (std::isnan(dt))
            dt = newTime - currentTime;

        compute_fluxes_(model, state, fluxes, bc, level, newTime);

        euler_using_computed_flux_(model, state, statenew, state.E, fluxes, bc, level, newTime, dt);
    }

    // MC2011 overload: stage metadata (stageIndex, chi, dtFine) threaded straight to
    // the matching euler_using_computed_flux_ overload. Used only at
    // SSPRK4_5Integrator's stage-1 call site (the only stage that goes through this
    // Euler wrapper rather than EulerUsingComputedFlux directly).
    void operator()(MHDModel& model, auto& state, auto& statenew, auto& fluxes,
                    std::size_t const stageIndex, double const chi, double const dtFine,
                    auto& bc, level_t& level, double const currentTime, double const newTime,
                    double dt = std::nan(""))
    {
        if (std::isnan(dt))
            dt = newTime - currentTime;

        compute_fluxes_(model, state, fluxes, bc, level, newTime);

        euler_using_computed_flux_(model, state, statenew, state.E, fluxes, stageIndex, chi,
                                   dtFine, bc, level, newTime, dt);
    }

    void registerResources(MHDModel& model) { compute_fluxes_.registerResources(model); }

    void allocate(MHDModel& model, auto& patch, double const allocateTime) const
    {
        compute_fluxes_.allocate(model, patch, allocateTime);
    }

    void fillMessengerInfo(auto& info) const { compute_fluxes_.fillMessengerInfo(info); }

private:
    ComputeFluxes_t compute_fluxes_;
    EulerUsingComputedFlux_t euler_using_computed_flux_;
};
} // namespace PHARE::solver

#endif
