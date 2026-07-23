#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_USING_COMPUTED_FLUX_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_USING_COMPUTED_FLUX_HPP


#include "amr/solvers/solver_mhd_field_evolvers.hpp"

namespace PHARE::solver
{
template<typename MHDModel>
class EulerUsingComputedFlux
{
    using level_t = MHDModel::level_t;
    // using Layout        = MHDModel::gridlayout_type;
    using Dispatchers_t = Dispatchers<MHDModel>;

    using FiniteVolumeEuler_t = Dispatchers_t::FiniteVolumeEuler_t;
    using Faraday_t           = Dispatchers_t::Faraday_t;

public:
    EulerUsingComputedFlux() {}

    // we provide dt here because we sometimes need it to be different from newTime-currentTime, for
    // example in the case of some rk integration methods
    void operator()(MHDModel& model, auto& state, auto& statenew, auto& E, auto& fluxes, auto& bc,
                    level_t& level, double const newTime, double const dt)
    {
        FiniteVolumeEuler_t{level, model}(state, statenew, fluxes, dt);
        TimeSetter{level, model, newTime}(state.rho, state.rhoV, state.Etot);

        Faraday_t{level, model}(state.B, E, statenew.B, dt);

        TimeSetter{level, model, newTime}(statenew.B);

        bc.fillMagneticGhosts(statenew.B, level, newTime);

        bc.fillMomentsGhosts(statenew, level, newTime);
    }

    // MC2011 overload: stage metadata (stageIndex, chi, dtFine) for the messenger's
    // Tier 2/3 assembly (see mhd_messenger.hpp::assembleMC2011_). Used at
    // SSPRK4_5Integrator's stage and final-blend call sites. No k out-param: the
    // messenger back-solves the stage derivatives from the persisted stage states
    // (core::mc2011::backSolve).
    void operator()(MHDModel& model, auto& state, auto& statenew, auto& E, auto& fluxes,
                    std::size_t const stageIndex, double const chi, double const dtFine,
                    auto& bc, level_t& level, double const newTime, double const dt)
    {
        FiniteVolumeEuler_t{level, model}(state, statenew, fluxes, dt);
        TimeSetter{level, model, newTime}(state.rho, state.rhoV, state.Etot);

        Faraday_t{level, model}(state.B, E, statenew.B, dt);

        TimeSetter{level, model, newTime}(statenew.B);

        bc.fillMagneticGhosts(statenew.B, level, newTime, stageIndex, chi, dtFine);

        bc.fillMomentsGhosts(statenew, level, newTime, stageIndex, chi, dtFine);
    }
};


} // namespace PHARE::solver

#endif
