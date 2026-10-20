#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_USING_COMPUTED_FLUX_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_USING_COMPUTED_FLUX_HPP


#include "amr/solvers/solver_mhd_field_evolvers.hpp"
#include "core/numerics/rk_stage_context.hpp"
#include "amr/debug/shared_mismatch.hpp"

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
                    level_t& level, double const newTime, core::RKStageContext const& context,
                    double const dt)
    {
        FiniteVolumeEuler_t{level, model}(state, statenew, fluxes, dt);
        TimeSetter{level, model, newTime}(statenew.rho, statenew.rhoV, statenew.Etot);
        using GL_ = typename MHDModel::gridlayout_type;
        auto& rm_ = *model.resourcesManager;
        amr::debug::mm_vec<GL_>("EU:E_in", rm_, E, level, newTime);
        Faraday_t{level, model}(state.B, E, statenew.B, dt);
        TimeSetter{level, model, newTime}(statenew.B);
        amr::debug::mm_vec<GL_>("EU:faraday", rm_, statenew.B, level, newTime);
        amr::debug::mm_field<GL_>("EU:fve", rm_, statenew.rho, level, newTime);
        amr::debug::mm_field<GL_>("EU:fve", rm_, statenew.Etot, level, newTime);
        bc.fillConservativeGhosts(statenew, level, newTime, context);
    }
};


} // namespace PHARE::solver

#endif
