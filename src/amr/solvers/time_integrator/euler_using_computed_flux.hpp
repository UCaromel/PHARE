#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_USING_COMPUTED_FLUX_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_USING_COMPUTED_FLUX_HPP

#include "initializer/data_provider.hpp"
#include "amr/solvers/solver_mhd_model_view.hpp"

namespace PHARE::solver
{
template<typename MHDModel>
class EulerUsingComputedFlux
{
    using level_t       = typename MHDModel::level_t;
    using Layout        = typename MHDModel::gridlayout_type;
    using Dispatchers_t = Dispatchers<Layout>;

    using FiniteVolumeEuler_t = Dispatchers_t::FiniteVolumeEuler_t;
    using Faraday_t           = Dispatchers_t::Faraday_t;

public:
    EulerUsingComputedFlux() {}

    void operator()(MHDModel& model, auto& state, auto& statenew, auto& E, auto& fluxes, auto& bc,
                    level_t& level, double const currentTime, double const newTime)
    {
        double const dt = newTime - currentTime;

        fv_euler_(level, model, newTime, state, statenew, fluxes, dt);

        faraday_(level, model, state, E, statenew, dt);

        bc.fillMomentsGhosts(statenew, level, newTime);
    }

private:
    FiniteVolumeEuler_t fv_euler_;
    Faraday_t faraday_;
};
} // namespace PHARE::solver

#endif
