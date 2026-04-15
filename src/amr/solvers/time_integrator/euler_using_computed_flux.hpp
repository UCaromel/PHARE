#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_USING_COMPUTED_FLUX_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_USING_COMPUTED_FLUX_HPP

#include "core/inner_boundary/inner_bc_context.hpp"
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

    // we provide dt here because we sometimes need it to be different from newTime-currentTime, for
    // example in the case of some rk integration methods
    void operator()(MHDModel& model, auto& state, auto& statenew, auto& E, auto& fluxes, auto& bc,
                    level_t& level, double const newTime, double const dt)
    {
        fv_euler_(level, model, newTime, state, statenew, fluxes, dt);

        faraday_(level, model, state, E, statenew, dt);

        bc.fillMagneticGhosts(statenew.B, level, newTime);

        bc.fillMomentsGhosts(statenew, level, newTime);

        if (model.hasInnerBoundary())
        {
            core::InnerBCContext<std::remove_reference_t<decltype(statenew)>> ctx{statenew, state,
                                                                                   newTime, dt};
            for (auto& patch : level)
            {
                auto const layout = amr::layoutFromPatch<Layout>(*patch);
                auto _ = model.resourcesManager->setOnPatch(*patch, *model.innerBoundaryManager,
                                                             statenew, state);
                model.innerBoundaryManager->applyBC(MHDModel::physical_quantity_type::Vector::B,
                                                    statenew.B, layout, ctx);
                model.innerBoundaryManager->applyBC(MHDModel::physical_quantity_type::Vector::rhoV,
                                                    statenew.rhoV, layout, ctx);
                model.innerBoundaryManager->applyBC(MHDModel::physical_quantity_type::Scalar::rho,
                                                    statenew.rho, layout, ctx);
                model.innerBoundaryManager->applyBC(MHDModel::physical_quantity_type::Scalar::Etot,
                                                    statenew.Etot, layout, ctx);
            }
        }
    }

private:
    FiniteVolumeEuler_t fv_euler_;
    Faraday_t faraday_;
};
} // namespace PHARE::solver

#endif
