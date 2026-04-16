#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_USING_COMPUTED_FLUX_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_USING_COMPUTED_FLUX_HPP

#include "core/inner_boundary/inner_bc_context.hpp"
#include "core/inner_boundary/inner_boundary_mesh_data.hpp"
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

        if (model.hasInnerBoundary())
        {
            for (auto& patch : level)
            {
                auto const layout = amr::layoutFromPatch<Layout>(*patch);
                auto _guard = model.resourcesManager->setOnPatch(
                    *patch, *model.innerBoundaryManager, statenew, state);

                auto& meshData   = model.innerBoundaryManager->getMeshData();
                auto& cellStatus = meshData.cellStatusField();

                // Restore cell-centered quantities for inactive/ghost cells
                layout.evalOnBox(statenew.rho, [&](auto&... args) {
                    auto idx = core::MeshIndex<Layout::dimension>{args...};
                    if (cellStatus(idx) > core::toDouble(core::ElemStatus::Cut))
                    {
                        statenew.rho(idx)                      = state.rho(idx);
                        statenew.Etot(idx)                     = state.Etot(idx);
                        statenew.rhoV(core::Component::X)(idx) = state.rhoV(core::Component::X)(idx);
                        statenew.rhoV(core::Component::Y)(idx) = state.rhoV(core::Component::Y)(idx);
                        statenew.rhoV(core::Component::Z)(idx) = state.rhoV(core::Component::Z)(idx);
                    }
                });

                // Restore face-centered B for inactive/ghost face elements
                auto restoreB = [&](auto component) {
                    auto centering   = layout.centering(statenew.B(component).physicalQuantity());
                    auto& faceStatus = meshData.getStatusFieldFromCentering(centering);
                    layout.evalOnBox(statenew.B(component), [&](auto&... args) {
                        auto idx = core::MeshIndex<Layout::dimension>{args...};
                        if (faceStatus(idx) > core::toDouble(core::ElemStatus::Cut))
                            statenew.B(component)(idx) = state.B(component)(idx);
                    });
                };
                restoreB(core::Component::X);
                restoreB(core::Component::Y);
                restoreB(core::Component::Z);
            }
        }

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
