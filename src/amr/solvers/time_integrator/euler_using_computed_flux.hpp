#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_USING_COMPUTED_FLUX_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_EULER_USING_COMPUTED_FLUX_HPP

#include "amr/resources_manager/amr_utils.hpp"
#include "amr/solvers/solver_mhd_model_view.hpp"

#include "core/inner_boundary/inner_bc_context.hpp"
#include "core/inner_boundary/inner_boundary_defs.hpp"

namespace PHARE::solver
{
template<typename MHDModel>
class EulerUsingComputedFlux
{
    using level_t                     = MHDModel::level_t;
    using gridlayout_type             = MHDModel::gridlayout_type;
    using state_type                  = MHDModel::state_type;
    using resources_manager_type      = MHDModel::resources_manager_type;
    using inner_boundary_manager_type = MHDModel::inner_boundary_manager_type;

    using Dispatchers_t = Dispatchers<gridlayout_type>;

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

        resources_manager_type& rm = *model.resourcesManager;

        // Update external magnetic field
        amr::visitLevel<gridlayout_type>(
            level, rm,
            [&](auto& layout, auto&&, auto&&) {
                statenew.updateExternalMagneticField(layout, newTime);
            },
            statenew);

        // Pin the inner-boundary inactive region to a canonical safe physical state.
        // We cannot simply copy the previous-step values into `statenew` because some
        // integrators (e.g. TVDRK3) reuse a single buffer for both `state` and `statenew`;
        // in that aliased case the copy is a self-assignment and the drift from
        // `fv_euler_` / `faraday_` persists. Recomputing the safe state every substep is
        // both alias-safe and matches what `mhd_level_initializer` does at init.
        if (model.hasInnerBoundary())
        {
            inner_boundary_manager_type& ibm = *model.innerBoundaryManager;

            amr::visitLevel<gridlayout_type>(
                level, rm,
                [&](auto& layout, auto&&, auto&&) {
                    auto& meshData = ibm.getMeshData();

                    // Zero inactive face-centered B1 first: the cell-centered safe reset
                    // recomputes Etot1 from the projected B at the cell centre, so any
                    // stale face value would leak into the energy.
                    auto resetB = [&](auto component) {
                        auto centering
                            = layout.centering(statenew.B1(component).physicalQuantity());
                        auto& faceStatus = meshData.getStatusFieldFromCentering(centering);
                        layout.evalOnBox(statenew.B1(component), [&](auto&... args) {
                            auto idx = core::MeshIndex<gridlayout_type::dimension>{args...};
                            if (faceStatus(idx) == core::toDouble(core::ElemStatus::Inactive))
                                statenew.B1(component)(idx) = 0.0;
                        });
                    };
                    resetB(core::Component::X);
                    resetB(core::Component::Y);
                    resetB(core::Component::Z);

                    // Pin inactive cell-centered conservatives to (rho=1, P=1, V=0).
                    auto& cellStatus = meshData.cellStatusField();
                    layout.evalOnGhostBox(statenew.rho, [&](auto&... args) {
                        auto idx = core::MeshIndex<gridlayout_type::dimension>{args...};
                        if (cellStatus(idx) == core::toDouble(core::ElemStatus::Inactive))
                            statenew.safeResetInactiveCell(idx, layout, *model.thermo);
                    });
                },
                ibm, statenew);
        }

        bc.fillMagneticGhosts(statenew.B1, level, newTime);
        bc.fillMomentsGhosts(statenew, level, newTime, dt);

        // Application of inner boundary conditions
        if (model.hasInnerBoundary())
        {
            inner_boundary_manager_type& ibm = *model.innerBoundaryManager;
            core::InnerBCContext<state_type> ctx{statenew, state, newTime, dt};

            amr::visitLevel<gridlayout_type>(
                level, rm,
                [&](auto& layout, auto&&, auto&&) {
                    // Apply inner boundary conditions
                    ibm.applyBC(statenew.B1, layout, ctx);
                    ibm.applyBC(statenew.rhoV, layout, ctx);
                    ibm.applyBC(statenew.rho, layout, ctx);
                    ibm.applyBC(statenew.Etot1, layout, ctx);
                },
                ibm, state, statenew);
        }
    }

private:
    FiniteVolumeEuler_t fv_euler_;
    Faraday_t faraday_;
};
} // namespace PHARE::solver

#endif
