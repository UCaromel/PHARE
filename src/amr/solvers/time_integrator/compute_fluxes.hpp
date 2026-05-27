#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP

#include "amr/resources_manager/amr_utils.hpp"
#include "amr/solvers/solver_mhd_model_view.hpp"

#include "core/inner_boundary/inner_bc_context.hpp"
#include "core/inner_boundary/inner_boundary_defs.hpp"

#include "initializer/data_provider.hpp"

namespace PHARE::solver
{
template<template<typename> typename FVMethodStrategy, typename MHDModel>
class ComputeFluxes
{
    using level_t                     = MHDModel::level_t;
    using gridlayout_type             = MHDModel::gridlayout_type;
    using state_type                  = MHDModel::state_type;
    using resources_manager_type      = MHDModel::resources_manager_type;
    using inner_boundary_manager_type = MHDModel::inner_boundary_manager_type;
    using Dispatchers_t               = Dispatchers<gridlayout_type>;

    using Ampere_t = Dispatchers_t::Ampere_t;

    using FVMethod_t = Dispatchers_t::template FVMethod_t<FVMethodStrategy>;

    constexpr static auto Hall = FVMethod_t::Hall;

    template<typename T>
    using Rec = FVMethod_t::template Rec<T>;

    using ConstrainedTransport_t
        = Dispatchers_t::template ConstrainedTransport_t<MHDModel, Rec, Hall>;

    using ToPrimitiveConverter_t    = Dispatchers_t::ToPrimitiveConverter_t;
    using ToConservativeConverter_t = Dispatchers_t::ToConservativeConverter_t;


public:
    ComputeFluxes(PHARE::initializer::PHAREDict const& dict)
        : fvm_{dict["fv_method"]}
        , ct_{dict["constrained_transport"]}
        , to_primitive_{dict["to_primitive"]}
        , to_conservative_{dict["to_conservative"]}
    {
    }

    void operator()(MHDModel& model, auto& state, auto& fluxes, auto& bc, level_t& level,
                    double const newTime)
    {
        to_primitive_(level, model, newTime, state);

        if constexpr (Hall)
            ampere_(level, model, newTime, state);
        else if (fvm_.resistivity() || fvm_.hyper_resistivity())
            ampere_(level, model, newTime, state);

        fvm_(level, model, newTime, ct_.constrained_transport_, state, fluxes);

        // unecessary if we decide to store both primitive and conservative variables
        to_conservative_(level, model, newTime, state);

        // bc.fillMagneticFluxesXGhosts(fluxes.B_fx, level, newTime);
        //
        // if constexpr (MHDModel::dimension >= 2)
        // {
        //     bc.fillMagneticFluxesYGhosts(fluxes.B_fy, level, newTime);
        //
        //     if constexpr (MHDModel::dimension == 3)
        //     {
        //         bc.fillMagneticFluxesZGhosts(fluxes.B_fz, level, newTime);
        //     }
        // }
        //
        ct_(level, model, state, fluxes);

        bc.fillElectricGhosts(state.E, level, newTime);

        if (model.hasInnerBoundary())
        {
            resources_manager_type& rm       = *model.resourcesManager;
            inner_boundary_manager_type& ibm = *model.innerBoundaryManager;
            core::InnerBCContext<state_type> ctx{state, state, newTime};
            amr::visitLevel<gridlayout_type>(
                level, rm,
                [&](auto& layout, auto&&, auto&&) {
                    ibm.applyBC(state.E, layout, ctx);

                    // Pin E to 0 in inactive cells (deep inside the body) so the large CT
                    // values there neither pollute diagnostics nor leak into the cut-cell
                    // Faraday stencil that consumes E next. Done per component centering.
                    auto& meshData = ibm.getMeshData();
                    auto zeroE     = [&](auto component) {
                        auto centering = layout.centering(state.E(component).physicalQuantity());
                        auto& status   = meshData.getStatusFieldFromCentering(centering);
                        layout.evalOnBox(state.E(component), [&](auto&... args) {
                            auto idx = core::MeshIndex<gridlayout_type::dimension>{args...};
                            if (status(idx) == core::toDouble(core::ElemStatus::Inactive))
                                state.E(component)(idx) = 0.0;
                        });
                    };
                    zeroE(core::Component::X);
                    zeroE(core::Component::Y);
                    zeroE(core::Component::Z);
                },
                ibm, state);
        }
    }

    void registerResources(MHDModel& model)
    {
        ct_.constrained_transport_.registerResources(model);
        fvm_.finite_volume_method_.registerResources(model);
    }

    void allocate(MHDModel& model, auto& patch, double const allocateTime) const
    {
        ct_.constrained_transport_.allocate(model, patch, allocateTime);
        fvm_.finite_volume_method_.allocate(model, patch, allocateTime);
    }

private:
    Ampere_t ampere_;
    FVMethod_t fvm_;
    ConstrainedTransport_t ct_;
    ToPrimitiveConverter_t to_primitive_;
    ToConservativeConverter_t to_conservative_;
};
} // namespace PHARE::solver

#endif
