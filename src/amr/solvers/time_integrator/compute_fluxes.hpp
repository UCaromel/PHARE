#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP

#include "initializer/data_provider.hpp"
#include "amr/solvers/solver_mhd_model_view.hpp"

namespace PHARE::solver
{
template<template<typename> typename FVMethodStrategy, typename MHDModel>
class ComputeFluxes
{
    using level_t = MHDModel::level_t;
    // using Layout        = MHDModel::gridlayout_type;
    using Dispatchers_t = Dispatchers<MHDModel>;

    using Ampere_t = Dispatchers_t::Ampere_t;

    using FVMethod_t     = Dispatchers_t::template FVMethod_t<FVMethodStrategy>;
    using FVMethodInfo_t = FVMethod_t::info_type;

    constexpr static auto Hall             = FVMethod_t::Hall;
    constexpr static auto Resistivity      = FVMethod_t::Resistivity;
    constexpr static auto HyperResistivity = FVMethod_t::HyperResistivity;

    template<typename T>
    using Rec = FVMethod_t::template Rec<T>;

    using ConstrainedTransport_t
        = Dispatchers_t::template ConstrainedTransport_t<Rec, Hall, Resistivity, HyperResistivity>;
    using ConstrainedTransportInfo_t = ConstrainedTransport_t::info_type;

    using ToPrimitiveConverter_t    = Dispatchers_t::ToPrimitiveConverter_t;
    using ToConservativeConverter_t = Dispatchers_t::ToConservativeConverter_t;


public:
    ComputeFluxes(PHARE::initializer::PHAREDict const& dict)
        : fVMethodInfo_{FVMethodInfo_t::FROM(dict["fv_method"])}
        , constrainedTransportInfo_{ConstrainedTransportInfo_t::FROM(dict["constrained_transport"])}
        , to_primitive_gamma_{dict["to_primitive"]["heat_capacity_ratio"]}
        , to_conservative_gamma_{dict["to_conservative"]["heat_capacity_ratio"]}
    {
    }

    void operator()(MHDModel& model, auto& state, auto& fluxes, auto& bc, level_t& level,
                    double const newTime)
    {
        ToPrimitiveConverter_t{level, model}(state, to_primitive_gamma_, newTime);

        if constexpr (Hall || Resistivity || HyperResistivity)
            Ampere_t{level, model}(state, newTime);

        FVMethod_t{level, model, fVMethodInfo_}(ct_, state, fluxes, newTime);

        // unecessary if we decide to store both primitive and conservative variables
        ToConservativeConverter_t{level, model}(state, to_conservative_gamma_, newTime);

        ConstrainedTransport_t{level, model, constrainedTransportInfo_}(state, fluxes);
    }

    void registerResources(MHDModel& model)
    {
        ct_.registerResources(model);
        fvm_.registerResources(model);
    }

    void allocate(MHDModel& model, auto& patch, double const allocateTime) const
    {
        ct_.allocate(model, patch, allocateTime);
        fvm_.allocate(model, patch, allocateTime);
    }

private:
    FVMethodInfo_t fVMethodInfo_;
    ConstrainedTransportInfo_t constrainedTransportInfo_;

    // Ampere_t ampere_;
    typename FVMethod_t::State_t fvm_{};
    typename ConstrainedTransport_t::State_t ct_{};
    // ToPrimitiveConverter_t to_primitive_;
    // ToConservativeConverter_t to_conservative_;
    double to_primitive_gamma_;
    double to_conservative_gamma_;
};
} // namespace PHARE::solver

#endif
