#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP

#include "initializer/data_provider.hpp"
#include "amr/solvers/solver_mhd_model_view.hpp"

namespace PHARE::solver
{
template<template<typename> typename FVMethodStrategy, typename MHDModel>
class ComputeFluxes
{
    using level_t       = typename MHDModel::level_t;
    using Layout        = typename MHDModel::gridlayout_type;
    using Dispatchers_t = Dispatchers<Layout>;

    using Ampere_t = Dispatchers_t::Ampere_t;

    using FVMethod_t = Dispatchers_t::template FVMethod_t<FVMethodStrategy>;

    constexpr static auto Hall             = FVMethod_t::Hall;
    constexpr static auto Resistivity      = FVMethod_t::Resistivity;
    constexpr static auto HyperResistivity = FVMethod_t::HyperResistivity;

    template<typename T>
    using Rec = FVMethod_t::template Rec<T>;

    using ConstrainedTransport_t
        = Dispatchers_t::template ConstrainedTransport_t<MHDModel, Rec, Hall, Resistivity,
                                                         HyperResistivity>;

    using ToPrimitiveConverter_t    = Dispatchers_t::ToPrimitiveConverter_t;
    using ToConservativeConverter_t = Dispatchers_t::ToConservativeConverter_t;
    using ToPointValue_t            = Dispatchers_t::template ToPointValue_t<MHDModel>;


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
        // Refresh cell-averaged primitives on a grow=2 box so the Jameson sensor in
        // build_troubled_mask_ has valid P in the ghost zone (troubled_raw_ needs grow=1).
        // Conserved ghosts (rho, rhoV, Etot, B) are valid from the previous step's ghost fill.
        to_primitive_(level, model, newTime, state, 2u);

        point_value_.build_mask(level, model, newTime, state);

        // Fill is_troubled ghosts before face/edge conversions and Godunov ghost-box reconstruction
        // read the mask. Follows PLUTO's ParallelExchangeFlag pattern: compute mask → fill ghosts
        // → use. Ghost access in face_is_troubled_/edge_is_troubled_ is valid after this point.
        bc.fillTroubledGhosts(point_value_.to_point_value_.is_troubled, level, newTime);

        point_value_(level, model, newTime, state);

        // need the point value magnetic ghosts for UCT and primitive projection of B
        bc.fillMagneticPointGhosts(point_value_.to_point_value_.B, level, newTime);

        to_primitive_(level, model, newTime, point_value_.to_point_value_);

        bc.fillPrimitivePointGhosts(point_value_.to_point_value_, level, newTime);

        if constexpr (Hall || Resistivity || HyperResistivity)
        {
            // also use point values for J.
            ampere_(level, model, newTime, point_value_.to_point_value_);

            bc.fillCurrentPointGhosts(point_value_.to_point_value_.J, level, newTime);
        }

        fvm_(level, model, newTime, ct_.constrained_transport_, point_value_.to_point_value_,
             fluxes);

        // unecessary if we decide to store both primitive and conservative variables
        // to_conservative_(level, model, newTime, state);

        ct_(level, model, point_value_.to_point_value_, state.E);

        // for laplacian, likely optimisable
        bc.fillElectricGhosts(state.E, level, newTime);

        point_value_.point_value_fluxes_to_integral(level, model, newTime, fluxes, state.E);

        // bc.fillElectricGhosts(state.E, level, newTime); -> dicrete stokes theorem only one ghost
    }

    void registerResources(MHDModel& model)
    {
        ct_.constrained_transport_.registerResources(model);
        fvm_.finite_volume_method_.registerResources(model);
        point_value_.to_point_value_.registerResources(model);
    }

    void allocate(MHDModel& model, auto& patch, double const allocateTime) const
    {
        ct_.constrained_transport_.allocate(model, patch, allocateTime);
        fvm_.finite_volume_method_.allocate(model, patch, allocateTime);
        point_value_.to_point_value_.allocate(model, patch, allocateTime);
    }

    void fillMessengerInfo(auto& info) const
    {
        point_value_.to_point_value_.fillMessengerInfo(info);
    }

private:
    Ampere_t ampere_;
    FVMethod_t fvm_;
    ConstrainedTransport_t ct_;
    ToPrimitiveConverter_t to_primitive_;
    ToConservativeConverter_t to_conservative_;
    ToPointValue_t point_value_;
};
} // namespace PHARE::solver

#endif
