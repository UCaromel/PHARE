#ifndef PHARE_SOLVER_SOLVER_MHD_MODEL_VIEW_HPP
#define PHARE_SOLVER_SOLVER_MHD_MODEL_VIEW_HPP


#include "core/numerics/ampere/ampere.hpp"
#include "core/numerics/faraday/faraday.hpp"
#include "core/numerics/time_integrator_utils.hpp"
#include "core/numerics/finite_volume_euler/finite_volume_euler.hpp"
#include "core/numerics/constrained_transport/upwind_constrained_transport.hpp"
#include "core/numerics/primite_conservative_converter/to_primitive_converter.hpp"
#include "core/numerics/primite_conservative_converter/to_conservative_converter.hpp"

#include "amr/resources_manager/amr_utils.hpp"

#include "solver_field_evolvers.hpp"

namespace PHARE::solver
{


template<typename Model>
class ToConservativeTransformer
{
    using GridLayout = Model::gridlayout_type;
    using level_t    = Model::amr_types::level_t;
    using core_type  = core::ToConservativeConverter<GridLayout>;

public:
    explicit ToConservativeTransformer(level_t& level, auto& model)
        : level{level}
        , model{model}
    {
    }

    void operator()(auto& state, double const gamma, double const newTime)
    {
        TimeSetter setTime{level, model, newTime};

        auto& rm = *model.resourcesManager;
        for (auto& patch : rm.enumerate(level, state))
        {
            auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
            core_type{layout, gamma}(state.rho, state.V, state.B, state.P, state.rhoV, state.Etot);
        }

        setTime(state.rho, state.V, state.P, state.rhoV, state.Etot);
    }

    level_t& level;
    Model& model;
};
template<typename Model>
ToConservativeTransformer(typename Model::amr_types::level_t&, Model&)
    -> ToConservativeTransformer<Model>;



template<typename Model>
class ToPrimitiveTransformer
{
    using GridLayout = Model::gridlayout_type;
    using level_t    = Model::amr_types::level_t;
    using core_type  = core::ToPrimitiveConverter<GridLayout>;

public:
    explicit ToPrimitiveTransformer(level_t& level, auto& model)
        : level{level}
        , model{model}
    {
    }

    void operator()(auto& state, double const gamma, double const newTime)
    {
        TimeSetter setTime{level, model, newTime};

        auto& rm = *model.resourcesManager;
        for (auto& patch : rm.enumerate(level, state))
        {
            auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
            core_type{layout}(gamma, state.rho, state.rhoV, state.B, state.Etot, state.V, state.P);
        }

        setTime(state.rho, state.rhoV, state.Etot, state.V, state.P);
    }

    level_t& level;
    Model& model;
};
template<typename Model>
ToPrimitiveTransformer(typename Model::amr_types::level_t&, Model&)
    -> ToPrimitiveTransformer<Model>;



template<typename Model>
class AmpereMHDTransformer
{
    using GridLayout = Model::gridlayout_type;
    using level_t    = Model::amr_types::level_t;
    using core_type  = core::Ampere<GridLayout>;

public:
    explicit AmpereMHDTransformer(level_t& level, auto& model)
        : level{level}
        , model{model}
    {
    }


    void operator()(auto& state, double const newTime)
    {
        TimeSetter setTime{level, model, newTime};

        auto& rm = *model.resourcesManager;
        for (auto& patch : rm.enumerate(level, state))
        {
            auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
            core_type{layout}(state.B, state.J);
        }

        setTime(state.B, state.J);
    }

    level_t& level;
    Model& model;
};
template<typename Model>
AmpereMHDTransformer(typename Model::amr_types::level_t&, Model&) -> AmpereMHDTransformer<Model>;



template<typename Model, template<typename> typename FVMethod>
class FVMethodTransformer
{
    using GridLayout = Model::gridlayout_type;
    using level_t    = Model::amr_types::level_t;
    using core_type  = FVMethod<GridLayout>;

public:
    using info_type = core_type::Info_t;
    using State_t   = core_type::State_t;

    template<typename T>
    using Rec = core_type::template Rec<T>;

    constexpr static auto Hall             = core_type::Hall;
    constexpr static auto Resistivity      = core_type::Resistivity;
    constexpr static auto HyperResistivity = core_type::HyperResistivity;

    explicit FVMethodTransformer(level_t& level, auto& model, info_type const& info)
        : level{level}
        , model{model}
        , info{info}
    {
    }


    void operator()(auto& ct, auto& state, auto& fluxes, double const newTime)
    {
        TimeSetter setTime{level, model, newTime};

        for (auto const& patch : level)
        {
            auto const layout = amr::layoutFromPatch<GridLayout>(*patch);

            core_type finite_volume_method{info, layout};
            auto _sp = model.resourcesManager->setOnPatch( //
                *patch, finite_volume_method, ct, state, fluxes);

            finite_volume_method(ct, state, fluxes);
        }

        setTime(state.rho, state.V, state.P, state.J);
    }

    level_t& level;
    Model& model;
    info_type const& info;
};



template<typename Model>
class FiniteVolumeEulerTransformer
{
    using GridLayout = Model::gridlayout_type;
    using level_t    = Model::amr_types::level_t;
    using core_type  = core::FiniteVolumeEuler<GridLayout>;

public:
    explicit FiniteVolumeEulerTransformer(level_t& level, auto& model)
        : level{level}
        , model{model}
    {
    }

    void operator()(double const newTime, Model::state_type& state, Model::state_type& statenew,
                    auto& fluxes, double const dt)
    {
        TimeSetter setTime{level, model, newTime};

        auto& rm = *model.resourcesManager;
        for (auto& patch : rm.enumerate(level, state, statenew, fluxes))
        {
            auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
            core_type{layout}(state, statenew, fluxes, dt);
        }

        setTime(state.rho, state.rhoV, state.Etot);
    }


    level_t& level;
    Model& model;
};
template<typename Model>
FiniteVolumeEulerTransformer(typename Model::amr_types::level_t&, Model&)
    -> FiniteVolumeEulerTransformer<Model>;




template<typename GridLayout, typename Model, template<typename> typename Reconstruction, auto Hall,
         auto Resistivity, auto HyperResistivity>
class ConstrainedTransportTransformer
{
    using level_t   = Model::amr_types::level_t;
    using core_type = core::UpwindConstrainedTransport<GridLayout, Model, Reconstruction, Hall,
                                                       Resistivity, HyperResistivity>;

public:
    using info_type = core_type::Info_t;
    using State_t   = core_type::State_t;

    explicit ConstrainedTransportTransformer(level_t& level, auto& model, info_type const& info)
        : level{level}
        , model{model}
        , info{info}
    {
    }

    void operator()(auto& state, auto& fluxes)
    {
        for (auto const& patch : level)
        {
            auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
            core_type constrained_transport_{info, layout};
            auto _sp = model.resourcesManager->setOnPatch(*patch, constrained_transport_, state);

            constrained_transport_(state);
        }
    }


    level_t& level;
    Model& model;
    info_type const info;
};




template<typename Model>
class FaradayMHDTransformer
{
    using GridLayout = Model::gridlayout_type;
    using level_t    = Model::amr_types::level_t;
    using core_type  = core::Faraday<GridLayout>;

public:
    explicit FaradayMHDTransformer(level_t& level, auto& model)
        : level{level}
        , model{model}
    {
    }



    void operator()(auto& state, auto& E, auto& statenew, double dt)
    {
        auto& rm = *model.resourcesManager;
        for (auto& patch : rm.enumerate(level, E, state, statenew))
        {
            auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
            core_type{layout}(state.B, E, statenew.B, dt);
        }
    }

    level_t& level;
    Model& model;
};


template<typename Model>
FaradayMHDTransformer(typename Model::amr_types::level_t&, Model&) -> FaradayMHDTransformer<Model>;




template<typename Model>
class RKUtilsTransformer
{
    using GridLayout = Model::gridlayout_type;
    using level_t    = Model::amr_types::level_t;
    using core_type  = core::RKUtils<GridLayout>;

public:
    void operator()(double const newTime, Model::state_type& res, auto... pairs)
    {
        TimeSetter setTime{level, model, newTime};

        auto& rm = *model.resourcesManager;
        for (auto& patch : rm.enumerate(level, res, pairs.state...))
        {
            auto const layout = amr::layoutFromPatch<GridLayout>(*patch);
            core_type{layout}(res, pairs...);
        }

        setTime(res.rho, res.rhoV, res.Etot);
    }


    level_t& level;
    Model& model;
};


template<typename Model>
struct Dispatchers
{
    using GridLayout = Model::gridlayout_type;

    using ToPrimitiveConverter_t    = ToPrimitiveTransformer<Model>;
    using ToConservativeConverter_t = ToConservativeTransformer<Model>;

    using Ampere_t = AmpereMHDTransformer<Model>;

    template<template<typename> typename FVMethodStrategy>
    using FVMethod_t = FVMethodTransformer<Model, FVMethodStrategy>;

    using FiniteVolumeEuler_t = FiniteVolumeEulerTransformer<Model>;

    template<template<typename> typename Reconstruction, auto Hall, auto Resistivity,
             auto HyperResistivity>
    using ConstrainedTransport_t
        = ConstrainedTransportTransformer<GridLayout, Model, Reconstruction, Hall, Resistivity,
                                          HyperResistivity>;

    using Faraday_t = FaradayMHDTransformer<Model>;
    using RKUtils_t = RKUtilsTransformer<Model>;
};



}; // namespace PHARE::solver

#endif // PHARE_SOLVER_SOLVER_MHD_MODEL_VIEW_HPP
