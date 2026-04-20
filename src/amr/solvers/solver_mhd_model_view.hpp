#ifndef PHARE_SOLVER_SOLVER_MHD_MODEL_VIEW_HPP
#define PHARE_SOLVER_SOLVER_MHD_MODEL_VIEW_HPP

#include "amr/physical_models/physical_model.hpp"
#include "amr/resources_manager/amr_utils.hpp"
#include "amr/solvers/solver.hpp"
#include "amr/utilities/box/amr_box.hpp"
#include "core/numerics/constrained_transport/constrained_transport.hpp"
#include "core/numerics/constrained_transport/upwind_constrained_transport.hpp"
#include "core/numerics/primite_conservative_converter/to_conservative_converter.hpp"
#include "core/numerics/primite_conservative_converter/to_primitive_converter.hpp"
#include "core/numerics/point_values_handler/point_value_handler.hpp"
#include "core/numerics/ampere/ampere.hpp"
#include "core/numerics/faraday/faraday.hpp"
#include "core/numerics/finite_volume_euler/finite_volume_euler.hpp"
#include "core/numerics/time_integrator_utils.hpp"
#include "core/utilities/box/box.hpp"
#include "core/utilities/point/point.hpp"

namespace PHARE::solver
{
template<typename MHDModel>
struct TimeSetter
{
    // MacOS clang has trouble constructing aggregates with template parameters
    TimeSetter(MHDModel& m, double t)
        : model(m)
        , newTime(t)
    {
    }

    template<typename... QuantityAccessors>
    void operator()(auto& patch, QuantityAccessors... accessors)
    {
        (model.resourcesManager->setTime(accessors(), patch, newTime), ...);
    }

    MHDModel& model;
    double newTime;
};

template<typename GridLayout>
class ToConservativeTransformer
{
    using core_type = PHARE::core::ToConservativeConverter<GridLayout>;

public:
    template<typename MHDModel>
    void operator()(MHDModel::level_t const& level, MHDModel& model, double const newTime,
                    auto& state)
    {
        TimeSetter setTime{model, newTime};

        for (auto const& patch : level)
        {
            auto layout = PHARE::amr::layoutFromPatch<GridLayout>(*patch);
            auto _sp    = model.resourcesManager->setOnPatch(*patch, state);
            auto _sl    = core::SetLayout(&layout, to_conservative_);

            setTime(
                *patch, [&]() -> auto&& { return state.rho; }, [&]() -> auto&& { return state.V; },
                [&]() -> auto&& { return state.P; }, [&]() -> auto&& { return state.rhoV; },
                [&]() -> auto&& { return state.Etot; });

            to_conservative_(state.rho, state.V, state.B, state.P, state.rhoV, state.Etot);
        }
    }

    core_type to_conservative_;
};

template<typename GridLayout>
class ToPrimitiveTransformer
{
    using core_type = PHARE::core::ToPrimitiveConverter<GridLayout>;

public:
    template<typename MHDModel>
    void operator()(MHDModel::level_t const& level, MHDModel& model, double const newTime,
                    auto& state)
    {
        TimeSetter setTime{model, newTime};

        for (auto const& patch : level)
        {
            auto layout = PHARE::amr::layoutFromPatch<GridLayout>(*patch);
            auto _sp    = model.resourcesManager->setOnPatch(*patch, state);
            auto _sl    = core::SetLayout(&layout, to_primitive_);

            setTime(
                *patch, [&]() -> auto&& { return state.rho; },
                [&]() -> auto&& { return state.rhoV; }, [&]() -> auto&& { return state.Etot; },
                [&]() -> auto&& { return state.V; }, [&]() -> auto&& { return state.P; });

            to_primitive_(state.rho, state.rhoV, state.B, state.Etot, state.V, state.P);
        }
    }

    template<typename MHDModel>
    void operator()(MHDModel::level_t const& level, MHDModel& model, double const newTime,
                    auto& state, uint32_t grow)
    {
        TimeSetter setTime{model, newTime};

        for (auto const& patch : level)
        {
            auto layout = PHARE::amr::layoutFromPatch<GridLayout>(*patch);
            auto _sp    = model.resourcesManager->setOnPatch(*patch, state);
            auto _sl    = core::SetLayout(&layout, to_primitive_);

            setTime(
                *patch, [&]() -> auto&& { return state.rho; },
                [&]() -> auto&& { return state.rhoV; }, [&]() -> auto&& { return state.Etot; },
                [&]() -> auto&& { return state.V; }, [&]() -> auto&& { return state.P; });

            to_primitive_(state.rho, state.rhoV, state.B, state.Etot, state.V, state.P, grow);
        }
    }

    core_type to_primitive_;
};

template<typename GridLayout, typename MHDModel>
class ToPointValueTransformer
{
    using core_type = PHARE::core::PointValueHandler<GridLayout, MHDModel>;

public:
    void build_mask(MHDModel::level_t const& level, MHDModel& model, double const newTime,
                    MHDModel::state_type& state)
    {
        for (auto const& patch : level)
        {
            auto layout = PHARE::amr::layoutFromPatch<GridLayout>(*patch);
            auto _sp    = model.resourcesManager->setOnPatch(*patch, to_point_value_, state);
            auto _sl    = core::SetLayout(&layout, to_point_value_);

            to_point_value_.build_mask(state);
        }
    }

    void operator()(MHDModel::level_t const& level, MHDModel& model, double const newTime,
                    MHDModel::state_type& state)
    {
        TimeSetter setTime{model, newTime};

        for (auto const& patch : level)
        {
            auto layout = PHARE::amr::layoutFromPatch<GridLayout>(*patch);
            auto _sp    = model.resourcesManager->setOnPatch(*patch, to_point_value_, state);
            auto _sl    = core::SetLayout(&layout, to_point_value_);

            setTime(
                *patch, [&]() -> auto&& { return to_point_value_.rho; },
                [&]() -> auto&& { return to_point_value_.V; },
                [&]() -> auto&& { return to_point_value_.B; },
                [&]() -> auto&& { return to_point_value_.P; });

            to_point_value_(state);
        }
    }

    void point_value_fluxes_to_integral(MHDModel::level_t const& level, MHDModel& model,
                                        double const newTime, auto& fluxes, auto& E)
    {
        for (auto const& patch : level)
        {
            auto layout = PHARE::amr::layoutFromPatch<GridLayout>(*patch);
            auto _sp    = model.resourcesManager->setOnPatch(*patch, to_point_value_, fluxes, E);
            auto _sl    = core::SetLayout(&layout, to_point_value_);

            to_point_value_.point_value_fluxes_to_integral(fluxes, E);
        }
    }

    core_type to_point_value_;
};

template<typename GridLayout>
class AmpereMHDTransformer
{
    using core_type = PHARE::core::Ampere<GridLayout>;

public:
    template<typename MHDModel>
    void operator()(MHDModel::level_t const& level, MHDModel& model, double const newTime,
                    auto& state)
    {
        TimeSetter setTime{model, newTime};

        for (auto const& patch : level)
        {
            auto layout = PHARE::amr::layoutFromPatch<GridLayout>(*patch);
            auto _sp    = model.resourcesManager->setOnPatch(*patch, state);
            auto _sl    = core::SetLayout(&layout, ampere_);

            setTime(
                *patch, [&]() -> auto&& { return state.B; }, [&]() -> auto&& { return state.J; });

            ampere_(state.B, state.J);
        }
    }

    core_type ampere_;
};

template<typename GridLayout, template<typename> typename FVMethod>
class FVMethodTransformer
{
    using core_type = FVMethod<GridLayout>;

public:
    template<typename T>
    using Rec = core_type::template Rec<T>;

    constexpr static auto Hall             = core_type::Hall;
    constexpr static auto Resistivity      = core_type::Resistivity;
    constexpr static auto HyperResistivity = core_type::HyperResistivity;

    template<typename MHDModel>
    void operator()(MHDModel::level_t const& level, MHDModel& model, double const newTime, auto& ct,
                    auto& state, auto& fluxes)
    {
        TimeSetter setTime{model, newTime};

        for (auto const& patch : level)
        {
            auto layout = PHARE::amr::layoutFromPatch<GridLayout>(*patch);
            auto _sp = model.resourcesManager->setOnPatch(*patch, finite_volume_method_, ct, state,
                                                          fluxes);
            auto _sl = core::SetLayout(&layout, finite_volume_method_, ct);

            setTime(
                *patch, [&]() -> auto&& { return state.rho; }, [&]() -> auto&& { return state.V; },
                [&]() -> auto&& { return state.P; }, [&]() -> auto&& { return state.J; });

            finite_volume_method_(ct, state, fluxes);
        }
    }

    core_type finite_volume_method_;
};


template<typename GridLayout>
class FiniteVolumeEulerTransformer
{
    using core_type = PHARE::core::FiniteVolumeEuler<GridLayout>;

public:
    template<typename MHDModel>
    void operator()(MHDModel::level_t const& level, MHDModel& model, double const newTime,
                    MHDModel::state_type& state, MHDModel::state_type& statenew, auto& fluxes,
                    double const dt)
    {
        TimeSetter setTime{model, newTime};

        for (auto const& patch : level)
        {
            auto layout = PHARE::amr::layoutFromPatch<GridLayout>(*patch);
            auto _sp    = model.resourcesManager->setOnPatch(*patch, state, statenew, fluxes);
            auto _sl    = core::SetLayout(&layout, euler_);

            setTime(
                *patch, [&]() -> auto&& { return state.rho; },
                [&]() -> auto&& { return state.rhoV; }, [&]() -> auto&& { return state.Etot; });

            // temporary, this is just to have the time setted for the model primitive variables.
            // this is needed for the primitive ghost schedules, which are technically only applied
            // on point values, but rely on model fills for now.
            setTime(
                *patch, [&]() -> auto&& { return state.V; }, [&]() -> auto&& { return state.P; });

            euler_(state, statenew, fluxes, dt);
        }
    }

    core_type euler_;
};

template<typename GridLayout, typename MHDModel, template<typename> typename Reconstruction,
         auto Hall, auto Resistivity, auto HyperResistivity>
class ConstrainedTransportTransformer
{
    using core_type = PHARE::core::UpwindConstrainedTransport<GridLayout, MHDModel, Reconstruction,
                                                              Hall, Resistivity, HyperResistivity>;

public:
    void operator()(MHDModel::level_t const& level, MHDModel& model, auto& state, auto& E)
    {
        for (auto const& patch : level)
        {
            auto layout = PHARE::amr::layoutFromPatch<GridLayout>(*patch);
            auto _sp = model.resourcesManager->setOnPatch(*patch, constrained_transport_, state, E);
            auto _sl = core::SetLayout(&layout, constrained_transport_);
            constrained_transport_(state, E);
        }
    }

    core_type constrained_transport_;
};

template<typename GridLayout>
class FaradayMHDTransformer
{
    using core_type = PHARE::core::Faraday<GridLayout>;

public:
    template<typename MHDModel>
    void operator()(MHDModel::level_t const& level, MHDModel& model, MHDModel::state_type& state,
                    MHDModel::vecfield_type& E, MHDModel::state_type& statenew, double dt)
    {
        for (auto const& patch : level)
        {
            auto layout = PHARE::amr::layoutFromPatch<GridLayout>(*patch);
            auto _sp    = model.resourcesManager->setOnPatch(*patch, E, state, statenew);
            auto _sl    = core::SetLayout(&layout, faraday_);
            faraday_(state.B, E, statenew.B, dt);
        }
    }

    core_type faraday_;
};

template<typename GridLayout>
class RKUtilsTransformer
{
    using core_type = PHARE::core::RKUtils<GridLayout>;

public:
    template<typename MHDModel, typename... Pairs>
    void operator()(MHDModel::level_t const& level, MHDModel& model, double const newTime,
                    MHDModel::state_type& res, Pairs... pairs)
    {
        TimeSetter setTime{model, newTime};

        for (auto const& patch : level)
        {
            auto layout = PHARE::amr::layoutFromPatch<GridLayout>(*patch);
            auto _sp    = model.resourcesManager->setOnPatch(*patch, res, pairs.state...);
            auto _sl    = core::SetLayout(&layout, rkutils_);

            setTime(
                *patch, [&]() -> auto&& { return res.rho; }, [&]() -> auto&& { return res.rhoV; },
                [&]() -> auto&& { return res.Etot; });

            rkutils_(res, pairs...);
        }
    }

    core_type rkutils_;
};


template<typename GridLayout>
class Dispatchers
{
public:
    using ToPrimitiveConverter_t    = ToPrimitiveTransformer<GridLayout>;
    using ToConservativeConverter_t = ToConservativeTransformer<GridLayout>;

    template<typename MHDModel>
    using ToPointValue_t = ToPointValueTransformer<GridLayout, MHDModel>;

    using Ampere_t = AmpereMHDTransformer<GridLayout>;

    template<template<typename> typename FVMethodStrategy>
    using FVMethod_t = FVMethodTransformer<GridLayout, FVMethodStrategy>;

    using FiniteVolumeEuler_t = FiniteVolumeEulerTransformer<GridLayout>;

    template<typename MHDModel, template<typename> typename Reconstruction, auto Hall,
             auto Resistivity, auto HyperResistivity>
    using ConstrainedTransport_t
        = ConstrainedTransportTransformer<GridLayout, MHDModel, Reconstruction, Hall, Resistivity,
                                          HyperResistivity>;

    using Faraday_t = FaradayMHDTransformer<GridLayout>;
    using RKUtils_t = RKUtilsTransformer<GridLayout>;
};

// for now keep identical interface as hybrid for simplicity
template<typename MHDModel_>
class MHDModelView : public ISolverModelView
{
public:
    using MHDModel_t       = MHDModel_;
    using level_t          = typename MHDModel_t::level_t;
    using IPhysicalModel_t = MHDModel_t::Interface;

    MHDModelView(level_t& level, IPhysicalModel_t& model)
        : model_{dynamic_cast<MHDModel_&>(model)}
    {
    }

    auto& model() { return model_; }
    auto& model() const { return model_; }

    MHDModel_t& model_;
};

}; // namespace PHARE::solver

#endif // PHARE_SOLVER_SOLVER_MHD_MODEL_VIEW_HPP
