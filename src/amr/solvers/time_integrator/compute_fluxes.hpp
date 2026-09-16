#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP

#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/constrained_transport/upwind_constrained_transport_utils.hpp"

#include "initializer/data_provider.hpp"

#include "amr/solvers/solver_mhd_field_evolvers.hpp"

namespace PHARE::solver
{
template<typename FVMethodStrategy, typename MHDModel, typename PointValueApproximation>
class ComputeFluxes
{
    using level_t       = typename MHDModel::level_t;
    using Dispatchers_t = Dispatchers<MHDModel>;

    using FVMethod_t     = typename Dispatchers_t::template FVMethod_t<FVMethodStrategy>;
    using FVMethodInfo_t = typename FVMethod_t::info_type;

    constexpr static auto Hall             = FVMethod_t::Hall;
    constexpr static auto Resistivity      = FVMethod_t::Resistivity;
    constexpr static auto HyperResistivity = FVMethod_t::HyperResistivity;
    constexpr static bool ComputeCurrent   = Hall || Resistivity || HyperResistivity;

    template<typename T>
    using Rec = FVMethod_t::template Rec<T>;

    using ConstrainedTransport_t
        = typename Dispatchers_t::template ConstrainedTransport_t<Rec, Hall, Resistivity,
                                                                  HyperResistivity>;
    using ConstrainedTransportInfo_t = typename ConstrainedTransport_t::info_type;
    using VecField                   = typename MHDModel::vecfield_type;
    using Equations_t                = typename FVMethod_t::Equations_t;

public:
    explicit ComputeFluxes(PHARE::initializer::PHAREDict const& dict)
        : fVMethodInfo_{FVMethodInfo_t::FROM(dict["fv_method"])}
        , constrainedTransportInfo_{ConstrainedTransportInfo_t::FROM(dict["constrained_transport"])}
        , gamma_{dict["to_primitive"]["heat_capacity_ratio"]}
    {
    }

    void operator()(MHDModel& model, auto& state, auto& fluxes, level_t& level,
                    double const newTime)
    {
        auto& numericalState
            = approximation_.template averageStateToPointValues<ComputeCurrent>(
                model, state, level, newTime, gamma_);

        FVMethod_t{level, model, fVMethodInfo_}(fvm_, ct_, numericalState, fluxes);
        ConstrainedTransport_t{level, model, constrainedTransportInfo_}(
            ct_, numericalState, state.E);

        approximation_.pointValueFluxesToAverages(model, numericalState, fluxes, state.E, level,
                                                  newTime);
    }

    void registerResources(MHDModel& model)
    {
        model.resourcesManager->registerResources(approximation_);
        model.resourcesManager->registerResources(fvm_);
        model.resourcesManager->registerResources(ct_);
    }

    void allocate(MHDModel& model, auto& patch, double const allocateTime) const
    {
        model.resourcesManager->allocate(approximation_, patch, allocateTime);
        model.resourcesManager->allocate(fvm_, patch, allocateTime);
        model.resourcesManager->allocate(ct_, patch, allocateTime);
    }

private:
    FVMethodInfo_t fVMethodInfo_;
    ConstrainedTransportInfo_t constrainedTransportInfo_;
    PointValueApproximation approximation_{};
    core::GodunovState<VecField, Equations_t> fvm_{};
    core::UpwindConstrainedTransportState<VecField, Hall, Resistivity, HyperResistivity> ct_{};
    double gamma_;
};
} // namespace PHARE::solver

#endif
