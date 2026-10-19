#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP

#include "amr/solvers/solver_mhd_field_evolvers.hpp"

#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/constrained_transport/upwind_constrained_transport_utils.hpp"

#include "initializer/data_provider.hpp"

#include <cassert>
#include <stdexcept>

namespace PHARE::solver
{
template<typename FVMethodStrategy, typename MHDModel, typename PointValueApproximation>
class ComputeFluxes
{
    using level_t       = MHDModel::level_t;
    using Dispatchers_t = Dispatchers<MHDModel>;

    using DissipativeElectricField_t = Dispatchers_t::DissipativeElectricField_t;

    using FVMethod_t     = Dispatchers_t::template FVMethod_t<FVMethodStrategy>;
    using FVMethodInfo_t = FVMethod_t::info_type;

    constexpr static auto Hall = FVMethod_t::Hall;

    template<typename T>
    using Rec = FVMethod_t::template Rec<T>;

    using ConstrainedTransport_t     = Dispatchers_t::template ConstrainedTransport_t<Rec, Hall>;
    using ConstrainedTransportInfo_t = ConstrainedTransport_t::info_type;

    // from point values, CT and the flux averaging read the dissipative EMF beyond the cell box
    constexpr static bool pointValues = Rec<typename MHDModel::gridlayout_type>::pointValues;

    using VecField = MHDModel::vecfield_type;


public:
    explicit ComputeFluxes(PHARE::initializer::PHAREDict const& dict)
        : fVMethodInfo_{FVMethodInfo_t::FROM(dict["fv_method"])}
        , constrainedTransportInfo_{ConstrainedTransportInfo_t::FROM(dict["constrained_transport"])}
        , gamma_{dict["to_primitive"]["heat_capacity_ratio"]}
        , needsCurrent_{Hall || fVMethodInfo_.isResistive() || fVMethodInfo_.isHyperResistive()}
    {
        // eta/nu are duplicated into both info structs
        assert(fVMethodInfo_.eta == constrainedTransportInfo_.eta
               && fVMethodInfo_.nu == constrainedTransportInfo_.nu);

        // hyper-resistivity is only meaningful alongside the Hall term
        if constexpr (!Hall)
            if (fVMethodInfo_.isHyperResistive())
                throw std::runtime_error(
                    "Error - hyper-resistivity (nu > 0) requires the Hall term to be enabled");
    }

    // The flux pipeline runs on the approximation's point values: the state itself at second
    // order, a separate point-value state at fourth order, whose fluxes and EMF are then turned
    // back into face and edge averages.
    void operator()(MHDModel& model, auto& state, auto& fluxes, auto& bc, level_t& level,
                    double const newTime)
    {
        auto& numericalState = approximation_.averageStateToPointValues(
            model, state, level, newTime, gamma_, needsCurrent_);

        if (isDissipative_)
            DissipativeElectricField_t{level, model, fVMethodInfo_,
                                       pointValues}(dissipative_electric_state_, numericalState);

        FVMethod_t{level, model, fVMethodInfo_}(ct_, dissipative_electric_state_, numericalState,
                                                fluxes);

        ConstrainedTransport_t{level, model, constrainedTransportInfo_}(
            ct_, dissipative_electric_state_, numericalState, state.E);

        approximation_.pointValueFluxesToAverages(model, fluxes, state.E, level, newTime);
    }

    void registerResources(MHDModel& model)
    {
        model.resourcesManager->registerResources(approximation_);
        model.resourcesManager->registerResources(ct_);
        model.resourcesManager->registerResources(dissipative_electric_state_);
    }

    void allocate(MHDModel& model, auto& patch, double const allocateTime) const
    {
        model.resourcesManager->allocate(approximation_, patch, allocateTime);
        model.resourcesManager->allocate(ct_, patch, allocateTime);
        model.resourcesManager->allocate(dissipative_electric_state_, patch, allocateTime);
    }

private:
    FVMethodInfo_t fVMethodInfo_;
    ConstrainedTransportInfo_t constrainedTransportInfo_;
    bool const isDissipative_{fVMethodInfo_.isResistive() || fVMethodInfo_.isHyperResistive()};

    PointValueApproximation approximation_{};
    core::UpwindConstrainedTransportState<VecField> ct_{Hall};
    core::DissipativeElectricFieldState<VecField> dissipative_electric_state_{isDissipative_};
    double gamma_;
    bool needsCurrent_;
};
} // namespace PHARE::solver

#endif
