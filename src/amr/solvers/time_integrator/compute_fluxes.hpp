#ifndef PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP
#define PHARE_CORE_NUMERICS_TIME_INTEGRATOR_COMPUTE_FLUXES_HPP

#include "initializer/data_provider.hpp"
#include "core/numerics/godunov_fluxes/godunov_utils.hpp"
#include "core/numerics/constrained_transport/upwind_constrained_transport_utils.hpp"
#include "core/numerics/point_values_handler/point_value_handler_utils.hpp"
#include "amr/solvers/solver_mhd_field_evolvers.hpp"

namespace PHARE::solver
{
template<typename FVMethodStrategy, typename MHDModel>
class ComputeFluxes
{
    using level_t       = MHDModel::level_t;
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

    using ToPrimitiveConverter_t = Dispatchers_t::ToPrimitiveConverter_t;
    using ToPointValue_t         = Dispatchers_t::ToPointValue_t;

    using VecField    = MHDModel::vecfield_type;
    using Equations_t = FVMethod_t::Equations_t;


public:
    ComputeFluxes(PHARE::initializer::PHAREDict const& dict)
        : fVMethodInfo_{FVMethodInfo_t::FROM(dict["fv_method"])}
        , constrainedTransportInfo_{ConstrainedTransportInfo_t::FROM(dict["constrained_transport"])}
        , to_primitive_gamma_{dict["to_primitive"]["heat_capacity_ratio"]}
    {
    }

    // Fluxes are computed in point-value space: the model state is converted to point values,
    // the finite-volume method produces point-value fluxes, constrained transport produces the
    // point-value electric field (written into the model state.E, treated as electric flux), and
    // finally fluxes and state.E are converted back to integral (quadrature-averaged) form.
    void operator()(MHDModel& model, auto& state, auto& fluxes, auto& bc, level_t& level,
                    double const newTime)
    {
        ToPointValue_t{level, model}(point_value_state_, state, newTime);

        // need the point value magnetic ghosts for UCT and primitive projection of B
        bc.fillMagneticPointGhosts(point_value_state_.B, level, newTime);

        if constexpr (Hall || Resistivity || HyperResistivity)
        {
            // also use point values for J.
            Ampere_t{level, model}(point_value_state_.B, point_value_state_.J);
            TimeSetter{level, model, newTime}(point_value_state_.B, point_value_state_.J);

            bc.fillCurrentPointGhosts(point_value_state_.J, level, newTime);
        }

        ToPrimitiveConverter_t{level, model}(point_value_state_, to_primitive_gamma_, newTime);

        bc.fillPrimitivePointGhosts(point_value_state_, level, newTime);

        FVMethod_t{level, model, fVMethodInfo_}(fvm_, ct_, point_value_state_, fluxes, newTime);

        // constrained transport runs in point-value world: it reads point B/J from
        // point_value_state_ and writes the point-value electric field into the model state.E.
        ConstrainedTransport_t{level, model, constrainedTransportInfo_}(ct_, point_value_state_,
                                                                        state.E);

        // for laplacian, likely optimisable
        bc.fillElectricGhosts(state.E, level, newTime);

        ToPointValue_t{level, model}.point_value_fluxes_to_integral(point_value_state_, fluxes,
                                                                    state.E, newTime);
    }

    void registerResources(MHDModel& model)
    {
        model.resourcesManager->registerResources(point_value_state_);
        model.resourcesManager->registerResources(fvm_);
        model.resourcesManager->registerResources(ct_);
    }

    void allocate(MHDModel& model, auto& patch, double const allocateTime) const
    {
        model.resourcesManager->allocate(point_value_state_, patch, allocateTime);
        model.resourcesManager->allocate(fvm_, patch, allocateTime);
        model.resourcesManager->allocate(ct_, patch, allocateTime);
    }

    void fillMessengerInfo(auto& info) const { point_value_state_.fillMessengerInfo(info); }

private:
    FVMethodInfo_t fVMethodInfo_;
    ConstrainedTransportInfo_t constrainedTransportInfo_;

    core::PointValueState<VecField> point_value_state_{};
    core::GodunovState<VecField, Equations_t> fvm_{};
    core::UpwindConstrainedTransportState<VecField, Hall, Resistivity> ct_{};

    double to_primitive_gamma_;
};
} // namespace PHARE::solver

#endif
