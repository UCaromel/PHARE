#ifndef PHARE_CORE_NUMERICS_SSPRK4_5_INTEGRATOR_HPP
#define PHARE_CORE_NUMERICS_SSPRK4_5_INTEGRATOR_HPP

#include "initializer/data_provider.hpp"
#include "amr/solvers/time_integrator/base_mhd_timestepper.hpp"
#include "amr/solvers/time_integrator/compute_fluxes.hpp"
#include "amr/solvers/time_integrator/euler_using_computed_flux.hpp"
#include "amr/solvers/solver_mhd_field_evolvers.hpp"
#include "amr/solvers/time_integrator/euler.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/solvers/time_integrator/point_value_approximation.hpp"
#include "core/numerics/time_integrator_utils.hpp"
#include "core/numerics/mc2011/mc2011_reconstruction.hpp"
#include "core/models/mhd_conserved_state.hpp"
#include "core/numerics/rk_stage_context.hpp"
#include "amr/messengers/mhd_messenger_fwd.hpp"

#include <optional>
#include <type_traits>

namespace PHARE::solver
{
template<typename FVMethodStrategy, typename MHDModel, typename PointValueApproximation,
         typename MessengerT = amr::MHDMessenger<MHDModel>>
class SSPRK4_5Integrator : public BaseMHDTimestepper<MHDModel, MessengerT>
{
    using Super = BaseMHDTimestepper<MHDModel, MessengerT>;

    using VecFieldT       = MHDModel::vecfield_type;
    using MHDStateT       = MHDModel::state_type;
    using ConservedStateT = core::MHDConservedState<VecFieldT>;
    using SSPRK54         = core::mc2011::SSPRK54;

    using Dispatchers_t = Dispatchers<MHDModel>;
    using RKUtils_t     = Dispatchers_t::RKUtils_t;

    using RKPair_t = core::RKPair<typename VecFieldT::value_type, MHDStateT>;

public:
    // The MC2011 coarse-fine transfer interpolates ghosts in time from every stage of the coarse
    // step plus its final state.
    static constexpr bool usesMC2011History
        = std::is_same_v<typename MessengerT::temporal_transfer_type,
                         amr::MC2011TemporalTransfer<MHDModel>>;

    SSPRK4_5Integrator(PHARE::initializer::PHAREDict const& dict)
        : Super{dict, /*n_extra_states=*/4}
        , euler_{dict}
        , compute_fluxes_{dict}
    {
        if constexpr (usesMC2011History)
            unp1_.emplace("unp1");
    }

    // Butcher fluxes are used to accumulate fluxes over multiple stages, the corresponding buffer
    // should only contain the fluxes over one time step. The accumulation over all substeps is
    // delegated to the solver.
    void operator()(MHDModel& model, Super::MHDStateT& state, Super::FluxT& fluxes,
                    Super::Messenger& bc, Super::level_t& level, double const currentTime,
                    double const newTime) override
    {
        auto& state1 = this->extra_states_[0];
        auto& state2 = this->extra_states_[1];
        auto& state3 = this->extra_states_[2];
        auto& state4 = this->extra_states_[3];

        this->resetButcherFluxes_(model, level);

        auto const dt = newTime - currentTime;

        // U1 = Un + w0*dt*F(Un). Fill U1's coarse-fine ghosts at its abscissa t_n + c1*dt.
        euler_(model, state, state1, fluxes, bc, level, currentTime, currentTime + SSPRK54::c1 * dt,
               core::RKStageContext{0, currentTime, dt}, SSPRK54::w0 * dt);

        this->accumulateButcherFluxes_(
            model, state.E, fluxes, level,
            (SSPRK54::w0 * SSPRK54::w11 * SSPRK54::w21 * SSPRK54::w31 * SSPRK54::w43
             + SSPRK54::w0 * SSPRK54::w11 * SSPRK54::w21 * SSPRK54::w41
             + SSPRK54::w0 * SSPRK54::w11 * SSPRK54::w40));

        // U2 = w10*Un + w11*U1 + w12*dt*F(U1)
        //
        // U2 = w10*Un + w11*U1
        RKUtils_t{level, model}(state2, RKPair_t{SSPRK54::w10, state},
                                RKPair_t{SSPRK54::w11, state1});

        // U2 = U2 + w12*dt*F(U1)
        compute_fluxes_(model, state1, fluxes, bc, level, newTime);

        euler_using_butcher_fluxes_(model, state2, state2, state1.E, fluxes, bc, level,
                                    currentTime + SSPRK54::c2 * dt,
                                    core::RKStageContext{1, currentTime, dt}, SSPRK54::w12 * dt);

        this->accumulateButcherFluxes_(model, state1.E, fluxes, level,
                                       (SSPRK54::w12 * SSPRK54::w21 * SSPRK54::w31 * SSPRK54::w43
                                        + SSPRK54::w12 * SSPRK54::w21 * SSPRK54::w41
                                        + SSPRK54::w12 * SSPRK54::w40));

        // U3 = w20*Un + w21*U2 + w22*dt*F(U2)
        //
        // U3 = w20*Un + w21*U2
        RKUtils_t{level, model}(state3, RKPair_t{SSPRK54::w20, state},
                                RKPair_t{SSPRK54::w21, state2});

        // U3 = U3 + w22*dt*F(U2)
        compute_fluxes_(model, state2, fluxes, bc, level, newTime);

        euler_using_butcher_fluxes_(model, state3, state3, state2.E, fluxes, bc, level,
                                    currentTime + SSPRK54::c3 * dt,
                                    core::RKStageContext{2, currentTime, dt}, SSPRK54::w22 * dt);

        this->accumulateButcherFluxes_(
            model, state2.E, fluxes, level,
            (SSPRK54::w22 * SSPRK54::w31 * SSPRK54::w43 + SSPRK54::w22 * SSPRK54::w41));

        // U4 = w30*Un + w31*U3 + w32*dt*F(U3)
        //
        // U4 = w30*Un + w31*U3
        RKUtils_t{level, model}(state4, RKPair_t{SSPRK54::w30, state},
                                RKPair_t{SSPRK54::w31, state3});

        // U4 = U4 + w32*dt*F(U3)
        // if we were not using butcher formulation, we would need a separate flux buffer for F(U3)
        // for the final step
        compute_fluxes_(model, state3, fluxes, bc, level, newTime);

        euler_using_butcher_fluxes_(model, state4, state4, state3.E, fluxes, bc, level,
                                    currentTime + SSPRK54::c4 * dt,
                                    core::RKStageContext{3, currentTime, dt}, SSPRK54::w32 * dt);

        this->accumulateButcherFluxes_(model, state3.E, fluxes, level,
                                       (SSPRK54::w32 * SSPRK54::w43 + SSPRK54::w42));

        compute_fluxes_(model, state4, fluxes, bc, level, newTime);

        this->accumulateButcherFluxes_(model, state4.E, fluxes, level, SSPRK54::w44);

        euler_using_butcher_fluxes_(model, state, state, this->butcherE_, this->butcherFluxes_, bc,
                                    level, newTime, core::RKStageContext{4, currentTime, dt}, dt);

        if constexpr (usesMC2011History)
            snapshotUnp1_(model, state, level);
    }

    void registerResources(MHDModel& model) override
    {
        Super::registerResources(model);
        if constexpr (usesMC2011History)
            model.resourcesManager->registerResources(*unp1_);
        euler_.registerResources(model);
        // no need to register compute_fluxes_, as it shares its resources with euler_
    }

    void allocate(MHDModel& model, SAMRAI::hier::Patch& patch,
                  double const allocateTime) const override
    {
        Super::allocate(model, patch, allocateTime);
        if constexpr (usesMC2011History)
            model.resourcesManager->allocate(*unp1_, patch, allocateTime);
        euler_.allocate(model, patch, allocateTime);
        // no need to allocate compute_fluxes_, as it shares its resources with euler_
    }

    using Super::exposeFluxes;

protected:
    void fillStageHistory_(std::optional<amr::SSPRK54HistoryNames>& history) const override
    {
        if constexpr (usesMC2011History)
        {
            using Names   = core::MHDStateNames;
            auto const& s = this->extra_states_;
            auto const stages
                = std::array{Names::of(s[0]), Names::of(s[1]), Names::of(s[2]), Names::of(s[3])};
            history = amr::SSPRK54HistoryNames{.stages = stages, .finalState = unp1_->names};
        }
    }

private:
    Euler<FVMethodStrategy, MHDModel, PointValueApproximation> euler_;
    ComputeFluxes<FVMethodStrategy, MHDModel, PointValueApproximation> compute_fluxes_;
    EulerUsingComputedFlux<MHDModel> euler_using_butcher_fluxes_;

    std::optional<ConservedStateT> unp1_;

    void snapshotUnp1_(MHDModel& model, MHDStateT& state, Super::level_t& level)
    {
        for (auto& patch : level)
        {
            auto _ = model.resourcesManager->setOnPatch(*patch, state.rho, state.rhoV, state.Etot,
                                                        state.B, *unp1_);
            unp1_->rho.copyData(state.rho);
            unp1_->rhoV.copyData(state.rhoV);
            unp1_->Etot.copyData(state.Etot);
            unp1_->B.copyData(state.B);
        }
    }
};

} // namespace PHARE::solver

#endif
