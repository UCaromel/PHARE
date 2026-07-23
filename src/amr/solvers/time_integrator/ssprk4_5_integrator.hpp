#ifndef PHARE_CORE_NUMERICS_SSPRK4_5_INTEGRATOR_HPP
#define PHARE_CORE_NUMERICS_SSPRK4_5_INTEGRATOR_HPP

#include "initializer/data_provider.hpp"
#include "amr/solvers/time_integrator/base_mhd_timestepper.hpp"
#include "amr/solvers/time_integrator/compute_fluxes.hpp"
#include "amr/solvers/time_integrator/euler_using_computed_flux.hpp"
#include "amr/solvers/solver_mhd_field_evolvers.hpp"
#include "amr/solvers/time_integrator/euler.hpp"
#include "core/numerics/time_integrator_utils.hpp"
#include "core/numerics/mc2011/mc2011_reconstruction.hpp"
#include "core/models/mhd_state_increment.hpp"

namespace PHARE::solver
{
template<typename FVMethodStrategy, typename MHDModel>
class SSPRK4_5Integrator : public BaseMHDTimestepper<MHDModel>
{
    using Super = BaseMHDTimestepper<MHDModel>;

    using level_t     = typename MHDModel::level_t;
    using FieldT      = typename MHDModel::field_type;
    using VecFieldT   = typename MHDModel::vecfield_type;
    using GridLayoutT = typename MHDModel::gridlayout_type;
    using MHDStateT   = typename MHDModel::state_type;
    using KIncrementT = core::MHDStateIncrement<VecFieldT>;

    using Dispatchers_t = Dispatchers<MHDModel>;
    using RKUtils_t     = Dispatchers_t::RKUtils_t;

    using RKPair_t = core::RKPair<typename VecFieldT::value_type, MHDStateT>;

public:
    SSPRK4_5Integrator(PHARE::initializer::PHAREDict const& dict)
        : Super{dict}
        , euler_{dict}
        , compute_fluxes_{dict}
    {
    }

    // Butcher fluxes are used to accumulate fluxes over multiple stages, the corresponding buffer
    // should only contain the fluxes over one time step. The accumulation over all substeps is
    // delegated to the solver.
    void operator()(MHDModel& model, auto& state, auto& fluxes, auto& bc, level_t& level,
                    double const currentTime, double const newTime)
    {
        this->resetButcherFluxes_(model, level);

        auto const dt = newTime - currentTime;

        // MC2011: chi is fixed for this whole RK step (all 5 stage/final ghost-fills
        // below share it, per full-derivation.md S5.4 / design decision 4). dtFine (the
        // Delta t_f of that same formula) is this level's own full step size, `dt` --
        // NOT the individual Shu-Osher sub-increments (w0_*dt, w12_*dt, ...), since the
        // per-stage SSPRK54::gamma tables (Table S5.1) already encode the intra-step
        // position. stageIndex 0..3 = state1_..state4_ (Y2..Y5), 4 = the final blend.
        double const chi = bc.mc2011Chi(level, currentTime);

        // U1 = Un + w0_*dt*F(Un). Fill U1's coarse-fine ghosts at its abscissa t_n + c1_*dt.
        euler_(model, state, state1_, fluxes, 0, chi, dt, bc, level, currentTime,
               currentTime + c1_ * dt, w0_ * dt);

        this->accumulateButcherFluxes_(
            model, state.E, fluxes, level,
            (w0_ * w11_ * w21_ * w31_ * w43_ + w0_ * w11_ * w21_ * w41_ + w0_ * w11_ * w40_));

        // U2 = w10_*Un + w11_*U1 + w12_*dt*F(U1)
        //
        // U2 = w10_Un + w11_*U1
        RKUtils_t{level, model}(state2_, RKPair_t{w10_, state}, RKPair_t{w11_, state1_});

        // U2 = U2 + w12_*dt*F(U1)
        compute_fluxes_(model, state1_, fluxes, bc, level, newTime);

        euler_using_butcher_fluxes_(model, state2_, state2_, state1_.E, fluxes, 1, chi, dt, bc,
                                    level, currentTime + c2_ * dt, w12_ * dt);

        this->accumulateButcherFluxes_(
            model, state1_.E, fluxes, level,
            (w12_ * w21_ * w31_ * w43_ + w12_ * w21_ * w41_ + w12_ * w40_));

        // U3 = w20_*Un + w21_*U2 + w22_*dt*F(U2)
        //
        // U3 = w20_*Un + w21_*U2
        RKUtils_t{level, model}(state3_, RKPair_t{w20_, state}, RKPair_t{w21_, state2_});

        // U3 = U3 + w22_*dt*F(U2)
        compute_fluxes_(model, state2_, fluxes, bc, level, newTime);

        euler_using_butcher_fluxes_(model, state3_, state3_, state2_.E, fluxes, 2, chi, dt, bc,
                                    level, currentTime + c3_ * dt, w22_ * dt);

        this->accumulateButcherFluxes_(model, state2_.E, fluxes, level,
                                       (w22_ * w31_ * w43_ + w22_ * w41_));

        // U4 = w30_*Un + w31_*U3 + w32_*dt*F(U3)
        //
        // U4 = w30_*Un + w31_*U3
        RKUtils_t{level, model}(state4_, RKPair_t{w30_, state}, RKPair_t{w31_, state3_});

        // U4 = U4 + w32_*dt*F(U3)
        // if we were not using butcher formulation, we would need a separate flux buffer for F(U3)
        // for the final step
        compute_fluxes_(model, state3_, fluxes, bc, level, newTime);

        euler_using_butcher_fluxes_(model, state4_, state4_, state3_.E, fluxes, 3, chi, dt, bc,
                                    level, currentTime + c4_ * dt, w32_ * dt);

        this->accumulateButcherFluxes_(model, state3_.E, fluxes, level, (w32_ * w43_ + w42_));

        // F(U4) feeds only the butcher accumulation below -- there is no k5
        // extraction anymore (the messenger back-solves every k from the persisted
        // states, see core::mc2011::backSolve), so state4_ stays Y5 for the whole
        // coarse interval: fine-level assemblies consume it long after this sweep.
        compute_fluxes_(model, state4_, fluxes, bc, level, newTime);

        this->accumulateButcherFluxes_(model, state4_.E, fluxes, level, w44_);

        // Final blend (Un -> Un+1): stageIndex 4, same chi as the 4 stages above.
        euler_using_butcher_fluxes_(model, state, state, this->butcherE_, this->butcherFluxes_, 4,
                                    chi, dt, bc, level, newTime, dt);

        // Un+1 = w40_*U2 + w41_*U3 + w42_*F(U3) + w43_*U4 + w44_*dt*F(U4)

        // Snapshot Un+1 for the messenger's back-solve (derivation.md S5.1). Taken
        // after the blend's own ghost fill just above, so the copy (ghost-inclusive,
        // like the messenger's prepareStep old-state snapshots) carries a valid
        // ghost box -- assembleMC2011_ evaluates the back-solve on evalOnGhostBox.
        snapshotUnp1_(model, state, level);
    }

    void registerResources(MHDModel& model)
    {
        Super::registerResources(model);
        model.resourcesManager->registerResources(state1_);
        model.resourcesManager->registerResources(state2_);
        model.resourcesManager->registerResources(state3_);
        model.resourcesManager->registerResources(state4_);
        model.resourcesManager->registerResources(unp1_);
        euler_.registerResources(model);
        // probably we should have the same resources for euler and compute_fluxes
        // compute_fluxes_.registerResources(model);
    }

    void allocate(MHDModel& model, auto& patch, double const allocateTime) const
    {
        Super::allocate(model, patch, allocateTime);
        model.resourcesManager->allocate(state1_, patch, allocateTime);
        model.resourcesManager->allocate(state2_, patch, allocateTime);
        model.resourcesManager->allocate(state3_, patch, allocateTime);
        model.resourcesManager->allocate(state4_, patch, allocateTime);
        model.resourcesManager->allocate(unp1_, patch, allocateTime);
        euler_.allocate(model, patch, allocateTime);
        // probably we should have the same resources for euler and compute_fluxes
        // compute_fluxes_.allocate(model, patch, allocateTime);
    }

    void fillMessengerInfo(auto& info) const
    {
        auto fill_info = [&](auto& state) {
            info.ghostDensity.push_back(state.rho.name());
            info.ghostVelocity.push_back(state.V.name());
            info.ghostPressure.push_back(state.P.name());
            info.ghostMomentum.push_back(state.rhoV.name());
            info.ghostTotalEnergy.push_back(state.Etot.name());
            info.ghostElectric.push_back(state.E.name());
            info.ghostMagnetic.push_back(state.B.name());
        };

        fill_info(state1_);
        fill_info(state2_);
        fill_info(state3_);
        fill_info(state4_);

        euler_.fillMessengerInfo(info);
        // we should have the same resources for euler and compute_fluxes
        // compute_fluxes_.fillMessengerInfo(info);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList()
    {
        return std::tuple_cat(Super::getCompileTimeResourcesViewList(),
                              std::forward_as_tuple(state1_, state2_, state3_, state4_, unp1_));
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::tuple_cat(Super::getCompileTimeResourcesViewList(),
                              std::forward_as_tuple(state1_, state2_, state3_, state4_, unp1_));
    }

    using Super::exposeFluxes;

    // Accessor mirroring Super::exposeFluxes() -- exposes the persisted post-stage
    // states (Butcher stages Y2..Y5) plus the final-blend snapshot unp1_, the
    // inputs the messenger's MC2011 assembly back-solves the stage derivatives
    // from (core::mc2011::backSolve).
    auto exposeStageStates()
    {
        return std::forward_as_tuple(state1_, state2_, state3_, state4_, unp1_);
    }

    auto exposeStageStates() const
    {
        return std::forward_as_tuple(state1_, state2_, state3_, state4_, unp1_);
    }

private:
    // All SSPRK(5,4)/MC2011 coefficients live in core::mc2011::SSPRK54 (single
    // source of truth, shared with MHDMessenger's Tier 2/3 assembly and locked
    // by tests/amr/solvers/time_integrator/test_mc2011_kernels.cpp). Aliased
    // here to keep the Shu-Osher stage sweep above readable.
    using SSPRK54_ = core::mc2011::SSPRK54;

    static constexpr double w0_{SSPRK54_::w0};
    static constexpr double w10_{SSPRK54_::w10};
    static constexpr double w11_{SSPRK54_::w11};
    static constexpr double w12_{SSPRK54_::w12};
    static constexpr double w20_{SSPRK54_::w20};
    static constexpr double w21_{SSPRK54_::w21};
    static constexpr double w22_{SSPRK54_::w22};
    static constexpr double w30_{SSPRK54_::w30};
    static constexpr double w31_{SSPRK54_::w31};
    static constexpr double w32_{SSPRK54_::w32};
    static constexpr double w40_{SSPRK54_::w40};
    static constexpr double w41_{SSPRK54_::w41};
    static constexpr double w42_{SSPRK54_::w42};
    static constexpr double w43_{SSPRK54_::w43};
    static constexpr double w44_{SSPRK54_::w44};

    // Stage abscissae c_i: the physical time t_n + c_i*dt that each
    // intermediate node approximates, used to fill coarse-fine ghosts at the
    // node's own time.
    static constexpr double c1_{SSPRK54_::c[0]}; // state1_
    static constexpr double c2_{SSPRK54_::c[1]}; // state2_
    static constexpr double c3_{SSPRK54_::c[2]}; // state3_
    static constexpr double c4_{SSPRK54_::c[3]}; // state4_

    Euler<FVMethodStrategy, MHDModel> euler_;
    ComputeFluxes<FVMethodStrategy, MHDModel> compute_fluxes_;
    EulerUsingComputedFlux<MHDModel> euler_using_butcher_fluxes_;

    MHDStateT state1_{"state1"};
    MHDStateT state2_{"state2"};
    MHDStateT state3_{"state3"};
    MHDStateT state4_{"state4"};

    // Final-blend snapshot Un+1 (conserved quads only): the fifth persisted input
    // of the messenger's back-solve. The blend writes INTO `state` in place, so
    // without this copy Un+1 would be gone by the time fine levels assemble.
    KIncrementT unp1_{"unp1"};

    // Ghost-inclusive copy (Field::copyData copies the whole allocation), taken
    // right after the final blend's own ghost fill.
    void snapshotUnp1_(MHDModel& model, MHDStateT& state, level_t& level)
    {
        for (auto& patch : level)
        {
            auto _ = model.resourcesManager->setOnPatch(*patch, state.rho, state.rhoV, state.Etot,
                                                        state.B, unp1_);
            unp1_.rho.copyData(state.rho);
            unp1_.rhoV.copyData(state.rhoV);
            unp1_.Etot.copyData(state.Etot);
            unp1_.B.copyData(state.B);
        }
    }
};

} // namespace PHARE::solver

#endif
