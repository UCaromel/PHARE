#ifndef PHARE_MC2011_TEMPORAL_TRANSFER_HPP
#define PHARE_MC2011_TEMPORAL_TRANSFER_HPP

#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/messengers/mhd_temporal_transfer.hpp"
#include "amr/resources_manager/resources_manager.hpp"
#include "amr/solvers/time_integrator/rk_stage_context.hpp"
#include "core/models/mhd_state_increment.hpp"
#include "core/numerics/mc2011/mc2011_reconstruction.hpp"

#include <memory>
#include <optional>
#include <array>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace PHARE::amr
{
template<typename MHDModel>
class MC2011TemporalTransfer
{
    using level_t   = typename MHDModel::level_t;
    using VecFieldT = typename MHDModel::vecfield_type;
    using Increment = core::MHDStateIncrement<VecFieldT>;

public:
    static constexpr MHDTemporalTransferKind kind = MHDTemporalTransferKind::MC2011_4;

    explicit MC2011TemporalTransfer(std::shared_ptr<typename MHDModel::resources_manager_type> rm)
        : resourcesManager_{std::move(rm)}
        , assembled_{"mc2011_assembled"}
    {
    }

    void registerQuantities(MHDMessengerInfo const& info)
    {
        auto const old  = info.oldState;
        auto const base = old.validatedBaseName();
        if (!info.ssprk54History)
            throw std::invalid_argument{"MC2011 temporal transfer requires SSPRK54 history"};
        auto requireOwner = [&](core::MHDStateIncrementNames const& names) {
            // validatedBaseName() still holds the eight-name family contract: it rejects any
            // bundle whose names do not share one base, per-component names included.
            auto const ownedBase = names.validatedBaseName();
            // What the owner *publishes*, though, is coarser than that bundle. rhoV and B are
            // TensorFields, and a TensorField is itself a resource (is_tensor_field_v feeds
            // is_resource), so ResourcesManager registers one patch data per TensorField under
            // the TensorField's own name. The per-component names are the names of the Fields
            // held inside it (core::detail::tensor_field_names) and are never keys, so asking
            // for them here could only ever fail. Ask for the keys the owner really has --
            // the same ones registerGhostComms_ hands to SAMRAI as `oldBase + "_rhoV"`.
            if (!resourcesManager_->getID(names.rho)
                || !resourcesManager_->getID(ownedBase + "_rhoV")
                || !resourcesManager_->getID(names.Etot)
                || !resourcesManager_->getID(ownedBase + "_B"))
                throw std::invalid_argument{
                    "MC2011 temporal history names have no registered owner"};
        };
        requireOwner(old);
        for (auto const& names : info.ssprk54History->stages)
            requireOwner(names);
        requireOwner(info.ssprk54History->finalState);
        old_.emplace(old);
        for (std::size_t i = 0; i < stages_.size(); ++i)
            stages_[i].emplace(info.ssprk54History->stages[i]);
        final_.emplace(info.ssprk54History->finalState);
        resourcesManager_->registerResources(*old_);
        for (auto& stage : stages_)
            resourcesManager_->registerResources(*stage);
        resourcesManager_->registerResources(*final_);
        // Built once, here, where assembled_'s names are already known -- assembledStateNames()
        // then just hands out a const& to this member instead of re-emplacing a mutable optional
        // on every call, which would invalidate any reference a caller is still holding.
        assembledNames_ = core::MHDStateIncrementNames{assembled_};
        static_cast<void>(base);
    }

    [[nodiscard]] core::MHDStateIncrementNames const& assembledStateNames() const
    {
        return assembledNames_;
    }

    template<typename Level, typename Hierarchy>
    void firstStep(Level const& level, std::shared_ptr<Hierarchy> const& hierarchy,
                   double const previousCoarseTime, double const newCoarseTime)
    {
        auto const fineLevel = static_cast<std::size_t>(level.getLevelNumber());
        if (fineLevel == 0)
            return;
        if (!(newCoarseTime > previousCoarseTime))
            throw std::invalid_argument{"MC2011 coarse time bracket must be increasing"};
        intervals_[fineLevel] = {previousCoarseTime, newCoarseTime,
                                 hierarchy->getPatchLevel(static_cast<int>(fineLevel - 1))};
    }

    template<typename State>
    void prepareTarget(State& state, level_t const& level, double const fillTime)
    {
        prepare_(state, level, fillTime, solver::RKStageContext{4, fillTime, 0.0});
    }

    template<typename State>
    void prepareTarget(State& state, level_t const& level, double const fillTime,
                       solver::RKStageContext const& context)
    {
        prepare_(state, level, fillTime, context);
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() { return std::forward_as_tuple(assembled_); }
    NO_DISCARD auto getCompileTimeResourcesViewList() const
    {
        return std::forward_as_tuple(assembled_);
    }

private:
    struct Interval
    {
        double previous, next;
        std::shared_ptr<level_t> coarse;
    };

    template<typename State>
    void prepare_(State& state, level_t const& level, double const fillTime,
                  solver::RKStageContext const& context)
    {
        if (context.stageIndex > 4 || !(context.stepDuration >= 0.0)
            || !std::isfinite(context.stepStartTime) || !std::isfinite(context.stepDuration)
            || !std::isfinite(fillTime))
            throw std::invalid_argument{"MC2011 invalid RK stage context"};
        auto const fineLevel = static_cast<std::size_t>(level.getLevelNumber());
        // A post-reflux ordinary fill has dtFine=0; it must not erase the exact
        // sweep duration persisted by one of the five RK-stage fills.
        if (context.stepDuration > 0.0)
            durations_[fineLevel] = context.stepDuration;
        if (fineLevel != 0)
        {
            auto const& interval = intervals_.at(fineLevel);
            double const chi
                = (context.stepStartTime - interval.previous) / (interval.next - interval.previous);
            if (!std::isfinite(chi) || !(chi >= -1e-12 && chi <= 1.0 + 1e-12))
                throw std::invalid_argument{"MC2011 fill is outside coarse time bracket"};
            auto const duration = durations_.at(fineLevel - 1);
            if (!(duration > 0.0))
                throw std::logic_error{"MC2011 coarse RK duration was not persisted"};
            assemble_(*interval.coarse, duration, chi, context.stepDuration, context.stageIndex);
        }
        copyInterior_(state, level);
        static_cast<void>(fillTime);
    }

    template<typename State>
    void copyInterior_(State& state, level_t const& level)
    {
        for (auto& patch : level)
        {
            auto const& layout = layoutFromPatch<typename MHDModel::gridlayout_type>(*patch);
            auto _ = resourcesManager_->setOnPatch(*patch, state.rho, state.rhoV, state.Etot,
                                                   state.B, assembled_);
            copy_(layout, state.rho, assembled_.rho);
            copy_(layout, state.Etot, assembled_.Etot);
            for (auto c : {core::Component::X, core::Component::Y, core::Component::Z})
            {
                copy_(layout, state.rhoV(c), assembled_.rhoV(c));
                copy_(layout, state.B(c), assembled_.B(c));
            }
        }
    }

    template<typename Layout, typename Field>
    static void copy_(Layout const& layout, Field const& from, Field& to)
    {
        layout.evalOnBox(to, [&](auto const&... i) { to(i...) = from(i...); });
    }

    void assemble_(level_t& coarse, double const dtCoarse, double const chi, double const dtFine,
                   std::size_t const stageIndex)
    {
        auto assembleField = [&](auto const& y0, auto const& y1, auto const& y2, auto const& y3,
                                 auto const& y4, auto const& yn, auto& out, auto const& layout) {
            layout.evalOnGhostBox(out, [&](auto const&... i) {
                auto const k      = core::mc2011::backSolve(y0(i...), y1(i...), y2(i...), y3(i...),
                                                            y4(i...), yn(i...), dtCoarse);
                auto const [a, b] = core::mc2011::splitTerms(k, 1.0 / (dtCoarse * dtCoarse));
                out(i...) = core::mc2011::reconstruct(y0(i...), k, a, b, chi, dtCoarse, dtFine,
                                                      stageIndex);
            });
        };
        for (auto& patch : coarse)
        {
            auto const& layout = layoutFromPatch<typename MHDModel::gridlayout_type>(*patch);
            auto _ = resourcesManager_->setOnPatch(*patch, *old_, *stages_[0], *stages_[1],
                                                   *stages_[2], *stages_[3], *final_, assembled_);
            assembleField(old_->rho, stages_[0]->rho, stages_[1]->rho, stages_[2]->rho,
                          stages_[3]->rho, final_->rho, assembled_.rho, layout);
            assembleField(old_->Etot, stages_[0]->Etot, stages_[1]->Etot, stages_[2]->Etot,
                          stages_[3]->Etot, final_->Etot, assembled_.Etot, layout);
            for (auto c : {core::Component::X, core::Component::Y, core::Component::Z})
            {
                assembleField(old_->rhoV(c), stages_[0]->rhoV(c), stages_[1]->rhoV(c),
                              stages_[2]->rhoV(c), stages_[3]->rhoV(c), final_->rhoV(c),
                              assembled_.rhoV(c), layout);
                assembleField(old_->B(c), stages_[0]->B(c), stages_[1]->B(c), stages_[2]->B(c),
                              stages_[3]->B(c), final_->B(c), assembled_.B(c), layout);
            }
        }
    }

    std::shared_ptr<typename MHDModel::resources_manager_type> resourcesManager_;
    Increment assembled_;
    core::MHDStateIncrementNames assembledNames_;
    std::optional<Increment> old_, final_;
    std::array<std::optional<Increment>, 4> stages_;
    std::unordered_map<std::size_t, Interval> intervals_;
    std::unordered_map<std::size_t, double> durations_;
};
} // namespace PHARE::amr
#endif
