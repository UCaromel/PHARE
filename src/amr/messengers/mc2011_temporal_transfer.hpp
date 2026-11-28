#ifndef PHARE_MC2011_TEMPORAL_TRANSFER_HPP
#define PHARE_MC2011_TEMPORAL_TRANSFER_HPP

#include "amr/messengers/mhd_messenger_info.hpp"
#include "amr/messengers/mhd_temporal_transfer.hpp"
#include "amr/resources_manager/resources_manager.hpp"
#include "core/models/mhd_conserved_state.hpp"
#include "core/numerics/rk_stage_context.hpp"
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
    using level_t        = typename MHDModel::level_t;
    using VecFieldT      = typename MHDModel::vecfield_type;
    using ConservedState = core::MHDConservedState<VecFieldT>;

public:
    explicit MC2011TemporalTransfer(std::shared_ptr<typename MHDModel::resources_manager_type> rm)
        : resourcesManager_{std::move(rm)}
        , assembled_{"mc2011_assembled"}
    {
    }

    void registerQuantities(MHDMessengerInfo const& info)
    {
        // The time integrator is a runtime choice: only SSPRK(5,4) publishes the stage history
        // MC2011 interpolates. Without it the transfer stays inactive, which is valid for a
        // single-level run; MHDMessenger refuses to register a finer level in that case.
        if (!info.ssprk54History)
            return;
        active_           = true;
        auto requireOwner = [&](core::MHDStateNames const& names) {
            if (!resourcesManager_->getID(names.rho()) || !resourcesManager_->getID(names.rhoV())
                || !resourcesManager_->getID(names.Etot()) || !resourcesManager_->getID(names.B()))
                throw std::invalid_argument{
                    "MC2011 temporal history names have no registered owner"};
        };
        requireOwner(info.oldState);
        for (auto const& names : info.ssprk54History->stages)
            requireOwner(names);
        requireOwner(info.ssprk54History->finalState);
        old_.emplace(info.oldState.base);
        for (std::size_t i = 0; i < stages_.size(); ++i)
            stages_[i].emplace(info.ssprk54History->stages[i].base);
        final_.emplace(info.ssprk54History->finalState.base);
        resourcesManager_->registerResources(*old_);
        for (auto& stage : stages_)
            resourcesManager_->registerResources(*stage);
        resourcesManager_->registerResources(*final_);
    }

    [[nodiscard]] bool hasCoarseFine() const { return active_; }

    [[nodiscard]] core::MHDStateNames const& assembledStateNames() const
    {
        return assembled_.names;
    }

    template<typename Level, typename Hierarchy>
    void firstStep(Level const& level, std::shared_ptr<Hierarchy> const& hierarchy,
                   double const previousCoarseTime, double const newCoarseTime)
    {
        auto const fineLevel = static_cast<std::size_t>(level.getLevelNumber());
        if (fineLevel == 0 || !active_)
            return;
        if (!(newCoarseTime > previousCoarseTime))
            throw std::invalid_argument{"MC2011 coarse time bracket must be increasing"};
        intervals_[fineLevel] = {previousCoarseTime, newCoarseTime,
                                 hierarchy->getPatchLevel(static_cast<int>(fineLevel - 1))};
    }

    template<typename State>
    void prepareTarget(State& state, level_t const& level, core::RKStageContext const& context)
    {
        if (!active_)
            return;
        prepare_(state, level, context);
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
    void prepare_(State& state, level_t const& level, core::RKStageContext const& context)
    {
        if (context.stageIndex > core::RKStageContext::finalStage || !(context.stepDuration >= 0.0)
            || !std::isfinite(context.stepStartTime) || !std::isfinite(context.stepDuration))
            throw std::invalid_argument{"MC2011 invalid RK stage context"};
        auto const fineLevel = static_cast<std::size_t>(level.getLevelNumber());
        // a fill outside the stages (dtFine = 0, e.g. after reflux) must not erase the coarse step
        // duration recorded by the stage fills
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
    ConservedState assembled_;
    std::optional<ConservedState> old_, final_;
    std::array<std::optional<ConservedState>, 4> stages_;
    bool active_ = false;
    std::unordered_map<std::size_t, Interval> intervals_;
    std::unordered_map<std::size_t, double> durations_;
};
} // namespace PHARE::amr
#endif
