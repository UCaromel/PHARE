#ifndef PHARE_MHD_TEMPORAL_TRANSFER_HPP
#define PHARE_MHD_TEMPORAL_TRANSFER_HPP

#include "amr/solvers/time_integrator/rk_stage_context.hpp"
#include "amr/messengers/mhd_messenger_info.hpp"
#include "core/models/mhd_state_increment.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>

namespace PHARE::amr
{
enum class MHDTemporalTransferKind { NoCoarseFine, Linear2, MC2011_4 };

// Uniform runs must not acquire coarse-fine data or construct coarse-fine operators.
template<typename MHDModel>
class NoCoarseFineTemporalTransfer
{
public:
    static constexpr MHDTemporalTransferKind kind = MHDTemporalTransferKind::NoCoarseFine;

    explicit NoCoarseFineTemporalTransfer(
        std::shared_ptr<typename MHDModel::resources_manager_type> const&)
    {
    }

    void registerQuantities(MHDMessengerInfo const&)
    {
    }

    template<typename Level, typename Hierarchy>
    void firstStep(Level const&, std::shared_ptr<Hierarchy> const&, double const, double const) const
    {
    }

    template<typename State, typename Level>
    void prepareTarget(State&, Level const&, double const) const
    {
    }

    template<typename State, typename Level>
    void prepareTarget(State&, Level const&, double const, solver::RKStageContext const&) const
    {
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() { return std::tuple{}; }
    NO_DISCARD auto getCompileTimeResourcesViewList() const { return std::tuple{}; }
};

// Linear temporal interpolation has no storage: SolverMHD owns immutable old conserved state.
template<typename MHDModel>
class Linear2TemporalTransfer
{
public:
    static constexpr MHDTemporalTransferKind kind = MHDTemporalTransferKind::Linear2;
    using Names = core::MHDStateIncrementNames;

    explicit Linear2TemporalTransfer(
        std::shared_ptr<typename MHDModel::resources_manager_type> const&)
    {
    }

    void registerQuantities(MHDMessengerInfo const& info)
    {
        static_cast<void>(info.oldState.validatedBaseName());
        oldState_ = info.oldState;
    }

    NO_DISCARD Names const& oldStateNames() const
    {
        if (!oldState_)
            throw std::logic_error{"Linear2 temporal transfer used before old-state binding"};
        return *oldState_;
    }


    template<typename Level, typename Hierarchy>
    void firstStep(Level const&, std::shared_ptr<Hierarchy> const&, double const, double const) const
    {
    }

    template<typename State, typename Level>
    void prepareTarget(State&, Level const&, double const) const
    {
    }

    template<typename State, typename Level>
    void prepareTarget(State&, Level const&, double const, solver::RKStageContext const&) const
    {
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() { return std::tuple{}; }
    NO_DISCARD auto getCompileTimeResourcesViewList() const { return std::tuple{}; }

private:
    std::optional<Names> oldState_;
};
} // namespace PHARE::amr

#endif // PHARE_MHD_TEMPORAL_TRANSFER_HPP
