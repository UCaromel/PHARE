#ifndef PHARE_MHD_TEMPORAL_TRANSFER_HPP
#define PHARE_MHD_TEMPORAL_TRANSFER_HPP

#include "amr/messengers/mhd_messenger_info.hpp"
#include "core/def.hpp"
#include "core/numerics/rk_stage_context.hpp"

#include <memory>
#include <tuple>

namespace PHARE::amr
{
// Second-order coarse-fine transfer: MHDMessenger interpolates linearly in time between the
// solver's old state (MHDMessengerInfo::oldState) and the model state, so there is nothing to
// store or prepare here.
template<typename MHDModel>
class Linear2TemporalTransfer
{
public:
    explicit Linear2TemporalTransfer(
        std::shared_ptr<typename MHDModel::resources_manager_type> const&)
    {
    }

    void registerQuantities(MHDMessengerInfo const&) {}

    NO_DISCARD bool hasCoarseFine() const { return true; }

    template<typename Level, typename Hierarchy>
    void firstStep(Level const&, std::shared_ptr<Hierarchy> const&, double const,
                   double const) const
    {
    }

    template<typename State, typename Level>
    void prepareTarget(State&, Level const&, core::RKStageContext const&) const
    {
    }

    NO_DISCARD auto getCompileTimeResourcesViewList() { return std::tuple{}; }
    NO_DISCARD auto getCompileTimeResourcesViewList() const { return std::tuple{}; }
};
} // namespace PHARE::amr

#endif // PHARE_MHD_TEMPORAL_TRANSFER_HPP
