#ifndef PHARE_MESSENGER_MANAGER_HPP
#define PHARE_MESSENGER_MANAGER_HPP



#include "amr/messengers/hybrid_hybrid_messenger_strategy.hpp"
#include "amr/messengers/hybrid_messenger.hpp"
#include "amr/messengers/messenger.hpp"
#include "amr/messengers/mhd_hybrid_messenger_strategy.hpp"
#include "amr/messengers/mhd_messenger.hpp"
#include "core/def.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace PHARE::amr
{
struct MessengerDescriptor
{
    std::string coarseModel;
    std::string fineModel;
};




NO_DISCARD std::vector<MessengerDescriptor> makeDescriptors(std::vector<std::string> modelNames);


// Variadic MessengerFactory - only instantiates messenger code for provided types
// This achieves model decoupling: MHD-only builds don't compile Hybrid messenger code
template<typename MHDModel, typename HybridModel, typename... Strategies>
class MessengerFactory
{
    using IPhysicalModel = typename HybridModel::Interface;
    static_assert(std::is_same_v<typename HybridModel::Interface, typename MHDModel::Interface>,
                  "MHD and Hybrid model need to have the same interface");

public:
    static constexpr auto dimension = HybridModel::dimension;
    static_assert(dimension == MHDModel::dimension,
                  "MHDModel::dimension != HybridModel::dimension");

    MessengerFactory(std::vector<MessengerDescriptor> messengerDescriptors)
        : descriptors_{messengerDescriptors}
    {
    }

    NO_DISCARD std::optional<std::string> name(IPhysicalModel const& coarseModel,
                                               IPhysicalModel const& fineModel) const
    {
        auto finder = [&coarseModel, &fineModel](MessengerDescriptor const& desc) {
            return desc.coarseModel == coarseModel.name() && desc.fineModel == fineModel.name();
        };

        auto messenger = std::find_if(std::begin(descriptors_), std::end(descriptors_), finder);

        if (messenger != std::end(descriptors_))
        {
            return coarseModel.name() + "-" + fineModel.name();
        }
        else
        {
            return {};
        }
    }

    NO_DISCARD std::unique_ptr<IMessenger<IPhysicalModel>> create(std::string messengerName,
                                                                   IPhysicalModel const& coarseModel,
                                                                   IPhysicalModel const& fineModel,
                                                                   int const firstLevel) const
    {
        std::unique_ptr<IMessenger<IPhysicalModel>> result;
        
        // Try each strategy in the pack using fold expression
        // The comma operator ensures we stop when result is set
        ((result = tryCreate<Strategies>(messengerName, coarseModel, fineModel, firstLevel)) || ...);
        
        return result;
    }

private:
    // Generic helper that tries to create a messenger for a given strategy type
    // Uses if constexpr to dispatch to correct construction logic
    template<typename Strategy>
    std::unique_ptr<IMessenger<IPhysicalModel>> tryCreate(
        std::string const& messengerName,
        IPhysicalModel const& coarseModel,
        IPhysicalModel const& fineModel,
        int firstLevel) const
    {
        if (messengerName != Strategy::stratName)
            return {};

        if constexpr (std::is_same_v<Strategy, MHDHybridMessengerStrategy<MHDModel, HybridModel>>)
        {
            auto& resourcesManager = dynamic_cast<HybridModel const&>(fineModel).resourcesManager;
            auto messengerStrategy = std::make_unique<Strategy>(resourcesManager, firstLevel);
            return std::make_unique<HybridMessenger<HybridModel>>(std::move(messengerStrategy));
        }
        else if constexpr (std::is_base_of_v<HybridMessengerStrategy<HybridModel>, Strategy>)
        {
            auto& resourcesManager = dynamic_cast<HybridModel const&>(coarseModel).resourcesManager;
            auto messengerStrategy = std::make_unique<Strategy>(resourcesManager, firstLevel);
            return std::make_unique<HybridMessenger<HybridModel>>(std::move(messengerStrategy));
        }
        else if constexpr (std::is_same_v<Strategy, MHDMessenger<MHDModel>>)
        {
            auto& mhdResourcesManager = dynamic_cast<MHDModel const&>(coarseModel).resourcesManager;
            return std::make_unique<Strategy>(mhdResourcesManager, firstLevel);
        }
        
        return {};
    }


private:
    std::vector<MessengerDescriptor> descriptors_;
};

} // namespace PHARE::amr



#endif
