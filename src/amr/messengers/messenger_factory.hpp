#ifndef PHARE_MESSENGER_MANAGER_HPP
#define PHARE_MESSENGER_MANAGER_HPP



#include "amr/messengers/cross_model_fill_context.hpp"
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
#include <vector>

namespace PHARE::amr
{
struct MessengerDescriptor
{
    std::string coarseModel;
    std::string fineModel;
};




NO_DISCARD std::vector<MessengerDescriptor> makeDescriptors(std::vector<std::string> modelNames);




template<typename MHDModel, typename HybridModel, typename RefinementParams>
class MessengerFactory
{
    using HybridHybridMessengerStrategy_t
        = HybridHybridMessengerStrategy<HybridModel, RefinementParams>;
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
        auto const crossing = [](MessengerDescriptor const& desc) {
            return desc.coarseModel != desc.fineModel;
        };
        if (std::any_of(std::begin(descriptors_), std::end(descriptors_), crossing))
            crossModelContext_ = std::make_shared<CrossModelFillContext>();
    }


    NO_DISCARD std::shared_ptr<CrossModelFillContext> crossModelContext() const
    {
        return crossModelContext_;
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
        if (messengerName == HybridHybridMessengerStrategy_t::stratName)
        {
            auto& resourcesManager = dynamic_cast<HybridModel const&>(coarseModel).resourcesManager;

            auto messengerStrategy = std::make_unique<HybridHybridMessengerStrategy_t>(
                resourcesManager, firstLevel, crossModelContext_);

            return std::make_unique<HybridMessenger<HybridModel>>(std::move(messengerStrategy));
        }



        else if (messengerName
                 == MHDHybridMessengerStrategy<MHDModel, HybridModel, RefinementParams>::stratName)
        {
            auto& resourcesManager = dynamic_cast<HybridModel const&>(fineModel).resourcesManager;

            auto messengerStrategy
                = std::make_unique<MHDHybridMessengerStrategy<MHDModel, HybridModel,
                                                               RefinementParams>>(
                    resourcesManager, firstLevel, crossModelContext_);

            return std::make_unique<HybridMessenger<HybridModel>>(std::move(messengerStrategy));
        }




        else if (messengerName == MHDMessenger<MHDModel>::stratName)
        {
            auto& mhdResourcesManager = dynamic_cast<MHDModel const&>(coarseModel).resourcesManager;

            return std::make_unique<MHDMessenger<MHDModel>>(mhdResourcesManager, firstLevel);
        }
        else
            return {};
    }


private:
    std::vector<MessengerDescriptor> descriptors_;
    std::shared_ptr<CrossModelFillContext> crossModelContext_; // non-null iff 2 models
};

} // namespace PHARE::amr



#endif
